#include "turbo_ocr/recognition/gpu_word_placer.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "turbo_ocr/common/cuda/cuda_check.h"
#include "turbo_ocr/common/cuda/cuda_ptr.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/recognition/word_boxes.h"
#include "gpu_word_placer_internal.h"

namespace turbo_ocr::recognition {

namespace {

// Crop pixels per GPU pass: bounds the scratch (~11 bytes a pixel, ~70 MB a
// placer) whatever the page. The widest crop (16384 + margins by 320 rows)
// fits, so no pass is larger.
constexpr int kChunkPixels = 6 << 20;
static_assert((detail::kMaxLineW + detail::kMaxLineH) * 2 * detail::kMaxLineH <= kChunkPixels);

// Component, column-histogram and bleed-candidate buffers: their first sizes,
// and past what sizes a page's growth is given back afterwards, so a rare
// page (a scan full of speckle) does not keep its buffers in every placer.
constexpr std::size_t kFirstComps = 1 << 16, kKeepComps = 1 << 18;
constexpr std::size_t kFirstHist = 1 << 18, kKeepHist = 1 << 20;
constexpr std::size_t kKeepCands = 1 << 12;

template <typename T>
void grow(CudaPtr<T> &p, std::size_t &cap, std::size_t need) {
  if (need <= cap) return;
  cap = std::max(need, cap + cap / 2);
  p.reset(cap);
}

// What one GPU pass returns for its lines.
struct ChunkResult {
  std::vector<gpu::LineOut> lines;
  std::vector<gpu::CompStats> comps;
  std::vector<int> hoff;
  std::vector<int> colhist;
};

// A line placed from the GPU pass (or handed back to the CPU code).
struct LineJob {
  std::size_t result = 0, box = 0;
  detail::LinePlan plan;
  int chunk = -1, index = 0;  // its ChunkResult and line within it
  bool cpu = false;           // placed by the CPU code
};

// The lines of `todo` planned (host work only): their jobs, and which of
// them the GPU pass places.
struct Planned {
  std::shared_ptr<std::vector<LineJob>> jobs = std::make_shared<std::vector<LineJob>>();
  std::vector<std::size_t> gpu;  // indices into jobs, in order
};

Planned plan_jobs(cv::Size page, const std::vector<Box> &boxes,
                  const std::vector<std::vector<CtcWord>> &words,
                  const std::vector<std::pair<std::size_t, std::size_t>> &todo,
                  bool whole_line_fallback = false) {
  Planned out;
  out.jobs->reserve(todo.size());
  for (const auto &[ri, bi] : todo) {
    if (words[bi].empty()) continue;  // locate_words places nothing
    LineJob j;
    j.result = ri;
    j.box = bi;
    j.plan = detail::plan_line(boxes[bi], page, boxes);
    j.cpu = !j.plan.runs_ok ||
            (whole_line_fallback &&
             std::any_of(words[bi].begin(), words[bi].end(), [](const CtcWord &word) {
               return word.first < 0.0f || word.last > 1.0f;
             }));
    if (!j.cpu) out.gpu.push_back(out.jobs->size());
    out.jobs->push_back(std::move(j));
  }
  return out;
}

// Components of one line of a GPU pass, in canonical order (index 0 = background).
detail::Components components_of(const ChunkResult &r, int line, int comp_end,
                                 const detail::LinePlan &plan) {
  const int first = r.lines[static_cast<std::size_t>(line)].comp_first;
  const int n = comp_end - first + 1;
  detail::Components cs;
  cs.n = n;
  const auto nsz = static_cast<std::size_t>(n);
  for (auto *v : {&cs.area, &cs.x0, &cs.y0, &cs.width, &cs.height, &cs.cx, &cs.in_core,
                  &cs.claimed_px, &cs.band_px, &cs.boxed_px})
    v->assign(nsz, 0);
  cs.core_x0.assign(nsz, plan.w + 2 * plan.m);
  cs.core_x1.assign(nsz, -1);
  for (int k = 1; k < n; ++k) {
    const gpu::CompStats &s = r.comps[static_cast<std::size_t>(first + k - 1)];
    const auto ku = static_cast<std::size_t>(k);
    cs.area[ku] = s.area;
    cs.x0[ku] = s.x0;
    cs.y0[ku] = s.y0;
    cs.width[ku] = s.x1 - s.x0 + 1;
    cs.height[ku] = s.y1 - s.y0 + 1;
    // connectedComponentsWithStats' centroid: the exact sum over the area.
    cs.cx[ku] = static_cast<int>(static_cast<double>(s.sumx) / static_cast<double>(s.area));
    cs.in_core[ku] = s.in_core;
    if (s.in_core > 0) {
      cs.core_x0[ku] = s.core_x0;
      cs.core_x1[ku] = s.core_x1;
    }
    cs.claimed_px[ku] = s.claimed;
    cs.band_px[ku] = s.band;
    cs.boxed_px[ku] = s.boxed;
  }
  return cs;
}

// Placers for deferred placement, shared by the threads that finish requests:
// few, each with its own stream, so their scratch stays bounded (~70 MB a
// placer, within the VRAM headroom pool_sizing.h leaves; each pipeline's own
// placer is within its runtime reserve).
class PlacerPool {
public:
  struct Slot {
    GpuWordPlacer placer;
    cudaStream_t stream = nullptr;
    int dev = 0;
  };
  class Lease {
  public:
    Lease(PlacerPool &pool, Slot *slot) : pool_(pool), slot_(slot) {}
    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;
    ~Lease() { release(); }
    Slot &operator*() const { return *slot_; }
    Slot *operator->() const { return slot_; }
    void release() {
      if (slot_) pool_.put(slot_);
      slot_ = nullptr;
    }

  private:
    PlacerPool &pool_;
    Slot *slot_;
  };

  static PlacerPool &instance() {
    // Never destroyed: no CUDA calls while the process exits.
    static auto *pool = new PlacerPool;
    return *pool;
  }

  Lease acquire(int dev) {
    std::unique_lock lock(m_);
    for (;;) {
      for (auto it = free_.begin(); it != free_.end(); ++it)
        if ((*it)->dev == dev) {
          Slot *s = *it;
          free_.erase(it);
          return {*this, s};
        }
      if (count_on(dev) < kSlots) {
        auto slot = std::make_unique<Slot>();
        slot->dev = dev;
        CUDA_CHECK(cudaStreamCreateWithFlags(&slot->stream, cudaStreamNonBlocking));
        all_.push_back(std::move(slot));
        return {*this, all_.back().get()};
      }
      cv_.wait(lock);
    }
  }

private:
  static constexpr int kSlots = 2;
  int count_on(int dev) const {
    return static_cast<int>(std::count_if(all_.begin(), all_.end(),
                                          [dev](const auto &s) { return s->dev == dev; }));
  }
  void put(Slot *s) {
    {
      std::lock_guard lock(m_);
      free_.push_back(s);
    }
    cv_.notify_one();
  }
  std::mutex m_;
  std::condition_variable cv_;
  std::vector<std::unique_ptr<Slot>> all_;
  std::vector<Slot *> free_;
};

// Device memory the page copies waiting for placement may hold together
// (MAX_IMAGE_PIXELS_MP lets one page reach ~384 MB); a page past it is placed
// at once instead.
constexpr std::size_t kWaitingCopyBytes = std::size_t{128} << 20;

// A device copy of a page for deferred placement; stream-ordered memory, freed
// after the placement (or, when the Finish never runs, when dropped), its
// share of kWaitingCopyBytes held until then.
struct PageCopy {
  GpuImage img;
  int dev = 0;
  std::size_t bytes = 0;

  static std::atomic<std::size_t> &waiting() {
    static std::atomic<std::size_t> n{0};
    return n;
  }
  // Takes `n` bytes of the budget; false when they do not fit.
  bool reserve(std::size_t n) {
    std::size_t cur = waiting().load();
    do {
      if (cur + n > kWaitingCopyBytes) return false;
    } while (!waiting().compare_exchange_weak(cur, cur + n));
    bytes = n;
    return true;
  }
  void free_on(cudaStream_t stream) {
    if (img.data) {
      cudaSetDevice(dev);
      cudaFreeAsync(img.data, stream);
      img.data = nullptr;
    }
    waiting() -= bytes;
    bytes = 0;
  }
  ~PageCopy() { free_on(nullptr); }
};

} // namespace

namespace gpu {

LineDesc line_desc(const detail::LinePlan &p, int off) {
  LineDesc d{};
  for (int i = 0; i < 9; ++i) d.M[i] = static_cast<double>(p.to_page.val[i]);
  d.W = p.W; d.H = p.H; d.w = p.w; d.h = p.h; d.m = p.m;
  d.off = off;
  d.sub_x = p.sub ? p.sub->x : -1;
  d.sub_y = p.sub ? p.sub->y : 0;
  // warpPerspective's blocks: 16 rows (fewer for short crops), and the columns
  // that make 1024 pixels.
  const int bh0 = std::min(16, p.H);
  d.bw0 = std::min(1024 / bh0, p.W);
  d.own_mid = p.own_mid;
  return d;
}

} // namespace gpu

struct GpuWordPlacer::Impl {
  CudaPtr<gpu::LineDesc> lines;
  CudaPtr<gpu::NbDesc> nbs;
  CudaPtr<std::int16_t> runs;
  CudaPtr<gpu::Tile> tiles;
  CudaPtr<unsigned> root_mask;
  CudaPtr<std::uint8_t> ink;
  CudaPtr<std::uint16_t> claims;
  CudaPtr<int> labels, cid, hist256, colw, hoff, colhist, tile_roots, tile_base, dark_tiles;
  CudaPtr<gpu::LineOut> line_out;
  CudaPtr<gpu::CompStats> comps;
  CudaPtr<gpu::Totals> totals;
  CudaPtr<unsigned char> scan_tmp;
  CudaPtr<gpu::CandPar> cands;
  CudaPtr<int> bytag, prof;
  std::size_t lines_cap = 0, nbs_cap = 0, runs_cap = 0, tiles_cap = 0, pixel_cap = 0,
              hist256_cap = 0, line_out_cap = 0, comp_cap = 0, colhist_cap = 0, scan_cap = 0,
              cand_cap = 0;
  std::size_t comp_want = kFirstComps, hist_want = kFirstHist;

  Impl() : totals(1) {}

  // place() with the lines planned; `cpu_lines` gets the lines handed back.
  Finish place(const GpuImage &page, cudaStream_t stream, Planned planned,
               const std::vector<Box> &boxes, const std::vector<std::vector<CtcWord>> &words,
               const cv::Mat *host_page, std::size_t &cpu_lines,
               Fallback fallback = Fallback::CpuPixels);

  // One GPU pass over `jobs[first, last)`, appended to `out`.
  void run(const gpu::PageView &page, cudaStream_t stream, std::vector<LineJob> &jobs,
           const std::vector<std::size_t> &idx, std::size_t first, std::size_t last,
           ChunkResult &out) {
    std::vector<gpu::LineDesc> ld;
    std::vector<gpu::NbDesc> nd;
    std::vector<std::int16_t> rd;
    std::vector<gpu::Tile> td;
    int px = 0;
    for (std::size_t k = first; k < last; ++k) {
      const detail::LinePlan &p = jobs[idx[k]].plan;
      gpu::LineDesc d = gpu::line_desc(p, px);
      d.nb_first = static_cast<int>(nd.size());
      d.nb_count = static_cast<int>(p.neighbours.size());
      for (const auto &nb : p.neighbours) {
        gpu::NbDesc n{};
        n.claims = nb.claims ? 1 : 0;
        n.wx0 = nb.wx0; n.wy0 = nb.wy0; n.wx1 = nb.wx1; n.wy1 = nb.wy1;
        n.lx = nb.left.x; n.ly = nb.left.y; n.slope = nb.slope;
        n.run_first = static_cast<int>(rd.size() / 4);
        n.run_y0 = nb.runs.y0;
        n.run_rows = static_cast<int>(nb.runs.rows.size());
        for (const auto &row : nb.runs.rows) rd.insert(rd.end(), row.begin(), row.end());
        nd.push_back(n);
      }
      const int line = static_cast<int>(ld.size());
      d.tile_first = static_cast<int>(td.size());
      for (int s = 0; s < p.W * p.H; s += gpu::kTilePixels)
        td.push_back({line, s, std::min(gpu::kTilePixels, p.W * p.H - s)});
      jobs[idx[k]].index = line;
      ld.push_back(d);
      px += p.W * p.H;
    }
    if (nd.empty()) nd.push_back({});
    if (rd.empty()) rd.assign(4, 0);

    grow(lines, lines_cap, ld.size());
    grow(nbs, nbs_cap, nd.size());
    grow(runs, runs_cap, rd.size());
    if (td.size() > tiles_cap) {
      grow(tiles, tiles_cap, td.size());
      tile_roots.reset(tiles_cap);
      tile_base.reset(tiles_cap);
      dark_tiles.reset(tiles_cap);
      root_mask.reset(tiles_cap * gpu::kTileWords);
    }
    ensure_pixels(static_cast<std::size_t>(px));
    grow(hist256, hist256_cap, ld.size() * 256);
    grow(line_out, line_out_cap, ld.size());
    ensure_comps(comp_want, hist_want, static_cast<std::size_t>(px));

    CUDA_CHECK(cudaMemcpyAsync(lines.get(), ld.data(), ld.size() * sizeof(ld[0]),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(nbs.get(), nd.data(), nd.size() * sizeof(nd[0]),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(runs.get(), rd.data(), rd.size() * sizeof(rd[0]),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(tiles.get(), td.data(), td.size() * sizeof(td[0]),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(hist256.get(), 0, ld.size() * 256 * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(line_out.get(), 0, ld.size() * sizeof(gpu::LineOut), stream));
    CUDA_CHECK(cudaMemsetAsync(totals.get(), 0, sizeof(gpu::Totals), stream));

    const int nl = static_cast<int>(ld.size()), nt = static_cast<int>(td.size());
    const auto totals_now = [&] {
      gpu::Totals t{};
      CUDA_CHECK(cudaMemcpyAsync(&t, totals.get(), sizeof(t), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      return t;
    };
    // Components and their statistics, with buffers grown to fit.
    const auto stats = [&] {
      gpu::stage_stats(buffers(nl, nt, px), stream);
      gpu::Totals t = totals_now();
      if (t.comps_overflow) {
        comp_want = static_cast<std::size_t>(t.comps) + t.comps / 4 + 1;
        ensure_comps(comp_want, hist_want, static_cast<std::size_t>(px));
        gpu::stage_stats(buffers(nl, nt, px), stream);
        t = totals_now();
      }
      return t;
    };
    gpu::stage_pixels(page, buffers(nl, nt, px), stream);
    gpu::Totals t = stats();
    if (t.cands > 0) {
      ensure_cands(static_cast<std::size_t>(t.cands));
      gpu::stage_split(buffers(nl, nt, px), stream);
      if (totals_now().any_split) {
        gpu::stage_relabel(buffers(nl, nt, px), stream);
        t = stats();
      }
    }
    gpu::stage_hist(buffers(nl, nt, px), stream);
    t = totals_now();
    if (t.hist_overflow) {
      hist_want = static_cast<std::size_t>(t.hist) + t.hist / 4 + 1;
      ensure_comps(comp_want, hist_want, static_cast<std::size_t>(px));
      gpu::stage_hist(buffers(nl, nt, px), stream);
      t = totals_now();
    }
    out.lines.resize(ld.size());
    out.comps.resize(static_cast<std::size_t>(t.comps));
    out.hoff.resize(static_cast<std::size_t>(t.comps));
    out.colhist.resize(static_cast<std::size_t>(t.hist));
    CUDA_CHECK(cudaMemcpyAsync(out.lines.data(), line_out.get(), ld.size() * sizeof(gpu::LineOut),
                               cudaMemcpyDeviceToHost, stream));
    if (t.comps > 0) {
      CUDA_CHECK(cudaMemcpyAsync(out.comps.data(), comps.get(),
                                 out.comps.size() * sizeof(gpu::CompStats),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(out.hoff.data(), hoff.get(), out.hoff.size() * sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
    }
    if (t.hist > 0)
      CUDA_CHECK(cudaMemcpyAsync(out.colhist.data(), colhist.get(),
                                 out.colhist.size() * sizeof(int), cudaMemcpyDeviceToHost,
                                 stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // The per-pixel buffers share one capacity.
  void ensure_pixels(std::size_t px) {
    if (px > pixel_cap) {
      constexpr auto kMax = static_cast<std::size_t>(kChunkPixels);
      pixel_cap = std::max(px, std::min(pixel_cap + pixel_cap / 2, kMax));
      ink.reset(pixel_cap);
      claims.reset(pixel_cap);
      labels.reset(pixel_cap);
      cid.reset(pixel_cap);
    }
    ensure_scan(px);
  }

  void ensure_comps(std::size_t ncomp, std::size_t nhist, std::size_t px) {
    if (ncomp > comp_cap) {
      comp_cap = ncomp;
      comps.reset(comp_cap);
      colw.reset(comp_cap);
      hoff.reset(comp_cap);
    }
    if (nhist > colhist_cap) {
      colhist_cap = nhist;
      colhist.reset(colhist_cap);
    }
    ensure_scan(std::max(px, comp_cap));
  }

  void ensure_cands(std::size_t n) {
    if (n <= cand_cap) return;
    cand_cap = std::max(n, cand_cap + cand_cap / 2);
    cands.reset(cand_cap);
    bytag.reset(cand_cap * gpu::kProfileBins);
    prof.reset(cand_cap * gpu::kProfileBins);
  }

  // Gives back what a rare page grew past kKeep* (after its passes).
  void trim() {
    if (comp_cap > kKeepComps) {
      comps = {};
      colw = {};
      hoff = {};
      comp_cap = 0;
      comp_want = kFirstComps;
    }
    if (colhist_cap > kKeepHist) {
      colhist = {};
      colhist_cap = 0;
      hist_want = kFirstHist;
    }
    if (cand_cap > kKeepCands) {
      cands = {};
      bytag = {};
      prof = {};
      cand_cap = 0;
    }
  }

  void ensure_scan(std::size_t n) {
    const std::size_t need = gpu::scan_scratch_bytes(static_cast<int>(n));
    if (need > scan_cap) {
      scan_cap = need;
      scan_tmp.reset(scan_cap);
    }
  }

  gpu::ChunkBuffers buffers(int n_lines, int n_tiles, int n_pixels) {
    gpu::ChunkBuffers b{};
    b.lines = lines.get();
    b.nbs = nbs.get();
    b.runs = runs.get();
    b.tiles = tiles.get();
    b.n_lines = n_lines;
    b.n_tiles = n_tiles;
    b.n_pixels = n_pixels;
    b.ink = ink.get();
    b.claims = claims.get();
    b.labels = labels.get();
    b.cid = cid.get();
    b.tile_roots = tile_roots.get();
    b.tile_base = tile_base.get();
    b.dark_tiles = dark_tiles.get();
    b.root_mask = root_mask.get();
    b.hist256 = hist256.get();
    b.line_out = line_out.get();
    b.comps = comps.get();
    b.colw = colw.get();
    b.hoff = hoff.get();
    b.colhist = colhist.get();
    b.totals = totals.get();
    b.scan_tmp = scan_tmp.get();
    b.scan_tmp_bytes = scan_cap;
    b.comp_capacity = static_cast<int>(comp_cap);
    b.hist_capacity = static_cast<int>(colhist_cap);
    b.cands = cands.get();
    b.bytag = bytag.get();
    b.prof = prof.get();
    b.cand_capacity = static_cast<int>(cand_cap);
    return b;
  }
};

GpuWordPlacer::GpuWordPlacer() : impl_(std::make_unique<Impl>()) {}
GpuWordPlacer::~GpuWordPlacer() = default;

bool GpuWordPlacer::enabled() {
  static const bool on = [] {
    std::vector<std::string> errors;
    return env::env_bool_strict("TURBO_WORDS_GPU", true, errors);
  }();
  return on;
}

GpuWordPlacer::Finish GpuWordPlacer::place_later(
    const GpuImage &page, cudaStream_t stream, const std::vector<Box> &boxes,
    const std::vector<std::vector<CtcWord>> &words,
    std::vector<std::pair<std::size_t, std::size_t>> todo, cv::Mat host_page) {
  const std::size_t row = static_cast<std::size_t>(page.cols) * 3;
  const std::size_t bytes = row * static_cast<std::size_t>(page.rows);
  auto copy = std::make_shared<PageCopy>();
  if (!copy->reserve(bytes))
    return place(page, stream, boxes, words, std::move(todo),
                 host_page.empty() ? nullptr : &host_page);
  CUDA_CHECK(cudaGetDevice(&copy->dev));
  void *data = nullptr;
  if (cudaMallocAsync(&data, bytes, stream) != cudaSuccess) CUDA_CHECK(cudaGetLastError());
  copy->img = {data, row, page.rows, page.cols};
  CUDA_CHECK(cudaMemcpy2DAsync(data, row, page.data, page.step, row, page.rows,
                               cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return [copy, boxes, words, todo = std::move(todo),
          host = std::move(host_page)](std::vector<OCRResultItem> &results) {
    // The planning needs no placer: done before taking one.
    Planned planned = plan_jobs({copy->img.cols, copy->img.rows}, boxes, words, todo);
    Finish finish;
    {
      auto lease = PlacerPool::instance().acquire(copy->dev);
      CUDA_CHECK(cudaSetDevice(copy->dev));
      // The copy goes on the placer's stream, after its passes, however
      // place() ends (before the placer is given back).
      struct FreeAfter {
        PageCopy &copy;
        cudaStream_t stream;
        ~FreeAfter() { copy.free_on(stream); }
      } free_after{*copy, lease->stream};
      std::size_t cpu_lines = 0;
      finish = lease->placer.impl_->place(copy->img, lease->stream, std::move(planned), boxes,
                                          words, host.empty() ? nullptr : &host, cpu_lines);
    }
    finish(results);
  };
}

GpuWordPlacer::Finish GpuWordPlacer::place(
    const GpuImage &page, cudaStream_t stream, const std::vector<Box> &boxes,
    const std::vector<std::vector<CtcWord>> &words,
    std::vector<std::pair<std::size_t, std::size_t>> todo, const cv::Mat *host_page,
    Fallback fallback) {
  return impl_->place(page, stream,
                      plan_jobs({page.cols, page.rows}, boxes, words, todo,
                                fallback == Fallback::WholeLine),
                      boxes, words, host_page, cpu_lines_, fallback);
}

GpuWordPlacer::Finish GpuWordPlacer::Impl::place(const GpuImage &page, cudaStream_t stream,
                                                 Planned planned, const std::vector<Box> &boxes,
                                                 const std::vector<std::vector<CtcWord>> &words,
                                                 const cv::Mat *host_page, std::size_t &cpu_lines,
                                                 Fallback fallback) {
  const auto jobs = std::move(planned.jobs);
  const std::vector<std::size_t> &gpu_jobs = planned.gpu;

  // GPU passes of at most kChunkPixels crop pixels each.
  auto chunks = std::make_shared<std::vector<ChunkResult>>();
  const gpu::PageView pv{static_cast<const std::uint8_t *>(page.data), page.step, page.rows,
                         page.cols};
  for (std::size_t first = 0; first < gpu_jobs.size();) {
    std::size_t last = first;
    long long px = 0;
    while (last < gpu_jobs.size()) {
      const auto &p = (*jobs)[gpu_jobs[last]].plan;
      const long long add = static_cast<long long>(p.W) * p.H;
      if (last > first && px + add > kChunkPixels) break;
      px += add;
      ++last;
    }
    chunks->emplace_back();
    run(pv, stream, *jobs, gpu_jobs, first, last, chunks->back());
    for (std::size_t k = first; k < last; ++k)
      (*jobs)[gpu_jobs[k]].chunk = static_cast<int>(chunks->size() - 1);
    first = last;
  }

  trim();

  // Lines the GPU pass hands back: a neighbour box it could not rasterize as
  // runs, or a glyph whose bleed-split profile is finer than its bins.
  bool any_cpu = false;
  cpu_lines = 0;
  for (auto &j : *jobs) {
    if (!j.cpu)
      j.cpu = (*chunks)[static_cast<std::size_t>(j.chunk)]
                  .lines[static_cast<std::size_t>(j.index)]
                  .cpu != 0;
    any_cpu |= j.cpu;
    cpu_lines += j.cpu ? 1 : 0;
  }
  cv::Mat host;
  if (any_cpu && fallback == Fallback::CpuPixels) {
    if (host_page != nullptr) {
      host = *host_page;
    } else {
      host.create(page.rows, page.cols, CV_8UC3);
      CUDA_CHECK(cudaMemcpy2DAsync(host.data, host.step, page.data, page.step,
                                   static_cast<std::size_t>(page.cols) * 3, page.rows,
                                   cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }

  return [jobs, chunks, host, boxes, words, fallback](std::vector<OCRResultItem> &results) {
    for (auto &j : *jobs) {
      auto &dst = results[j.result].words;
      if (j.cpu) {
        if (fallback == Fallback::WholeLine) {
          dst.reserve(words[j.box].size());
          for (const auto &word : words[j.box])
            dst.push_back({word.text, word.score, boxes[j.box]});
          continue;
        }
        dst = locate_words(host, boxes[j.box], words[j.box], boxes);
        continue;
      }
      const ChunkResult &r = (*chunks)[static_cast<std::size_t>(j.chunk)];
      const int end = j.index + 1 < static_cast<int>(r.lines.size())
                          ? r.lines[static_cast<std::size_t>(j.index + 1)].comp_first
                          : static_cast<int>(r.comps.size());
      detail::Components comps = components_of(r, j.index, end, j.plan);
      const int first = r.lines[static_cast<std::size_t>(j.index)].comp_first;
      const auto &plan = j.plan;
      const auto col_profile = [&](const std::vector<char> &own) {
        std::vector<int> col(static_cast<std::size_t>(plan.w), 0);
        for (int k = 1; k < comps.n; ++k) {
          const auto ku = static_cast<std::size_t>(k);
          if (!own[ku] || comps.in_core[ku] == 0) continue;
          const int h0 = r.hoff[static_cast<std::size_t>(first + k - 1)];
          for (int x = comps.core_x0[ku]; x < comps.core_x1[ku]; ++x)
            col[static_cast<std::size_t>(x - plan.m)] +=
                r.colhist[static_cast<std::size_t>(h0 + x - comps.core_x0[ku])];
        }
        return col;
      };
      dst = detail::words_from_components(plan, comps, words[j.box], col_profile,
                                          [](detail::Components &) {},
                                          fallback == Fallback::WholeLine ? &boxes[j.box] : nullptr);
    }
  };
}

} // namespace turbo_ocr::recognition
