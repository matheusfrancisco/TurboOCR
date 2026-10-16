// The GPU word placer against the CPU implementation, to the pixel.

#include <catch_amalgamated.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "turbo_ocr/recognition/gpu_word_placer.h"
#include "turbo_ocr/recognition/word_boxes.h"
#include "../../src/recognition/gpu_word_placer_internal.h"

using turbo_ocr::Box;
using turbo_ocr::recognition::CtcWord;

namespace {

bool have_gpu() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// A page on the device, freed on scope exit.
struct DevicePage {
  turbo_ocr::GpuImage img;
  explicit DevicePage(const cv::Mat &page) {
    void *p = nullptr;
    std::size_t step = 0;
    REQUIRE(cudaMallocPitch(&p, &step, static_cast<std::size_t>(page.cols) * 3, page.rows) == cudaSuccess);
    REQUIRE(cudaMemcpy2D(p, step, page.data, page.step, static_cast<std::size_t>(page.cols) * 3,
                         page.rows, cudaMemcpyHostToDevice) == cudaSuccess);
    img = {p, step, page.rows, page.cols};
  }
  ~DevicePage() { cudaFree(img.data); }
};

Box rotated_box(float cx, float cy, float w, float h, float ang) {
  const float ca = std::cos(ang), sa = std::sin(ang);
  const float sx[4] = {-1, 1, 1, -1}, sy[4] = {-1, -1, 1, 1};
  Box b{};
  for (int k = 0; k < 4; ++k) {
    const float px = 0.5f * w * sx[k], py = 0.5f * h * sy[k];
    b[k] = {static_cast<int>(std::lround(cx + ca * px - sa * py)),
            static_cast<int>(std::lround(cy + sa * px + ca * py))};
  }
  return b;
}

} // namespace

TEST_CASE("the GPU crop is the CPU warp and gray conversion to the pixel", "[words_gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  using namespace turbo_ocr::recognition;
  // Noise, so every interpolation tap shows.
  cv::Mat page(2400, 3000, CV_8UC3);
  cv::RNG rng(0xc0de);
  rng.fill(page, cv::RNG::UNIFORM, 0, 256);
  DevicePage dev(page);
  const gpu::PageView pv{static_cast<const std::uint8_t *>(dev.img.data), dev.img.step,
                         page.rows, page.cols};
  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
  int cut = 0, affine = 0, perspective = 0, uncached = 0;
  const auto check = [&](int i, const Box &box) {
    const detail::LinePlan plan = detail::plan_line(box, page.size(), {});
    cv::Mat crop, want;
    if (plan.sub) {
      crop = page(*plan.sub);
      ++cut;
    } else {
      cv::warpPerspective(page, crop, plan.to_page, cv::Size(plan.W, plan.H),
                          cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
      ++(plan.to_page(2, 0) == 0.0f ? affine : perspective);
      // A tile's column-block bases stop fitting shared memory past 8192
      // columns (128 blocks of 64): those are computed per pixel.
      uncached += plan.W > 8192 && plan.H >= 16 ? 1 : 0;
    }
    cv::cvtColor(crop, want, cv::COLOR_BGR2GRAY);

    const gpu::LineDesc ld = gpu::line_desc(plan, 0);
    std::vector<gpu::Tile> tiles;
    for (int s = 0; s < plan.W * plan.H; s += gpu::kTilePixels)
      tiles.push_back({0, s, std::min(gpu::kTilePixels, plan.W * plan.H - s)});
    gpu::LineDesc *d_line = nullptr;
    gpu::Tile *d_tiles = nullptr;
    std::uint8_t *d_gray = nullptr;
    REQUIRE(cudaMalloc(&d_line, sizeof(ld)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_tiles, tiles.size() * sizeof(gpu::Tile)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_gray, static_cast<std::size_t>(plan.W) * plan.H) == cudaSuccess);
    cudaMemcpy(d_line, &ld, sizeof(ld), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tiles, tiles.data(), tiles.size() * sizeof(gpu::Tile), cudaMemcpyHostToDevice);
    gpu::ChunkBuffers b{};
    b.lines = d_line;
    b.tiles = d_tiles;
    b.n_lines = 1;
    b.n_tiles = static_cast<int>(tiles.size());
    b.n_pixels = plan.W * plan.H;
    b.ink = d_gray;
    gpu::crop_gray(pv, b, stream);
    cv::Mat got(plan.H, plan.W, CV_8U);
    REQUIRE(cudaMemcpyAsync(got.data, d_gray, got.total(), cudaMemcpyDeviceToHost, stream) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    cudaFree(d_line);
    cudaFree(d_tiles);
    cudaFree(d_gray);
    INFO("box " << i << " " << plan.W << "x" << plan.H << (plan.sub ? " cut" : " warped"));
    REQUIRE(cv::norm(want, got, cv::NORM_INF) == 0);
  };
  for (int i = 0; i < 400; ++i) {
    // Upright, rotated, tall (vertical text), tiny (upsampled) and huge
    // (downsampled) boxes, some running off the page.
    const float h = i % 5 == 0 ? rng.uniform(4.0f, 15.0f)
                  : i % 5 == 1 ? rng.uniform(170.0f, 420.0f) : rng.uniform(16.0f, 150.0f);
    const float w = i % 7 == 0 ? h / rng.uniform(1.6f, 4.0f) : rng.uniform(1.5f * h, 1400.0f);
    const float ang = i % 3 == 0 ? 0.0f : rng.uniform(-0.5f, 0.5f);
    const float cx = rng.uniform(-100.0f, 3100.0f), cy = rng.uniform(-60.0f, 2460.0f);
    Box box = rotated_box(cx, cy, w, h, ang);
    if (ang == 0.0f && i % 2 == 0)  // integer upright boxes: the sub-image path
      box = {{{{{static_cast<int>(cx), static_cast<int>(cy)}},
                {{static_cast<int>(cx + w), static_cast<int>(cy)}},
                {{static_cast<int>(cx + w), static_cast<int>(cy + h)}},
                {{static_cast<int>(cx), static_cast<int>(cy + h)}}}}};
    check(i, box);
  }
  // Lines wider than the page, turned a little (a banner, a table rule).
  for (int i = 0; i < 6; ++i)
    check(400 + i, rotated_box(rng.uniform(1000.0f, 2000.0f), rng.uniform(200.0f, 2200.0f),
                               rng.uniform(8500.0f, 12000.0f), rng.uniform(18.0f, 60.0f),
                               (i % 2 ? 1.0f : -1.0f) * rng.uniform(0.005f, 0.02f)));
  cudaStreamDestroy(stream);
  INFO(cut << " cut, " << affine << " affine, " << perspective << " perspective, " << uncached
            << " wide");
  CHECK(affine > 20);
  CHECK(perspective > 20);
  CHECK(uncached >= 6);
  CHECK(cut > 30);
}

namespace {

// A printed page with its line boxes and word positions, as the recognizer
// would report them.
struct Page {
  cv::Mat img;
  std::vector<Box> boxes;
  std::vector<std::vector<CtcWord>> words;
};

void add_line(Page &p, const std::string &text, int x, int y, double scale, int thick,
              bool inverted = false) {
  int base = 0;
  const cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thick, &base);
  const int top = y - sz.height - 2, bottom = y + base + 2;
  if (inverted)
    cv::rectangle(p.img, {x - 6, top - 4}, {x + sz.width + 6, bottom + 4}, cv::Scalar(20, 20, 20),
                  cv::FILLED);
  cv::putText(p.img, text, {x, y}, cv::FONT_HERSHEY_SIMPLEX, scale,
              inverted ? cv::Scalar(240, 240, 240) : cv::Scalar(10, 10, 10), thick, cv::LINE_AA);
  p.boxes.push_back({{{{{x - 2, top}}, {{x + sz.width + 2, top}}, {{x + sz.width + 2, bottom}},
                       {{x - 2, bottom}}}}});
  std::vector<CtcWord> ws;
  const float n = static_cast<float>(text.size());
  std::size_t s = 0;
  while (s < text.size()) {
    std::size_t e = text.find(' ', s);
    if (e == std::string::npos) e = text.size();
    ws.push_back({text.substr(s, e - s), 0.9f, (static_cast<float>(s) + 0.5f) / n,
                  (static_cast<float>(e) - 0.5f) / n});
    s = e + 1;
  }
  p.words.push_back(ws);
}

Page printed_page() {
  Page p;
  p.img = cv::Mat(2200, 1700, CV_8UC3, cv::Scalar(245, 245, 245));
  add_line(p, "A HEADLINE SET LARGE", 60, 220, 5.5, 12);  // taller than 160 px
  for (int i = 0; i < 40; ++i)  // body text, lines tight enough to claim
    add_line(p, "the quick brown fox jumps over the lazy dog " + std::to_string(i), 60,
             320 + i * 30, 0.9, 2);
  add_line(p, "tiny print below sixteen pixels high", 60, 1560, 0.35, 1);
  add_line(p, "INVERTED LINE ON A DARK BAND", 60, 1660, 1.4, 3, true);
  add_line(p, "edge", 1640, 2190, 1.2, 2);  // runs off the page corner
  for (int i = 0; i < 12; ++i)  // lines whose glyphs touch the next line's
    add_line(p, "gjpqy descenders touching ascenders bdfhkl " + std::to_string(i), 60,
             1760 + i * 22, 0.95, 2);
  return p;
}

// The same page turned by `deg`, boxes turned with it.
Page turned(const Page &src, double deg) {
  Page p = src;
  const cv::Point2f c(src.img.cols / 2.0f, src.img.rows / 2.0f);
  const cv::Mat R = cv::getRotationMatrix2D(c, deg, 1.0);
  cv::warpAffine(src.img, p.img, R, src.img.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
  for (auto &b : p.boxes)
    for (int k = 0; k < 4; ++k) {
      auto &pt = b[k];
      const double x = R.at<double>(0, 0) * pt[0] + R.at<double>(0, 1) * pt[1] + R.at<double>(0, 2);
      const double y = R.at<double>(1, 0) * pt[0] + R.at<double>(1, 1) * pt[1] + R.at<double>(1, 2);
      pt = {static_cast<int>(std::lround(x)), static_cast<int>(std::lround(y))};
    }
  return p;
}

} // namespace

TEST_CASE("GPU word boxes are the CPU word boxes, line for line", "[words_gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  using namespace turbo_ocr::recognition;
  const Page upright = printed_page();
  for (const double deg : {0.0, 1.7, -3.2}) {
    const Page p = deg == 0.0 ? upright : turned(upright, deg);
    DevicePage dev(p.img);
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    GpuWordPlacer placer;
    std::vector<std::pair<std::size_t, std::size_t>> todo;
    for (std::size_t i = 0; i < p.boxes.size(); ++i) todo.emplace_back(i, i);
    auto finish = placer.place(dev.img, stream, p.boxes, p.words, todo, nullptr);
    std::vector<turbo_ocr::OCRResultItem> results(p.boxes.size());
    finish(results);
    // Deferred: the page copied, the rest finished later on another thread.
    auto later = placer.place_later(dev.img, stream, p.boxes, p.words, todo, cv::Mat());
    std::vector<turbo_ocr::OCRResultItem> deferred(p.boxes.size());
    std::thread([&] { later(deferred); }).join();
    cudaStreamDestroy(stream);
    INFO("turned " << deg << " deg");
    // Inverted type and glyphs of touching lines included, the GPU decides all.
    CHECK(placer.cpu_lines() == 0);
    for (std::size_t i = 0; i < p.boxes.size(); ++i) {
      const auto want = locate_words(p.img, p.boxes[i], p.words[i], p.boxes);
      const auto &got = results[i].words;
      INFO("line " << i);
      REQUIRE(got.size() == want.size());
      REQUIRE(deferred[i].words.size() == want.size());
      for (std::size_t w = 0; w < want.size(); ++w)
        CHECK(deferred[i].words[w].box == want[w].box);
      for (std::size_t w = 0; w < want.size(); ++w) {
        CHECK(got[w].text == want[w].text);
        CHECK(got[w].box == want[w].box);
      }
    }
  }
}

namespace {

// place() without and with the host page, and place_later() finished on
// another thread, each against locate_words; the lines place() handed back.
std::size_t require_cpu_words(const Page &p) {
  using namespace turbo_ocr::recognition;
  DevicePage dev(p.img);
  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
  GpuWordPlacer placer;
  std::vector<std::pair<std::size_t, std::size_t>> todo;
  for (std::size_t i = 0; i < p.boxes.size(); ++i) todo.emplace_back(i, i);
  std::vector<std::vector<turbo_ocr::OCRResultItem>> got(
      3, std::vector<turbo_ocr::OCRResultItem>(p.boxes.size()));
  placer.place(dev.img, stream, p.boxes, p.words, todo, nullptr)(got[0]);
  const std::size_t handed_back = placer.cpu_lines();
  placer.place(dev.img, stream, p.boxes, p.words, todo, &p.img)(got[1]);
  auto later = placer.place_later(dev.img, stream, p.boxes, p.words, todo, cv::Mat());
  std::thread([&] { later(got[2]); }).join();
  cudaStreamDestroy(stream);
  for (std::size_t i = 0; i < p.boxes.size(); ++i) {
    const auto want = locate_words(p.img, p.boxes[i], p.words[i], p.boxes);
    for (std::size_t k = 0; k < got.size(); ++k) {
      INFO("line " << i << ", way " << k);
      REQUIRE(got[k][i].words.size() == want.size());
      for (std::size_t w = 0; w < want.size(); ++w) {
        REQUIRE(got[k][i].words[w].text == want[w].text);
        REQUIRE(got[k][i].words[w].box == want[w].box);
      }
    }
  }
  return handed_back;
}

} // namespace

TEST_CASE("GPU word boxes: grown buffers, lines handed back, nothing to place", "[words_gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  using namespace turbo_ocr::recognition;
  SECTION("a page of speckle: far more components than the first buffers hold") {
    Page p;
    p.img = cv::Mat(4000, 1700, CV_8UC3, cv::Scalar(245, 245, 245));
    for (int i = 0; i < 120; ++i)
      add_line(p, "the quick brown fox jumps over the lazy dog " + std::to_string(i), 60,
               60 + i * 32, 0.9, 2);
    // A dot every other pixel, none touching another, over one full pass of
    // crops: ~1.2 million components (the buffers start at 65,536), ~600,000
    // of them in the line boxes' columns (the histogram starts at 262,144).
    for (int y = 0; y < p.img.rows; y += 2)
      for (int x = 0; x < p.img.cols; x += 2) p.img.at<cv::Vec3b>(y, x) = {15, 15, 15};
    CHECK(require_cpu_words(p) == 0);
  }
  SECTION("a line the GPU pass hands back to the CPU code") {
    Page p = printed_page();
    // A concave (chevron) box across body line 10: no two runs a row paint
    // it, so that line's claims are the CPU code's.
    const int y = 320 + 10 * 30;
    p.boxes.push_back({{{{{300, y - 25}}, {{550, y + 15}}, {{800, y - 25}}, {{550, y - 15}}}}});
    p.words.emplace_back();  // it reads nothing itself
    REQUIRE_FALSE(detail::plan_line(p.boxes[11], p.img.size(), p.boxes).runs_ok);
    CHECK(require_cpu_words(p) >= 1);
  }
  SECTION("a page too large to copy is placed at once") {
    // 150 MB of pixels, past what waiting page copies may hold together.
    Page p;
    p.img = cv::Mat(7000, 7200, CV_8UC3, cv::Scalar(245, 245, 245));
    for (int i = 0; i < 6; ++i)
      add_line(p, "a line on a very large page " + std::to_string(i), 300, 400 + i * 900, 2.0, 4);
    CHECK(require_cpu_words(p) == 0);
  }
  SECTION("nothing to place") {
    const Page p = printed_page();
    DevicePage dev(p.img);
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    GpuWordPlacer placer;
    std::vector<turbo_ocr::OCRResultItem> results(p.boxes.size());
    placer.place(dev.img, stream, p.boxes, p.words, {}, nullptr)(results);
    placer.place_later(dev.img, stream, p.boxes, p.words, {}, cv::Mat())(results);
    // Lines that read nothing place nothing.
    const std::vector<std::vector<CtcWord>> none(p.boxes.size());
    std::vector<std::pair<std::size_t, std::size_t>> todo;
    for (std::size_t i = 0; i < p.boxes.size(); ++i) todo.emplace_back(i, i);
    placer.place(dev.img, stream, p.boxes, none, todo, nullptr)(results);
    placer.place_later(dev.img, stream, p.boxes, none, todo, cv::Mat())(results);
    cudaStreamDestroy(stream);
    for (const auto &r : results) CHECK(r.words.empty());
  }
}

TEST_CASE("device-only words preserve whole-line coverage when geometry is unresolved",
          "[words_gpu][device]") {
  if (!have_gpu()) SKIP("no CUDA device");
  using namespace turbo_ocr::recognition;
  Page p;
  std::size_t line = 0;
  SECTION("unrasterizable neighbouring text cannot force a host-pixel fallback") {
    p = printed_page();
    const int y = 320 + 10 * 30;
    p.boxes.push_back({{{{{300, y - 25}}, {{550, y + 15}}, {{800, y - 25}},
                         {{550, y - 15}}}}});
    p.words.emplace_back();
    line = 11;
    REQUIRE_FALSE(detail::plan_line(p.boxes[line], p.img.size(), p.boxes).runs_ok);
  }
  SECTION("words with no attributable ink cannot receive guessed sub-line boxes") {
    p.img = cv::Mat(200, 600, CV_8UC3, cv::Scalar(255, 255, 255));
    p.boxes.push_back({{{{{20, 30}}, {{500, 30}}, {{500, 80}}, {{20, 80}}}}});
    p.words.push_back({{"first", 0.9f, 0.1f, 0.3f}, {"last", 0.9f, 0.7f, 0.9f}});
  }
  SECTION("CTC anchors in recognizer padding cannot locate a word inside the line") {
    p.img = cv::Mat(200, 600, CV_8UC3, cv::Scalar(255, 255, 255));
    add_line(p, "padded", 40, 90, 1.0, 2);
    p.words[0][0].first = 1.1f;
    p.words[0][0].last = 1.2f;
  }
  DevicePage dev(p.img);
  GpuWordPlacer placer;
  std::vector<turbo_ocr::OCRResultItem> results(1);
  placer.place(dev.img, nullptr, p.boxes, p.words, {{0, line}}, nullptr,
                GpuWordPlacer::Fallback::WholeLine)(results);
  REQUIRE(results[0].words.size() == p.words[line].size());
  for (std::size_t i = 0; i < p.words[line].size(); ++i) {
    CHECK(results[0].words[i].text == p.words[line][i].text);
    CHECK(results[0].words[i].confidence == p.words[line][i].score);
    CHECK(results[0].words[i].box == p.boxes[line]);
  }
}

// Real pages (TURBO_WORDS_GPU_PAGES = files of "image path, lines, words" as
// written by the evaluation tooling): the GPU words must be the CPU words.
TEST_CASE("GPU word boxes on real pages", "[words_gpu_pages]") {
  const char *list = std::getenv("TURBO_WORDS_GPU_PAGES");
  if (!list || !have_gpu()) SKIP("TURBO_WORDS_GPU_PAGES not set");
  using namespace turbo_ocr::recognition;
  std::ifstream files(list);
  std::string file;
  std::size_t lines = 0, cpu = 0, words_total = 0;
  double t_gpu = 0, t_cpu = 0, t_worker = 0;
  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
  GpuWordPlacer placer;
  while (std::getline(files, file)) {
    std::ifstream in(file);
    std::string path;
    std::getline(in, path);
    int n = 0;
    in >> n;
    Page p;
    p.img = cv::imread(path, cv::IMREAD_COLOR);
    REQUIRE(!p.img.empty());
    for (int i = 0; i < n; ++i) {
      Box b{};
      for (int k = 0; k < 4; ++k) in >> b[k][0] >> b[k][1];
      int nw = 0;
      in >> nw;
      std::vector<CtcWord> ws;
      for (int j = 0; j < nw; ++j) {
        CtcWord w;
        in >> w.first >> w.last;
        std::getline(in, w.text);
        w.score = 0.9f;
        ws.push_back(w);
      }
      p.boxes.push_back(b);
      p.words.push_back(ws);
    }
    DevicePage dev(p.img);
    std::vector<std::pair<std::size_t, std::size_t>> todo;
    for (std::size_t i = 0; i < p.boxes.size(); ++i) todo.emplace_back(i, i);
    const auto t0 = std::chrono::steady_clock::now();
    auto finish = placer.place(dev.img, stream, p.boxes, p.words, todo, nullptr);
    const auto tm = std::chrono::steady_clock::now();
    std::vector<turbo_ocr::OCRResultItem> results(p.boxes.size());
    finish(results);
    const auto t1 = std::chrono::steady_clock::now();
    t_worker += std::chrono::duration<double, std::milli>(tm - t0).count();
    std::vector<std::vector<turbo_ocr::OCRWord>> want(p.boxes.size());
    for (std::size_t i = 0; i < p.boxes.size(); ++i)
      want[i] = locate_words(p.img, p.boxes[i], p.words[i], p.boxes);
    const auto t2 = std::chrono::steady_clock::now();
    t_gpu += std::chrono::duration<double, std::milli>(t1 - t0).count();
    t_cpu += std::chrono::duration<double, std::milli>(t2 - t1).count();
    lines += p.boxes.size();
    cpu += placer.cpu_lines();
    for (std::size_t i = 0; i < p.boxes.size(); ++i) {
      INFO(path << " line " << i);
      REQUIRE(results[i].words.size() == want[i].size());
      for (std::size_t w = 0; w < want[i].size(); ++w) {
        REQUIRE(results[i].words[w].text == want[i][w].text);
        REQUIRE(results[i].words[w].box == want[i][w].box);
        ++words_total;
      }
    }
  }
  cudaStreamDestroy(stream);
  WARN("real pages: " << lines << " lines, " << words_total << " words identical; "
       << cpu << " lines handed to the CPU; GPU placer " << t_gpu << " ms (" << t_worker
       << " ms before the decisions), CPU (1 thread) " << t_cpu << " ms");
}
