#include "turbo_ocr/recognition/word_boxes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "turbo_ocr/common/geometry/perspective.h"
#include "word_boxes_internal.h"

// Built with -ffp-contract=off (CMakeLists.txt): the float geometry below is
// evaluated as written, so the GPU path (word_boxes_gpu.cu, no fused
// multiply-adds either) and every CPU target decide the same pixels.

namespace turbo_ocr::recognition {

namespace {

using detail::Components;
using detail::LinePlan;

float edge_len(const Box &b, int i, int j) {
  const float dx = static_cast<float>(b[i][0] - b[j][0]);
  const float dy = static_cast<float>(b[i][1] - b[j][1]);
  return std::sqrt(dx * dx + dy * dy);
}

// Widest run of columns in [lo, hi) whose ink is at most `floor_ink`, as
// [begin, end). Empty when no column qualifies.
std::pair<int, int> widest_run(const std::vector<int> &col, int lo, int hi,
                               int floor_ink) {
  std::pair<int, int> best{lo, lo};
  int begin = -1;
  for (int x = lo; x <= hi; ++x) {
    const bool clear = x < hi && col[static_cast<size_t>(x)] <= floor_ink;
    if (clear && begin < 0) begin = x;
    if (!clear && begin >= 0) {
      if (x - begin > best.second - best.first) best = {begin, x};
      begin = -1;
    }
  }
  return best;
}

// From an anchor column outwards (dir -1 = left, +1 = right) past the word's
// ink to the first clear run of `gap` columns, or to the crop border; returns
// that side's boundary. Ink beyond such a gap belongs to a neighbour the
// line box grazed, not to the line. Clear columns before any ink is met do
// not count: the anchor itself may sit just outside its glyph.
int outer_edge(const std::vector<int> &col, int anchor, int dir, int gap,
               int noise) {
  const int w = static_cast<int>(col.size());
  bool seen_ink = false;
  int run = 0;
  for (int x = anchor; x >= 0 && x < w; x += dir) {
    if (col[static_cast<size_t>(x)] > noise) {
      seen_ink = true;
      run = 0;
    } else if (seen_ink && ++run >= gap) {
      return dir < 0 ? x + run : x - run + 1;
    }
  }
  return dir < 0 ? 0 : w;
}

// Whether the light pixels are the ground: most of the light inside `core`
// reaches the crop's border through light, the way paper runs between and
// around heavy letters. A light glyph on a dark ground is enclosed by it.
bool light_is_ground(const cv::Mat &ink, const cv::Rect &core) {
  cv::Mat lab;
  const cv::Mat light = 1 - ink;
  const int n = cv::connectedComponents(light, lab, 4, CV_32S);
  std::vector<unsigned char> border(static_cast<size_t>(n), 0);
  for (int x = 0; x < lab.cols; ++x) {
    border[static_cast<size_t>(lab.at<int>(0, x))] = 1;
    border[static_cast<size_t>(lab.at<int>(lab.rows - 1, x))] = 1;
  }
  for (int y = 0; y < lab.rows; ++y) {
    border[static_cast<size_t>(lab.at<int>(y, 0))] = 1;
    border[static_cast<size_t>(lab.at<int>(y, lab.cols - 1))] = 1;
  }
  int light_px = 0, reaching = 0;
  for (int y = core.y; y < core.y + core.height; ++y) {
    const int *row = lab.ptr<int>(y);
    for (int x = core.x; x < core.x + core.width; ++x) {
      if (row[x] == 0) continue;  // label 0 is the ink
      ++light_px;
      reaching += border[static_cast<size_t>(row[x])];
    }
  }
  return 2 * reaching >= light_px;
}

// --- cv::fillConvexPoly(LINE_8, shift 0), as runs --------------------------
// OpenCV 4 draws the polygon's outline with 8-connected Bresenham lines
// clipped to the mask, then fills scanlines between its two edges in 16.16
// fixed point; this is that arithmetic, step for step (drawing.cpp:
// clipLine, LineIterator, FillConvexPoly), so the runs are the painted pixels.

bool clip_line(std::int64_t W, std::int64_t H, std::int64_t &x1, std::int64_t &y1,
               std::int64_t &x2, std::int64_t &y2) {
  const std::int64_t right = W - 1, bottom = H - 1;
  if (W <= 0 || H <= 0) return false;
  int c1 = (x1 < 0) + (x1 > right) * 2 + (y1 < 0) * 4 + (y1 > bottom) * 8;
  int c2 = (x2 < 0) + (x2 > right) * 2 + (y2 < 0) * 4 + (y2 > bottom) * 8;
  if ((c1 & c2) == 0 && (c1 | c2) != 0) {
    std::int64_t a;
    if (c1 & 12) {
      a = c1 < 8 ? 0 : bottom;
      x1 += static_cast<std::int64_t>(static_cast<double>(a - y1) * (x2 - x1) / (y2 - y1));
      y1 = a;
      c1 = (x1 < 0) + (x1 > right) * 2;
    }
    if (c2 & 12) {
      a = c2 < 8 ? 0 : bottom;
      x2 += static_cast<std::int64_t>(static_cast<double>(a - y2) * (x2 - x1) / (y2 - y1));
      y2 = a;
      c2 = (x2 < 0) + (x2 > right) * 2;
    }
    if ((c1 & c2) == 0 && (c1 | c2) != 0) {
      if (c1) {
        a = c1 == 1 ? 0 : right;
        y1 += static_cast<std::int64_t>(static_cast<double>(a - x1) * (y2 - y1) / (x2 - x1));
        x1 = a;
        c1 = 0;
      }
      if (c2) {
        a = c2 == 1 ? 0 : right;
        y2 += static_cast<std::int64_t>(static_cast<double>(a - x2) * (y2 - y1) / (x2 - x1));
        x2 = a;
        c2 = 0;
      }
    }
  }
  return (c1 | c2) == 0;
}

// The pixels cv::Line(mask, p1, p2, color, 8) paints, left to right.
template <typename Emit>
void line_pixels(int W, int H, cv::Point p1, cv::Point p2, Emit &&emit) {
  if (static_cast<unsigned>(p1.x) >= static_cast<unsigned>(W) ||
      static_cast<unsigned>(p2.x) >= static_cast<unsigned>(W) ||
      static_cast<unsigned>(p1.y) >= static_cast<unsigned>(H) ||
      static_cast<unsigned>(p2.y) >= static_cast<unsigned>(H)) {
    std::int64_t x1 = p1.x, y1 = p1.y, x2 = p2.x, y2 = p2.y;
    if (!clip_line(W, H, x1, y1, x2, y2)) return;
    p1 = {static_cast<int>(x1), static_cast<int>(y1)};
    p2 = {static_cast<int>(x2), static_cast<int>(y2)};
  }
  int delta_x = 1, delta_y = 1;
  int dx = p2.x - p1.x, dy = p2.y - p1.y;
  if (dx < 0) {  // left to right
    dx = -dx;
    dy = -dy;
    p1 = p2;
  }
  if (dy < 0) {
    dy = -dy;
    delta_y = -1;
  }
  const bool vert = dy > dx;
  if (vert) {
    std::swap(dx, dy);
    std::swap(delta_x, delta_y);
  }
  int err = dx - (dy + dy);
  const int plus_delta = dx + dx, minus_delta = -(dy + dy);
  int minus_shift = delta_x, plus_shift = 0, minus_step = 0, plus_step = delta_y;
  if (vert) {
    std::swap(plus_step, plus_shift);
    std::swap(minus_step, minus_shift);
  }
  int x = p1.x, y = p1.y;
  for (int i = 0; i <= dx; ++i) {
    emit(x, y);
    const int mask = err < 0 ? -1 : 0;
    err += minus_delta + (plus_delta & mask);
    x += minus_shift + (plus_shift & mask);
    y += minus_step + (plus_step & mask);
  }
}

}  // namespace

namespace detail {

std::optional<cv::Rect> exact_subimage(const cv::Matx33f &to_page, int W, int H,
                                       cv::Size page) {
  const double tx = std::round(to_page(0, 2)), ty = std::round(to_page(1, 2));
  for (const auto &[x, y] : {std::pair{0, 0}, {W - 1, 0}, {0, H - 1}, {W - 1, H - 1}}) {
    const double w = to_page(2, 0) * x + to_page(2, 1) * y + to_page(2, 2);
    const double px = (to_page(0, 0) * x + to_page(0, 1) * y + to_page(0, 2)) / w;
    const double py = (to_page(1, 0) * x + to_page(1, 1) * y + to_page(1, 2)) / w;
    if (std::abs(px - (x + tx)) > 1.0 / 1024 || std::abs(py - (y + ty)) > 1.0 / 1024)
      return std::nullopt;
  }
  const cv::Rect r(static_cast<int>(tx), static_cast<int>(ty), W, H);
  if ((r & cv::Rect(cv::Point(0, 0), page)) != r) return std::nullopt;
  return r;
}

bool quad_runs(const std::array<cv::Point, 4> &v, int W, int H, QuadRuns &out) {
  constexpr int kShift = 16;
  constexpr std::int64_t kOne = std::int64_t{1} << kShift;
  constexpr std::int64_t kHalf = kOne >> 1;
  out = {};
  // Per row, the intervals painted: the fill span and each outline edge's run.
  struct Iv { int y, a, b; };
  std::vector<Iv> ivs;
  const auto add = [&](int y, int a, int b) { ivs.push_back({y, a, b}); };

  // Outline: edges v3-v0, v0-v1, v1-v2, v2-v3.
  cv::Point p0 = v[3];
  for (int i = 0; i < 4; ++i) {
    int row = std::numeric_limits<int>::min(), a = 0, b = 0;
    line_pixels(W, H, p0, v[static_cast<size_t>(i)], [&](int x, int y) {
      if (y == row) {
        a = std::min(a, x);
        b = std::max(b, x);
        return;
      }
      if (row != std::numeric_limits<int>::min()) add(row, a, b);
      row = y;
      a = b = x;
    });
    if (row != std::numeric_limits<int>::min()) add(row, a, b);
    p0 = v[static_cast<size_t>(i)];
  }

  // Fill.
  std::int64_t xmin = v[0].x, xmax = v[0].x, ymin = v[0].y, ymax = v[0].y;
  int imin = 0;
  for (int i = 0; i < 4; ++i) {
    const cv::Point p = v[static_cast<size_t>(i)];
    if (p.y < ymin) {
      ymin = p.y;
      imin = i;
    }
    ymax = std::max<std::int64_t>(ymax, p.y);
    xmax = std::max<std::int64_t>(xmax, p.x);
    xmin = std::min<std::int64_t>(xmin, p.x);
  }
  if (!(xmax < 0 || ymax < 0 || xmin >= W || ymin >= H)) {
    ymax = std::min<std::int64_t>(ymax, H - 1);
    struct Edge { int idx, di; std::int64_t x, dx; int ye; } edge[2];
    edge[0].idx = edge[1].idx = imin;
    int y = static_cast<int>(ymin);
    edge[0].ye = edge[1].ye = y;
    edge[0].di = 1;
    edge[1].di = 3;
    edge[0].x = edge[1].x = -kOne;
    edge[0].dx = edge[1].dx = 0;
    int edges = 4;
    do {
      for (int i = 0; i < 2; ++i) {
        if (y >= edge[i].ye) {
          int idx0 = edge[i].idx, di = edge[i].di;
          int idx = idx0 + di;
          if (idx >= 4) idx -= 4;
          for (; edges-- > 0;) {
            const int ty = v[static_cast<size_t>(idx)].y;
            if (ty > y) {
              const std::int64_t xs = std::int64_t{v[static_cast<size_t>(idx0)].x} << kShift;
              const std::int64_t xe = std::int64_t{v[static_cast<size_t>(idx)].x} << kShift;
              edge[i].ye = ty;
              edge[i].dx = ((xe - xs) * 2 + (ty - y)) / (2 * (ty - y));
              edge[i].x = xs;
              edge[i].idx = idx;
              break;
            }
            idx0 = idx;
            idx += di;
            if (idx >= 4) idx -= 4;
          }
        }
      }
      if (edges < 0) break;
      if (y >= 0) {
        const int left = edge[0].x > edge[1].x ? 1 : 0, right = 1 - left;
        int xx1 = static_cast<int>((edge[left].x + kHalf) >> kShift);
        int xx2 = static_cast<int>((edge[right].x + kHalf) >> kShift);
        if (xx2 >= 0 && xx1 < W) {
          xx1 = std::max(xx1, 0);
          xx2 = std::min(xx2, W - 1);
          if (xx1 <= xx2) add(y, xx1, xx2);
        }
      }
      edge[0].x += edge[0].dx;
      edge[1].x += edge[1].dx;
    } while (++y <= static_cast<int>(ymax));
  }

  if (ivs.empty()) return true;
  std::sort(ivs.begin(), ivs.end(), [](const Iv &p, const Iv &q) {
    return p.y != q.y ? p.y < q.y : p.a < q.a;
  });
  out.y0 = ivs.front().y;
  out.rows.assign(static_cast<size_t>(ivs.back().y - out.y0 + 1), {1, 0, 1, 0});
  for (size_t i = 0; i < ivs.size();) {
    const int y = ivs[i].y;
    int runs = 0, a = ivs[i].a, b = ivs[i].b;
    auto &row = out.rows[static_cast<size_t>(y - out.y0)];
    for (; i < ivs.size() && ivs[i].y == y; ++i) {
      if (ivs[i].a <= b + 1) {
        b = std::max(b, ivs[i].b);
        continue;
      }
      if (runs == 2) return false;
      row[2 * runs] = static_cast<std::int16_t>(a);
      row[2 * runs + 1] = static_cast<std::int16_t>(b);
      ++runs;
      a = ivs[i].a;
      b = ivs[i].b;
    }
    if (runs == 2) return false;
    row[2 * runs] = static_cast<std::int16_t>(a);
    row[2 * runs + 1] = static_cast<std::int16_t>(b);
  }
  return true;
}

LinePlan plan_line(const Box &line, cv::Size page, const std::vector<Box> &page_lines) {
  LinePlan p;
  // compute_crop_transform's orientation rule: after its vertical-text swap
  // the text runs along the crop's x axis, the other edge is its height.
  const float e01 = edge_len(line, 0, 1);
  const float e03 = edge_len(line, 0, 3);
  p.vertical = e03 >= e01 * kVerticalAspectRatio;
  p.h = std::clamp(static_cast<int>(std::lround(p.vertical ? e01 : e03)), kMinLineH,
                   kMaxLineH);
  p.ct = compute_crop_transform(line, p.h, kMaxLineW);
  p.w = p.ct.crop_width;
  // The crop reaches `m` pixels past the line box on every side, so glyphs
  // the detector's box clips (a descender, a comma's tail, an accent, a tall
  // bracket) are still whole. Extended crop pixel (x, y) is line-crop pixel
  // (x - m, y - m); the line box itself is the `core` rectangle.
  p.m = std::max(2, p.h / 2);
  p.W = p.w + 2 * p.m;
  p.H = p.h + 2 * p.m;
  p.core = cv::Rect(p.m, p.m, p.w, p.h);
  const cv::Matx33f m_inv(p.ct.M_inv[0], p.ct.M_inv[1], p.ct.M_inv[2],
                          p.ct.M_inv[3], p.ct.M_inv[4], p.ct.M_inv[5],
                          p.ct.M_inv[6], p.ct.M_inv[7], p.ct.M_inv[8]);
  const cv::Matx33f shift(1, 0, static_cast<float>(-p.m),
                          0, 1, static_cast<float>(-p.m), 0, 0, 1);
  p.to_page = m_inv * shift;
  // Most lines are upright boxes the crop maps 1:1: cut them out instead.
  p.sub = exact_subimage(p.to_page, p.W, p.H, page);
  p.own_mid = static_cast<float>(p.m) + 0.5f * static_cast<float>(p.h);

  // Other detected lines reaching into the crop. A line whose box misses the
  // page footprint of the crop grown by two crop pixels (more than the
  // rounding of its corners below) paints and claims nothing, so it is
  // skipped before any transform.
  float fx0 = std::numeric_limits<float>::max(), fy0 = fx0, fx1 = -fx0, fy1 = -fx0;
  for (const auto &[cx, cy] :
       {std::pair{-2, -2}, {p.W + 2, -2}, {-2, p.H + 2}, {p.W + 2, p.H + 2}}) {
    const cv::Vec3f q = p.to_page * cv::Vec3f(static_cast<float>(cx), static_cast<float>(cy), 1.0f);
    fx0 = std::min(fx0, q[0] / q[2]); fx1 = std::max(fx1, q[0] / q[2]);
    fy0 = std::min(fy0, q[1] / q[2]); fy1 = std::max(fy1, q[1] / q[2]);
  }
  const cv::Matx33f to_crop = p.to_page.inv();
  std::size_t claimers = 0;
  for (const Box &o : page_lines) {
    if (claimers >= kMaxClaimers) break;
    if (o == line) continue;
    const auto [ox0, ox1] = std::minmax({o[0][0], o[1][0], o[2][0], o[3][0]});
    const auto [oy0, oy1] = std::minmax({o[0][1], o[1][1], o[2][1], o[3][1]});
    if (static_cast<float>(ox1) < fx0 || static_cast<float>(ox0) > fx1 ||
        static_cast<float>(oy1) < fy0 || static_cast<float>(oy0) > fy1)
      continue;
    std::array<cv::Point2f, 4> q;
    for (int k = 0; k < 4; ++k) {
      const cv::Vec3f v = to_crop * cv::Vec3f(static_cast<float>(o[k][0]),
                                              static_cast<float>(o[k][1]), 1.0f);
      q[static_cast<size_t>(k)] = {v[0] / v[2], v[1] / v[2]};
    }
    float qx0 = q[0].x, qx1 = q[0].x, qy0 = q[0].y, qy1 = q[0].y;
    for (const auto &c : q) {
      qx0 = std::min(qx0, c.x); qx1 = std::max(qx1, c.x);
      qy0 = std::min(qy0, c.y); qy1 = std::max(qy1, c.y);
    }
    if (qx1 < 0 || qy1 < 0 || qx0 >= static_cast<float>(p.W) || qy0 >= static_cast<float>(p.H))
      continue;
    std::array<cv::Point, 4> poly;
    for (int k = 0; k < 4; ++k)
      poly[static_cast<size_t>(k)] = {static_cast<int>(std::lround(q[static_cast<size_t>(k)].x)),
                                      static_cast<int>(std::lround(q[static_cast<size_t>(k)].y))};
    Neighbour nb;
    if (!quad_runs(poly, p.W, p.H, nb.runs)) p.runs_ok = false;
    // A line running this way with its own middle claims ink; the same line
    // cut into two boxes shares this middle: nothing to claim.
    const cv::Point2f L = 0.5f * (q[0] + q[3]), R = 0.5f * (q[1] + q[2]);
    if (!(std::abs(R.x - L.x) < 1.0f)) {
      const float slope = (R.y - L.y) / (R.x - L.x);
      const auto mid_at = [&](float x) { return L.y + slope * (x - L.x); };
      nb.claims = !(std::abs(slope) > 0.6f) &&
                  !(std::abs(mid_at(0.5f * static_cast<float>(p.W)) - p.own_mid) <
                    0.25f * static_cast<float>(p.h));
      nb.slope = slope;
    }
    if (nb.claims) {
      ++claimers;
      nb.left = L;
      nb.wy0 = std::max(0, static_cast<int>(qy0));
      nb.wy1 = std::min(p.H, static_cast<int>(qy1) + 1);
      nb.wx0 = std::max(0, static_cast<int>(qx0));
      nb.wx1 = std::min(p.W, static_cast<int>(qx1) + 1);
    }
    p.neighbours.push_back(std::move(nb));
  }
  return p;
}

std::vector<OCRWord> words_from_components(const LinePlan &plan, Components &comps,
                                           const std::vector<CtcWord> &words,
                                           const ColProfileFn &col_profile,
                                           const BandBoxedFn &band_boxed,
                                           const Box *unresolved_line) {
  std::vector<OCRWord> out;
  const int h = plan.h, w = plan.w, m = plan.m, H = plan.H, W = plan.W;
  const CropTransform &ct = plan.ct;
  const int ncomp = comps.n;

  // Extended coordinates; `label` is the component.
  struct Glyph { int x0, y0, x1, y1, core_x0, core_x1, cx, label; };
  const int min_area = std::max(3, h * h / 400);
  std::vector<char> own(static_cast<size_t>(ncomp), 0);
  std::vector<Glyph> glyphs;
  std::vector<Glyph> theirs;  // glyphs another line claimed
  for (int c = 1; c < ncomp; ++c) {
    const auto cu = static_cast<size_t>(c);
    const int area = comps.area[cu];
    const int y0 = comps.y0[cu];
    const int gh = comps.height[cu];
    const int core_px = comps.in_core[cu];
    if (area < min_area || core_px == 0) continue;
    // Running through the whole margin to the crop's edge -- half a line
    // height past the box -- is no glyph of this line: a frame, a column
    // rule, a border.
    if (y0 == 0 || y0 + gh >= h + 2 * m) continue;
    const int mid2 = 2 * y0 + gh;  // doubled vertical middle
    const int tol2 = 2 * std::max(1, h / 8);
    const bool edge_stroke = 4 * gh <= h && mid2 >= 2 * m - tol2 &&
                             mid2 < 2 * (m + h) + tol2;
    const int x0 = comps.x0[cu];
    if (2 * comps.claimed_px[cu] >= area) {  // another line's
      theirs.push_back({x0, y0, x0 + comps.width[cu], y0 + gh, 0, 0, 0, c});
      continue;
    }
    if (2 * core_px < area && !edge_stroke) continue;
    own[cu] = 1;
    glyphs.push_back({x0, y0, x0 + comps.width[cu], y0 + gh, comps.core_x0[cu],
                      comps.core_x1[cu], comps.cx[cu], c});
  }
  // A box that stops at the capitals' height leaves their accents -- and the
  // top strokes of CJK characters -- above it: a small, whole mark mostly
  // above the box and right over one of the line's own glyphs is that
  // glyph's. (The line above reaching down is a whole letter, too tall to
  // pass; and only above: the next line's accents and i-dots sit below.)
  const size_t n_own = glyphs.size();
  const int touch = std::max(2, h / 6);
  for (int c = 1; c < ncomp; ++c) {
    const auto cu = static_cast<size_t>(c);
    if (own[cu]) continue;
    const int area = comps.area[cu];
    const int x0 = comps.x0[cu];
    const int x1 = x0 + comps.width[cu];
    const int y0 = comps.y0[cu];
    const int y1 = y0 + comps.height[cu];
    if (area < min_area || y0 == 0 || 3 * (y1 - y0) > h || y0 + y1 >= 2 * m ||
        2 * comps.claimed_px[cu] >= area)
      continue;
    for (size_t g = 0; g < n_own; ++g) {
      const auto &o = glyphs[g];
      if (std::min(x1, o.x1) > std::max(x0, o.x0) && o.y0 >= y1 - 1 &&
          o.y0 - y1 <= touch) {
        glyphs.push_back({x0, y0, x1, y1, std::max(x0, m), std::min(x1, m + w),
                          comps.cx[cu], c});
        break;
      }
    }
  }

  // A small mark -- an i-dot, an accent, a period -- belongs with the nearest
  // full-size glyph: the next line's i-dots, reaching into this box where the
  // lines sit close, go with that line's letters. Small = under a third of the
  // median glyph's area.
  if (!theirs.empty()) {
    const auto gap = [](const Glyph &a, const Glyph &b) {
      const int dx = std::max({0, a.x0 - b.x1, b.x0 - a.x1});
      const int dy = std::max({0, a.y0 - b.y1, b.y0 - a.y1});
      return dx * dx + dy * dy;
    };
    const auto area_of = [&](const Glyph &g) {
      return comps.area[static_cast<size_t>(g.label)];
    };
    std::vector<int> areas;
    for (const auto &g : glyphs) areas.push_back(area_of(g));
    for (const auto &g : theirs) areas.push_back(area_of(g));
    std::nth_element(areas.begin(), areas.begin() + areas.size() / 2, areas.end());
    const int median_area = areas[areas.size() / 2];
    const auto small = [&](const Glyph &g) { return 3 * area_of(g) < median_area; };
    std::vector<Glyph> kept;
    for (const auto &g : glyphs) {
      if (small(g)) {
        int mine = std::numeric_limits<int>::max(), other = mine;
        const Glyph *mine_g = nullptr, *other_g = nullptr;
        for (const auto &o : glyphs)
          if (&o != &g && !small(o) && gap(g, o) < mine) { mine = gap(g, o); mine_g = &o; }
        for (const auto &o : theirs)
          if (!small(o) && gap(g, o) < other) { other = gap(g, o); other_g = &o; }
        // Near-equal: a dot or accent sits on top of the glyph it belongs to.
        const auto under = [&](const Glyph *o) { return o && o->y0 >= g.y1 - 1; };
        if (other < mine || (other_g && 4 * other <= 9 * mine && under(other_g) &&
                             !under(mine_g))) {
          own[static_cast<size_t>(g.label)] = 0;
          continue;
        }
      }
      kept.push_back(g);
    }
    glyphs.swap(kept);
  }

  // Column profile of the line's own glyphs inside the line box.
  const std::vector<int> col = col_profile(own);
  // A column with a pixel or two of glyph (an anti-aliased serif) is still a
  // gap between glyphs.
  const int noise = h / 24;

  const int n = static_cast<int>(words.size());
  auto at = [&](float frac) {
    return std::clamp(static_cast<int>(std::lround(frac * static_cast<float>(w))),
                      0, w - 1);
  };
  // Each word's column span [left, right) in line-crop coordinates.
  std::vector<int> left(static_cast<size_t>(n)), right(static_cast<size_t>(n));
  std::vector<int> gaps;
  for (int k = 1; k < n; ++k) {
    const int lo = at(words[static_cast<size_t>(k - 1)].last);
    const int hi = std::max(at(words[static_cast<size_t>(k)].first) + 1, lo + 1);
    // Across a clean gap the window's floor is 0; an underline running
    // through it raises the floor to the line's own thickness.
    const int floor_ink = std::max(
        noise, *std::min_element(col.begin() + lo, col.begin() + hi));
    const auto [gb, ge] = widest_run(col, lo, hi, floor_ink);
    right[static_cast<size_t>(k - 1)] = gb;
    left[static_cast<size_t>(k)] = ge;
    if (ge > gb) gaps.push_back(ge - gb);
  }
  // Ink past a clear run as wide as this line's own word spaces (at least a
  // third of the line height) is a neighbour's; anything closer -- a period
  // set apart in old justified print -- still belongs to the line. Where the
  // line's ink runs to the box's edge, the span runs on into the margin.
  int margin_gap = std::max(2, h / 3);
  int word_space = margin_gap;
  if (!gaps.empty()) {
    std::nth_element(gaps.begin(), gaps.begin() + gaps.size() / 2, gaps.end());
    word_space = std::max(2, gaps[gaps.size() / 2]);
    margin_gap = std::max(margin_gap, word_space);
  }
  left[0] = outer_edge(col, at(words[0].first), -1, margin_gap, noise);
  if (left[0] == 0) left[0] = -m;
  auto &last_right = right[static_cast<size_t>(n - 1)];
  last_right = outer_edge(col, at(words[static_cast<size_t>(n - 1)].last), +1,
                          margin_gap, noise);
  if (last_right == w) last_right = w + m;

  // A box that stops short of its line clips the first or last letter, or
  // leaves it just outside. Where the line's ink runs to the box's end, the
  // glyphs continuing it across the margin are the line's too: in its band,
  // each closer to the one before than the line's own word space, and in no
  // other line's box. A stroke running past both the box's top and bottom is
  // a rule, and one the crop's side cuts off is too far out to tell.
  if (!glyphs.empty() && (left[0] == -m || last_right == w + m)) {
    if (comps.band_px.empty()) band_boxed(comps);
    std::vector<Glyph> cand;
    for (int c = 1; c < ncomp; ++c) {
      const auto cu = static_cast<size_t>(c);
      const int area = comps.area[cu];
      const int x0 = comps.x0[cu];
      const int x1 = x0 + comps.width[cu];
      const int y0 = comps.y0[cu];
      const int y1 = y0 + comps.height[cu];
      if (own[cu] || area < min_area || y0 == 0 || y1 >= H || (x0 >= m && x1 <= m + w) ||
          (y0 < m && y1 > m + h) || x0 == 0 || x1 >= W ||
          2 * comps.claimed_px[cu] >= area || 2 * comps.band_px[cu] < area ||
          2 * comps.boxed_px[cu] >= area)
        continue;
      cand.push_back({x0, y0, x1, y1, x0, x1, comps.cx[cu], c});
    }
    int lo = W, hi = 0;
    for (const auto &g : glyphs) { lo = std::min(lo, g.x0); hi = std::max(hi, g.x1); }
    if (last_right == w + m) {
      std::sort(cand.begin(), cand.end(),
                [](const Glyph &a, const Glyph &b) { return a.x0 < b.x0; });
      for (const auto &g : cand) {
        if (g.x1 <= m + w || own[static_cast<size_t>(g.label)]) continue;
        if (g.x0 - hi >= word_space) break;
        glyphs.push_back(g);
        own[static_cast<size_t>(g.label)] = 1;
        hi = std::max(hi, g.x1);
      }
    }
    if (left[0] == -m) {
      std::sort(cand.begin(), cand.end(),
                [](const Glyph &a, const Glyph &b) { return a.x1 > b.x1; });
      for (const auto &g : cand) {
        if (g.x0 >= m || own[static_cast<size_t>(g.label)]) continue;
        if (lo - g.x1 >= word_space) break;
        glyphs.push_back(g);
        own[static_cast<size_t>(g.label)] = 1;
        lo = std::min(lo, g.x0);
      }
    }
  }

  const auto pt = [&](float u, float v) {
    const auto p = apply_crop_transform_inv(ct, u, v);
    return std::array<int, 2>{static_cast<int>(std::lround(p[0])),
                              static_cast<int>(std::lround(p[1]))};
  };
  // Word spans in extended coordinates.
  std::vector<int> sl(static_cast<size_t>(n)), sr(static_cast<size_t>(n));
  for (int k = 0; k < n; ++k) {
    sl[static_cast<size_t>(k)] = left[static_cast<size_t>(k)] + m;
    sr[static_cast<size_t>(k)] =
        std::max(right[static_cast<size_t>(k)] + m, sl[static_cast<size_t>(k)] + 1);
  }
  // Each glyph belongs whole to the word nearest its ink's centre -- a
  // hairline tail or hook reaching back across the gap (a "j", a CJK "了")
  // included. Only a stroke running on into the next word by more than half
  // a line height (an underline, a dash across words) is shared, cut at each
  // span's sides. Glyphs centred past the outer spans are a neighbour's.
  struct Part { int word, x0, x1, y0, y1; };
  std::vector<Part> parts;
  const int overhang = std::max(2, h / 2);
  for (const auto &g : glyphs) {
    if (g.cx < sl.front() || g.cx >= sr.back()) continue;
    int owner = 0, best = std::numeric_limits<int>::max();
    for (int k = 0; k < n; ++k) {
      const auto ku = static_cast<size_t>(k);
      const int dist = std::max({0, sl[ku] - g.cx, g.cx - (sr[ku] - 1)});
      if (dist < best) { best = dist; owner = k; }
    }
    const auto ou = static_cast<size_t>(owner);
    if (g.core_x0 >= sl[ou] - overhang && g.core_x1 <= sr[ou] + overhang) {
      parts.push_back({owner, g.x0, g.x1, g.y0, g.y1});
      continue;
    }
    for (int k = 0; k < n; ++k) {
      const auto ku = static_cast<size_t>(k);
      if (g.core_x1 <= sl[ku] || g.core_x0 >= sr[ku]) continue;
      parts.push_back({k, g.core_x0 < sl[ku] ? sl[ku] : g.x0,
                       g.core_x1 > sr[ku] ? sr[ku] : g.x1, g.y0, g.y1});
    }
  }

  out.reserve(static_cast<size_t>(n));
  for (int k = 0; k < n; ++k) {
    const int wl = sl[static_cast<size_t>(k)], wr = sr[static_cast<size_t>(k)];
    int x0 = wr, x1 = wl, y0 = h + 2 * m, y1 = 0;
    for (const auto &p : parts) {
      if (p.word != k) continue;
      x0 = std::min(x0, p.x0);
      x1 = std::max(x1, p.x1);
      y0 = std::min(y0, p.y0);
      y1 = std::max(y1, p.y1);
    }
    if (x0 >= x1) {  // no ink of its own: the span across the line box
      if (unresolved_line) {
        out.push_back({words[static_cast<size_t>(k)].text,
                       words[static_cast<size_t>(k)].score, *unresolved_line});
        continue;
      }
      x0 = std::max(wl, m);
      x1 = std::min(wr, m + w);
      y0 = m;
      y1 = m + h;
    }
    // One pixel of margin for the anti-aliased rim the threshold drops; back
    // to line-crop coordinates (the right/bottom edges are exclusive).
    const float u0 = static_cast<float>(x0 - 1 - m);
    const float u1 = static_cast<float>(x1 - m);
    const float v0 = static_cast<float>(y0 - 1 - m);
    const float v1 = static_cast<float>(y1 - m);
    OCRWord word;
    word.text = words[static_cast<size_t>(k)].text;
    word.confidence = words[static_cast<size_t>(k)].score;
    // Same point order as the line box: point i goes where the line box's
    // point i went in the crop.
    const std::array<std::array<float, 2>, 4> corner{
        {{u0, v0}, {u1, v0}, {u1, v1}, {u0, v1}}};
    for (int i = 0; i < 4; ++i) {
      const auto &[u, v] = corner[static_cast<size_t>(crop_corner(ct.vertical, i))];
      word.box[static_cast<size_t>(i)] = pt(u, v);
    }
    out.push_back(std::move(word));
  }
  return out;
}

}  // namespace detail

std::vector<OCRWord> locate_words(const cv::Mat &img, const Box &line,
                                  const std::vector<CtcWord> &words,
                                  const std::vector<Box> &page_lines) {
  if (words.empty() || img.empty()) return {};
  const LinePlan plan = detail::plan_line(line, img.size(), page_lines);
  const int w = plan.w, h = plan.h, m = plan.m, W = plan.W, H = plan.H;
  const cv::Rect &core = plan.core;

  cv::Mat crop;
  if (plan.sub)
    crop = img(*plan.sub);
  else
    cv::warpPerspective(img, crop, plan.to_page, cv::Size(W, H),
                        cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
  cv::Mat gray;
  if (crop.channels() == 3)
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
  else if (crop.channels() == 4)
    cv::cvtColor(crop, gray, cv::COLOR_BGRA2GRAY);
  else
    gray = crop;
  if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8U);

  // Ink: Otsu's threshold of the line itself, applied to the margins too.
  cv::Mat ink, scratch;
  const double thr = cv::threshold(gray(core), scratch, 0, 1,
                                   cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  cv::threshold(gray, ink, thr, 1, cv::THRESH_BINARY_INV);
  // Dark text on light paper is the norm. A box that is mostly dark holds
  // either light text on a dark ground or heavy type the box fits tightly,
  // and the ground is the side that surrounds the other.
  if (2 * cv::countNonZero(ink(core)) > w * h && !light_is_ground(ink, core))
    ink = 1 - ink;

  // Other detected lines reaching into the crop claim the ink inside their own
  // box that lies nearer their middle than this line's -- a headline's letters
  // stay above the small line's box, so it can never take them. Ink claimed
  // beyond the other line's middle ("deep") means a blob runs right through
  // that line: glyphs of two lines touching, not one glyph.
  cv::Mat claimed = cv::Mat::zeros(ink.size(), CV_8U);  // 1 + claiming line
  cv::Mat deep = cv::Mat::zeros(ink.size(), CV_8U);
  cv::Mat boxed = cv::Mat::zeros(ink.size(), CV_8U);  // inside any other box
  bool any_claim = false;
  const float own_mid = plan.own_mid;
  std::vector<const detail::Neighbour *> others;  // the claiming ones; tag = index + 1
  for (const auto &nb : plan.neighbours) {
    const auto &runs = nb.runs;
    for (size_t i = 0; i < runs.rows.size(); ++i) {
      uchar *br = boxed.ptr<uchar>(runs.y0 + static_cast<int>(i));
      const auto &r = runs.rows[i];
      for (int x = r[0]; x <= r[1]; ++x) br[x] = 1;
      for (int x = r[2]; x <= r[3]; ++x) br[x] = 1;
    }
    if (!nb.claims) continue;
    others.push_back(&nb);
    const auto tag = static_cast<uchar>(others.size());
    const cv::Point2f L = nb.left;
    const float slope = nb.slope;
    const auto mid_at = [&](float x) { return L.y + slope * (x - L.x); };
    for (int y = nb.wy0; y < nb.wy1; ++y) {
      const uchar *ir = ink.ptr<uchar>(y);
      uchar *cr = claimed.ptr<uchar>(y), *dr = deep.ptr<uchar>(y);
      for (int x = nb.wx0; x < nb.wx1; ++x) {
        if (!ir[x] || !runs.contains(x, y)) continue;
        const float fy = static_cast<float>(y), oy = mid_at(static_cast<float>(x));
        if (std::abs(fy - oy) >= std::abs(fy - own_mid)) continue;
        if (cr[x] && std::abs(fy - others[cr[x] - 1u]->left.y -
                              others[cr[x] - 1u]->slope *
                                  (static_cast<float>(x) - others[cr[x] - 1u]->left.x)) <=
                         std::abs(fy - oy))
          continue;  // a nearer line already claims it
        cr[x] = tag;
        any_claim = true;
        if ((oy - own_mid) * (fy - oy) > 0) dr[x] = 1;
      }
    }
  }

  // Glyphs = connected ink components. One is the line's own when it stays
  // within the margin and most of it lies inside the line box -- a glyph the
  // box clips stays, a letter of the line above reaching down with its
  // descender does not -- or, for a short stroke, when its middle is on the
  // box's edge (the bar under a "⊆").
  // Specks are no glyph at all.
  cv::Mat labels, stats, centroids;
  int ncomp = cv::connectedComponentsWithStats(ink, labels, stats, centroids, 8,
                                               CV_32S);
  // A blob reaching deep into another line may hold glyphs of both, joined
  // by bleed. Where its row profile shows a clear valley between the two
  // middles -- a hairline of bleed -- it is cut there, and the part past the
  // cut leaves this line's ink.
  std::vector<int> claimed_px(static_cast<size_t>(ncomp), 0);
  if (any_claim) {
    std::vector<int> deep_px(static_cast<size_t>(ncomp), 0);
    std::vector<int> taken(static_cast<size_t>(ncomp), 0);
    std::vector<std::vector<int>> by_line(static_cast<size_t>(ncomp));
    for (int y = 0; y < H; ++y) {
      const int *lr = labels.ptr<int>(y);
      const uchar *dr = deep.ptr<uchar>(y), *cr = claimed.ptr<uchar>(y);
      for (int x = 0; x < W; ++x) {
        if (cr[x]) ++taken[static_cast<size_t>(lr[x])];
        if (dr[x]) {
          const auto c = static_cast<size_t>(lr[x]);
          ++deep_px[c];
          if (by_line[c].empty()) by_line[c].assign(others.size() + 1, 0);
          ++by_line[c][cr[x]];
        }
      }
    }
    bool split = false;
    for (int c = 1; c < ncomp; ++c) {
      const auto cu = static_cast<size_t>(c);
      const int area = stats.at<int>(c, cv::CC_STAT_AREA);
      // Both lines' ink: some reaching past the other line's middle, and a
      // real share left on this line's side (a glyph of the other line alone
      // is all its own and is simply the other line's).
      if (!detail::bleed_candidate(area, deep_px[cu], taken[cu])) continue;
      const auto j = static_cast<size_t>(
          std::max_element(by_line[cu].begin() + 1, by_line[cu].end()) - by_line[cu].begin() - 1);
      const detail::Neighbour &o = *others[j];
      // Position of a pixel between this line's middle (0) and the other's (1).
      const auto frac = [&](int x, int y) {
        const float oy = o.left.y + o.slope * (static_cast<float>(x) - o.left.x);
        return (static_cast<float>(y) - own_mid) / (oy - own_mid);
      };
      const int cx0 = stats.at<int>(c, cv::CC_STAT_LEFT), cy0 = stats.at<int>(c, cv::CC_STAT_TOP);
      const int cx1 = cx0 + stats.at<int>(c, cv::CC_STAT_WIDTH);
      const int cy1 = cy0 + stats.at<int>(c, cv::CC_STAT_HEIGHT);
      const float cx = 0.5f * static_cast<float>(cx0 + cx1);
      const float span = std::abs(o.left.y + o.slope * (cx - o.left.x) - own_mid);
      // Bins at least two rows tall: finer ones leave empty bins between rows
      // of one continuous stroke, which would pass for a gap. The profile runs
      // on past the other line's middle, so its glyph counts whole; the cut
      // itself lies between the two middles.
      const int nb = std::max(4, static_cast<int>(span / 2.0f));
      const int nb_all = nb + nb / 2;
      std::vector<int> prof(static_cast<size_t>(nb_all), 0);
      for (int y = cy0; y < cy1; ++y) {
        const int *lr = labels.ptr<int>(y);
        for (int x = cx0; x < cx1; ++x) {
          if (lr[x] != c) continue;
          const float f = frac(x, y);
          if (f >= 0.0f && f < 1.5f)
            ++prof[std::min(static_cast<size_t>(f * static_cast<float>(nb)), prof.size() - 1)];
        }
      }
      // The valley lies inside the blob, with its ink on both sides.
      size_t lo = 0, hi = prof.size();
      while (lo < hi && prof[lo] == 0) ++lo;
      while (hi > lo && prof[hi - 1] == 0) --hi;
      if (hi - lo < 3) continue;
      size_t best = lo + 1;
      for (size_t k = lo + 1; k + 1 < hi && k < static_cast<size_t>(nb); ++k)
        if (prof[k] < prof[best] ||
            (prof[k] == prof[best] &&
             std::abs(static_cast<int>(k) - nb / 2) < std::abs(static_cast<int>(best) - nb / 2)))
          best = k;
      const int before = *std::max_element(prof.begin() + static_cast<std::ptrdiff_t>(lo),
                                           prof.begin() + static_cast<std::ptrdiff_t>(best));
      const int after = *std::max_element(prof.begin() + static_cast<std::ptrdiff_t>(best) + 1,
                                          prof.begin() + static_cast<std::ptrdiff_t>(hi));
      // No clear valley -- a bridge as thick as a stroke, or this line's own
      // ascender reaching into an overlapping box -- keeps the blob whole.
      if (3 * prof[best] > std::min(before, after)) continue;
      const float cut = (static_cast<float>(best) + 0.5f) / static_cast<float>(nb);
      for (int y = cy0; y < cy1; ++y) {
        const int *lr = labels.ptr<int>(y);
        uchar *ir = ink.ptr<uchar>(y);
        for (int x = cx0; x < cx1; ++x)
          if (lr[x] == c && frac(x, y) > cut) ir[x] = 0;
      }
      split = true;
    }
    if (split)
      ncomp = cv::connectedComponentsWithStats(ink, labels, stats, centroids, 8,
                                               CV_32S);
    claimed_px.assign(static_cast<size_t>(ncomp), 0);
    for (int y = 0; y < H; ++y) {
      const int *lr = labels.ptr<int>(y);
      const uchar *cr = claimed.ptr<uchar>(y), *ir = ink.ptr<uchar>(y);
      for (int x = 0; x < W; ++x)
        if (cr[x] && ir[x]) ++claimed_px[static_cast<size_t>(lr[x])];
    }
  }

  // Canonical order: by each component's first pixel in raster order, which
  // does not depend on how OpenCV numbers them (and matches the GPU path).
  std::vector<int> first(static_cast<size_t>(ncomp), 0);
  for (int c = 1; c < ncomp; ++c) {
    const int y0 = stats.at<int>(c, cv::CC_STAT_TOP), x0 = stats.at<int>(c, cv::CC_STAT_LEFT);
    const int *row = labels.ptr<int>(y0);
    int x = x0;
    while (row[x] != c) ++x;
    first[static_cast<size_t>(c)] = y0 * W + x;
  }
  std::vector<int> order(static_cast<size_t>(ncomp));  // canonical -> OpenCV label
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin() + 1, order.end(), [&](int a, int b) {
    return first[static_cast<size_t>(a)] < first[static_cast<size_t>(b)];
  });
  std::vector<int> canon(static_cast<size_t>(ncomp));  // OpenCV label -> canonical
  for (int k = 0; k < ncomp; ++k) canon[static_cast<size_t>(order[static_cast<size_t>(k)])] = k;

  Components comps;
  comps.n = ncomp;
  const auto nsz = static_cast<size_t>(ncomp);
  for (auto *v : {&comps.area, &comps.x0, &comps.y0, &comps.width, &comps.height, &comps.cx,
                  &comps.in_core, &comps.claimed_px})
    v->assign(nsz, 0);
  comps.core_x0.assign(nsz, w + 2 * m);
  comps.core_x1.assign(nsz, -1);
  for (int k = 1; k < ncomp; ++k) {
    const int c = order[static_cast<size_t>(k)];
    const auto ku = static_cast<size_t>(k);
    comps.area[ku] = stats.at<int>(c, cv::CC_STAT_AREA);
    comps.x0[ku] = stats.at<int>(c, cv::CC_STAT_LEFT);
    comps.y0[ku] = stats.at<int>(c, cv::CC_STAT_TOP);
    comps.width[ku] = stats.at<int>(c, cv::CC_STAT_WIDTH);
    comps.height[ku] = stats.at<int>(c, cv::CC_STAT_HEIGHT);
    comps.cx[ku] = static_cast<int>(centroids.at<double>(c, 0));
    comps.claimed_px[ku] = claimed_px[static_cast<size_t>(c)];
  }
  for (int y = m; y < m + h; ++y) {
    const int *row = labels.ptr<int>(y);
    for (int x = m; x < m + w; ++x) {
      const auto k = static_cast<size_t>(canon[static_cast<size_t>(row[x])]);
      ++comps.in_core[k];
      comps.core_x0[k] = std::min(comps.core_x0[k], x);
      comps.core_x1[k] = std::max(comps.core_x1[k], x + 1);
    }
  }
  const auto col_profile = [&](const std::vector<char> &own) {
    std::vector<int> col(static_cast<size_t>(w), 0);
    for (int y = m; y < m + h; ++y) {
      const int *row = labels.ptr<int>(y);
      for (int x = m; x < m + w; ++x)
        if (own[static_cast<size_t>(canon[static_cast<size_t>(row[x])])])
          ++col[static_cast<size_t>(x - m)];
    }
    return col;
  };
  const auto band_boxed = [&](Components &cs) {
    cs.band_px.assign(nsz, 0);
    cs.boxed_px.assign(nsz, 0);
    for (int y = 0; y < H; ++y) {
      const int *row = labels.ptr<int>(y);
      const uchar *br = boxed.ptr<uchar>(y);
      const int in_band = y >= m && y < m + h ? 1 : 0;
      for (int x = 0; x < W; ++x) {
        const auto k = static_cast<size_t>(canon[static_cast<size_t>(row[x])]);
        cs.band_px[k] += in_band;
        cs.boxed_px[k] += br[x];
      }
    }
  };
  return detail::words_from_components(plan, comps, words, col_profile, band_boxed);
}

}  // namespace turbo_ocr::recognition
