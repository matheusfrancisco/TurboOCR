#pragma once

// The stages of recognition::locate_words, shared by the CPU implementation
// (word_boxes.cpp) and the GPU one (word_boxes_gpu.*). The pixel work differs
// between the two; everything that decides -- the line's geometry, and what
// its connected components mean -- is this one code, so both place the same
// words to the pixel.

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/common/geometry/perspective.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/recognition/ctc_decode.h"

namespace turbo_ocr::recognition::detail {

// The line is re-sampled at its own height in pixels, kept within these
// bounds: tiny text is upsampled so a gap still spans a few columns, huge
// text is capped to bound the work.
inline constexpr int kMinLineH = 16;
inline constexpr int kMaxLineH = 160;
inline constexpr int kMaxLineW = 16384;
// Other lines' boxes one line's analysis takes into account; tags are a byte.
inline constexpr std::size_t kMaxClaimers = 254;

// A convex quadrilateral rasterized into a W x H mask exactly as
// cv::fillConvexPoly(mask, q, 4, color, LINE_8, 0) paints it: per row from
// `y0`, at most two inclusive runs [a0, a1] and [b0, b1] (an empty run has
// its end before its start).
struct QuadRuns {
  int y0 = 0;
  std::vector<std::array<std::int16_t, 4>> rows;
  [[nodiscard]] bool contains(int x, int y) const noexcept {
    const int i = y - y0;
    if (i < 0 || i >= static_cast<int>(rows.size())) return false;
    const auto &r = rows[static_cast<std::size_t>(i)];
    return (x >= r[0] && x <= r[1]) || (x >= r[2] && x <= r[3]);
  }
};
// False when some row needs more than two runs (not seen for real boxes).
bool quad_runs(const std::array<cv::Point, 4> &q, int W, int H, QuadRuns &out);

// Another detected line reaching into this line's crop.
struct Neighbour {
  QuadRuns runs;  // its box in crop pixels: the `boxed` mask
  // A line running this way with its own middle: it claims the ink of its
  // box nearer its middle than this line's, inside the window below.
  bool claims = false;
  int wx0 = 0, wy0 = 0, wx1 = 0, wy1 = 0;  // claim window [wx0, wx1) x [wy0, wy1)
  cv::Point2f left;                        // its midline in crop coordinates
  float slope = 0.0f;
};

// Everything about one line that does not depend on its pixels.
struct LinePlan {
  bool vertical = false;
  int h = 0, w = 0, m = 0, W = 0, H = 0;  // line box h x w; crop W x H with margin m
  cv::Rect core;                          // the line box inside the crop
  CropTransform ct{};
  cv::Matx33f to_page;           // crop pixel -> page pixel
  std::optional<cv::Rect> sub;   // the crop as an exact sub-image of the page
  float own_mid = 0.0f;          // the line's middle row in the crop
  std::vector<Neighbour> neighbours;  // in page order
  bool runs_ok = true;                // every neighbour box rasterized
};
LinePlan plan_line(const Box &line, cv::Size page, const std::vector<Box> &page_lines);

// A line's ink components in canonical order: by the first pixel in raster
// order of the crop. Index 0 is the background.
struct Components {
  int n = 1;
  std::vector<int> area, x0, y0, width, height;
  std::vector<int> cx;  // centroid column, truncated
  std::vector<int> in_core, core_x0, core_x1, claimed_px;
  // Pixels inside the line box's rows, and inside another line's box: needed
  // only for the line-end pass, so a provider may fill them on demand.
  std::vector<int> band_px, boxed_px;
};

// Column profile inside the line box of the components flagged `own`.
using ColProfileFn = std::function<std::vector<int>(const std::vector<char> &own)>;
// Fills comps.band_px / boxed_px when they are empty.
using BandBoxedFn = std::function<void(Components &)>;

// The word boxes from the components: which glyphs are the line's, where each
// word boundary falls, each word's box. Shared by the CPU and GPU paths.
std::vector<OCRWord> words_from_components(const LinePlan &plan, Components &comps,
                                           const std::vector<CtcWord> &words,
                                           const ColProfileFn &col_profile,
                                           const BandBoxedFn &band_boxed,
                                           const Box *unresolved_line = nullptr);

// A glyph of two lines that touch: the CPU path cuts it at the valley of its
// row profile (bleed split), so a GPU pass that finds one hands the line over.
[[nodiscard]] inline bool bleed_candidate(int area, int deep_px, int taken) noexcept {
  return deep_px >= std::max(3, area / 50) && 4 * (area - taken) >= area;
}

}  // namespace turbo_ocr::recognition::detail
