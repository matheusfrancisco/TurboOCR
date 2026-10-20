#include "turbo_ocr/detection/det_postprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include "turbo_ocr/common/cv_geometry.h"

using turbo_ocr::Box;

namespace turbo_ocr::detection {

cv::Rect filled_contour_mask(int w, int h,
                     const std::vector<cv::Point> &contour,
                     std::vector<cv::Point> &shifted_buf,
                     cv::Mat &mask_buf) {
  if (w <= 0 || h <= 0) return {};

  // Clip the contour's axis-aligned extent to the map. Contour points come
  // from findContours on the bitmap so they are already in [0,w-1]x[0,h-1];
  // the clamps only guard the GPU-CCL callers that shift ROI-local points.
  int xmin = w, xmax = 0, ymin = h, ymax = 0;
  for (const auto &pt : contour) {
    xmin = std::min(xmin, std::max(0, pt.x));
    xmax = std::max(xmax, std::min(w - 1, pt.x));
    ymin = std::min(ymin, std::max(0, pt.y));
    ymax = std::max(ymax, std::min(h - 1, pt.y));
  }
  if (xmax <= xmin || ymax <= ymin)
    return {};

  // Shift contour into local ROI coords using the caller-owned scratch buffer
  // (per-thread) so no per-call heap allocation occurs.
  shifted_buf.clear();
  shifted_buf.reserve(contour.size());
  for (const auto &pt : contour)
    shifted_buf.emplace_back(pt.x - xmin, pt.y - ymin);

  const int mask_w = xmax - xmin + 1;
  const int mask_h = ymax - ymin + 1;
  if (mask_buf.rows != mask_h || mask_buf.cols != mask_w ||
      mask_buf.type() != CV_8UC1)
    mask_buf.create(mask_h, mask_w, CV_8UC1);
  mask_buf.setTo(0);

  const cv::Point *pts_ptr = shifted_buf.data();
  const int npts = static_cast<int>(shifted_buf.size());
  cv::fillPoly(mask_buf, &pts_ptr, &npts, 1, cv::Scalar(1));

  return {xmin, ymin, mask_w, mask_h};
}

float box_score_fast(const cv::Mat &pred_map,
                     const std::vector<cv::Point> &contour,
                     std::vector<cv::Point> &shifted_buf,
                     cv::Mat &mask_buf) {
  const auto r = filled_contour_mask(pred_map.cols, pred_map.rows, contour,
                                    shifted_buf, mask_buf);
  if (r.empty()) return 0.0f;
  return static_cast<float>(cv::mean(pred_map(r), mask_buf)[0]);
}

void order_quad_tl_tr_br_bl(float xs[4], float ys[4]) noexcept {
  // Stable insertion sort by x (strict '>' keeps ties in original order,
  // mirroring Python sorted()'s stability exactly).
  for (int i = 1; i < 4; ++i) {
    const float kx = xs[i], ky = ys[i];
    int j = i - 1;
    while (j >= 0 && xs[j] > kx) {
      xs[j + 1] = xs[j];
      ys[j + 1] = ys[j];
      --j;
    }
    xs[j + 1] = kx;
    ys[j + 1] = ky;
  }
  const int tl = (ys[0] < ys[1]) ? 0 : 1;
  const int bl = 1 - tl;
  const int tr = (ys[2] < ys[3]) ? 2 : 3;
  const int br = 5 - tr; // {2,3} sum to 5

  const float ox[4] = {xs[tl], xs[tr], xs[br], xs[bl]};
  const float oy[4] = {ys[tl], ys[tr], ys[br], ys[bl]};
  for (int k = 0; k < 4; ++k) { xs[k] = ox[k]; ys[k] = oy[k]; }
}

cv::RotatedRect unclip_rect(const cv::RotatedRect &rect, float unclip_ratio) {
  const float w = rect.size.width, h = rect.size.height;
  const float d = w + h > 0.0f ? unclip_ratio * w * h / (2.0f * (w + h)) : 0.0f;
  return {rect.center, cv::Size2f(w + 2.0f * d, h + 2.0f * d), rect.angle};
}

std::optional<Box> region_to_box(const cv::RotatedRect &rect, float unclip_ratio,
                                 float min_unclipped_side, float ratio_w,
                                 float ratio_h, int orig_w, int orig_h) {
  const cv::RotatedRect grown = unclip_rect(rect, unclip_ratio);
  if (std::min(grown.size.width, grown.size.height) < min_unclipped_side)
    return std::nullopt;

  std::array<cv::Point2f, 4> pts;
  grown.points(pts.data());
  float xs[4], ys[4];
  for (int k = 0; k < 4; ++k) { xs[k] = pts[k].x; ys[k] = pts[k].y; }
  order_quad_tl_tr_br_bl(xs, ys);
  Box box;
  for (int k = 0; k < 4; ++k)
    box[k] = {std::clamp(static_cast<int>(std::round(xs[k] / ratio_w)), 0, orig_w - 1),
              std::clamp(static_cast<int>(std::round(ys[k] / ratio_h)), 0, orig_h - 1)};

  const int dx01 = box[0][0] - box[1][0], dy01 = box[0][1] - box[1][1];
  const int dx03 = box[0][0] - box[3][0], dy03 = box[0][1] - box[3][1];
  if (dx01 * dx01 + dy01 * dy01 < 16 || dx03 * dx03 + dy03 * dy03 < 16)
    return std::nullopt;
  return box;
}

std::vector<Box> extract_boxes_from_bitmap(
    const cv::Mat &pred_map, cv::Mat &bitmap,
    int orig_h, int orig_w, int resize_h, int resize_w,
    float det_db_box_thresh, float det_db_unclip_ratio,
    float min_box_side, float min_unclipped_side,
    std::vector<cv::Point> &shifted_buf, cv::Mat &mask_buf,
    std::vector<std::vector<cv::Point>> &contours_buf,
    std::vector<cv::Vec4i> &hierarchy_buf) {

  const float ratio_h = static_cast<float>(resize_h) / orig_h;
  const float ratio_w = static_cast<float>(resize_w) / orig_w;

  std::vector<Box> boxes;
  boxes.reserve(256);

  contours_buf.clear();
  hierarchy_buf.clear();
  cv::findContours(bitmap, contours_buf, hierarchy_buf, cv::RETR_LIST,
                   cv::CHAIN_APPROX_SIMPLE);

  // DB candidate budget. Must equal kernels::kMaxGpuComponents (the GPU CCL
  // path's component cap) so CPU and GPU detectors truncate identically.
  static constexpr int kMaxCandidates = 3000;
  const int total = static_cast<int>(contours_buf.size());
  const int num_contours = std::min(total, kMaxCandidates);
  if (total > kMaxCandidates) {
    std::cerr << "[Det] WARNING: truncated to " << kMaxCandidates
              << " contours (found " << total << ")\n";
  }

  for (int i = 0; i < num_contours; ++i) {
    const auto &contour = contours_buf[i];
    if (contour.size() <= 2)
      continue;

    const cv::Rect br = cv::boundingRect(contour);
    if (br.width < 3 || br.height < 3)
      continue;

    // Filters below are commutative (all reject via `continue`), so the kept
    // set is order-independent. The min-side test runs before the score to
    // match PaddleOCR's ordering and to skip the area-proportional
    // fillPoly+mean of box_score_fast on candidates too thin to survive.
    const cv::RotatedRect rect = cv::minAreaRect(contour);
    if (std::min(rect.size.width, rect.size.height) < min_box_side)
      continue;

    const float score = box_score_fast(pred_map, contour, shifted_buf, mask_buf);
    if (score < det_db_box_thresh)
      continue;

    if (auto box = region_to_box(rect, det_db_unclip_ratio, min_unclipped_side,
                                 ratio_w, ratio_h, orig_w, orig_h))
      boxes.push_back(*box);
  }

  return boxes;
}

} // namespace turbo_ocr::detection
