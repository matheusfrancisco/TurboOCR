#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <vector>

#include "turbo_ocr/common/geometry/box.h"

namespace turbo_ocr::detection {

// Detection post-processing free functions (shared between GPU and CPU detectors)

// Rasterize the same filled polygon for CPU and GPU scoring. Returns its
// clipped map ROI, or an empty ROI for a degenerate contour. Scratch is owned
// by the caller. The mask is ROI-local CV_8UC1, with values 0 or 1.
[[nodiscard]] cv::Rect filled_contour_mask(int width, int height,
                                           const std::vector<cv::Point> &contour,
                                           std::vector<cv::Point> &shifted_buf,
                                           cv::Mat &mask_buf);

// Compute mean probability inside contour polygon.
// shifted_buf and mask_buf are caller-owned scratch buffers for thread-safety.
[[nodiscard]] float box_score_fast(const cv::Mat &pred_map,
                                   const std::vector<cv::Point> &contour,
                                   std::vector<cv::Point> &shifted_buf,
                                   cv::Mat &mask_buf);

// DBNet's unclip as PaddleOCR runs it for quadrilateral boxes: the region's
// min-area rectangle offset outward by d = area * ratio / perimeter. Offset
// with round joins, a rectangle's min-area rectangle is that rectangle grown
// by d on every side, so the offset polygon itself is never built. (Offsetting
// the region's outline instead grows it less: a jagged outline has less area
// and more perimeter than its rectangle.)
[[nodiscard]] cv::RotatedRect unclip_rect(const cv::RotatedRect &rect, float unclip_ratio);

// Every detector path's last step for a region whose score passed: unclip its
// min-area rectangle, reject it when the result is thinner than
// `min_unclipped_side` or smaller than 4 px a side in the original image, and
// return its corners there: [tl, tr, br, bl], scaled by 1/ratio before
// rounding, clamped to the image.
[[nodiscard]] std::optional<Box> region_to_box(const cv::RotatedRect &rect, float unclip_ratio,
                                               float min_unclipped_side, float ratio_w,
                                               float ratio_h, int orig_w, int orig_h);

// Order 4 quad corners in place as [tl, tr, br, bl] — PaddleOCR's convention:
// stable sort by x (ties keep original order, mirroring Python sorted()), the
// left pair splits {tl,bl} by y, the right pair {tr,br} by y. SINGLE source of
// truth for every detector path, so the orderings can never drift apart.
void order_quad_tl_tr_br_bl(float xs[4], float ys[4]) noexcept;

// Extract boxes from contours -- the shared loop used by both GPU and CPU detectors.
[[nodiscard]] std::vector<Box> extract_boxes_from_bitmap(
    const cv::Mat &pred_map, cv::Mat &bitmap,
    int orig_h, int orig_w, int resize_h, int resize_w,
    float det_db_box_thresh, float det_db_unclip_ratio,
    float min_box_side, float min_unclipped_side,
    std::vector<cv::Point> &shifted_buf, cv::Mat &mask_buf,
    std::vector<std::vector<cv::Point>> &contours_buf,
    std::vector<cv::Vec4i> &hierarchy_buf);

} // namespace turbo_ocr::detection
