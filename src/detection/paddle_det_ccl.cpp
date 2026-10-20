#include "turbo_ocr/detection/paddle_det.h"
#include "turbo_ocr/detection/det_config.h"
#include "turbo_ocr/kernels/kernels.h"
#include "turbo_ocr/detection/det_postprocess.h"

#include "turbo_ocr/common/errors.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ranges>

#include "turbo_ocr/common/cv_geometry.h"

using turbo_ocr::engine::TrtEngine;

namespace turbo_ocr::detection {

// GPU CCL path: connected component labeling on GPU, then each component's
// real contour from the bitmap within its bbox, for its min-area rectangle.
std::vector<Box>
PaddleDet::run_gpu_ccl(const float *d_pred, const uint8_t *d_bitmap,
                        int resize_h, int resize_w,
                        int orig_h, int orig_w,
                        cudaStream_t stream,
                        int content_h, int content_w) {
  if (content_h <= 0) content_h = resize_h;
  if (content_w <= 0) content_w = resize_w;
  float ratio_h = static_cast<float>(content_h) / orig_h;
  float ratio_w = static_cast<float>(content_w) / orig_w;

  int h_num_boxes = 0;
  turbo_ocr::kernels::cuda_gpu_ccl_detect(
      d_bitmap, d_pred, resize_w, resize_h,
      0.0f, // no foreground-mean prefilter; filled-contour score decides below
      d_ccl_labels_.get(), d_ccl_compact_ids_.get(), d_ccl_id_counter_.get(),
      d_ccl_bboxes_.get(), d_ccl_num_boxes_.get(),
      h_ccl_boxes_.get(), &h_num_boxes, stream);

  std::vector<Box> boxes;
  if (h_num_boxes == 0)
    return boxes;

  // Download only the derived bitmap for contour extraction; the probability
  // map stays on device for filled-contour scoring.
  // We need the bitmap for per-ROI findContours to get accurate polygon contours.
  const size_t bitmap_pixels = static_cast<size_t>(resize_h) * resize_w;
  if (bitmap_pixels > h_bitmap_pixels_) [[unlikely]] {
    h_bitmap_ = CudaHostPtr<uint8_t>(bitmap_pixels);
    h_bitmap_pixels_ = bitmap_pixels;
  }
  cv::Mat bitmap(resize_h, resize_w, CV_8UC1, h_bitmap_.get());
  CUDA_CHECK(cudaMemcpyAsync(bitmap.data, d_bitmap, resize_w * resize_h,
                              cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  // The foreground-only gb.score is intentionally not used for acceptance.

  boxes.reserve(h_num_boxes);
  contours_buf_.clear();
  std::vector<cv::RotatedRect> rects;
  rects.reserve(h_num_boxes);

  // For each GPU-detected component, extract the real contour from the bitmap
  // within the bbox region, then take the same min-area-rectangle unclip as the
  // CPU path.
  for (int i = 0; i < h_num_boxes; i++) {
    const auto &gb = h_ccl_boxes_.get()[i];

    int bw = gb.xmax - gb.xmin + 1;
    int bh = gb.ymax - gb.ymin + 1;
    if (bw < 3 || bh < 3)
      continue;

    // Extract the small bitmap ROI for this component's bbox
    // Pad by 1 pixel to ensure findContours can find closed contours at edges
    int roi_x = std::max(0, gb.xmin - 1);
    int roi_y = std::max(0, gb.ymin - 1);
    int roi_x2 = std::min(resize_w - 1, gb.xmax + 1);
    int roi_y2 = std::min(resize_h - 1, gb.ymax + 1);
    int roi_w = roi_x2 - roi_x + 1;
    int roi_h = roi_y2 - roi_y + 1;

    // Clone the ROI since findContours may modify the source image
    cv::Mat roi = bitmap(cv::Rect(roi_x, roi_y, roi_w, roi_h)).clone();

    // Find contours within this small ROI (~50x20 pixels, negligible cost)
    ccl_roi_contours_buf_.clear();
    cv::findContours(roi, ccl_roi_contours_buf_, cv::RETR_LIST,
                     cv::CHAIN_APPROX_SIMPLE);

    if (ccl_roi_contours_buf_.empty())
      continue;

    // Pick the largest contour in the ROI (should be the component itself)
    const auto &best_contour = (ccl_roi_contours_buf_.size() == 1)
      ? ccl_roi_contours_buf_[0]
      : *std::ranges::max_element(ccl_roi_contours_buf_, {}, [](const std::vector<cv::Point> &c) {
          return cv::contourArea(c);
        });

    if (best_contour.size() <= 2)
      continue;

    // Shift contour from ROI-local coords to global bitmap coords
    ccl_contour_buf_.clear();
    ccl_contour_buf_.reserve(best_contour.size());
    for (const auto &pt : best_contour)
      ccl_contour_buf_.push_back(cv::Point(pt.x + roi_x, pt.y + roi_y));

    const cv::RotatedRect rect = cv::minAreaRect(ccl_contour_buf_);
    if (std::min(rect.size.width, rect.size.height) < kMinBoxSide)
      continue;
    contours_buf_.push_back(ccl_contour_buf_);
    rects.push_back(rect);
  }

  if (!contour_scorer_)
    throw InferenceError("GPU contour scorer not initialized");
  const auto scores = contour_scorer_->score(d_pred, resize_w, resize_h,
                                             contours_buf_, stream);
  for (size_t i = 0; i < rects.size(); ++i) {
    if (scores[i] < box_thresh_) continue;
    if (auto box = region_to_box(rects[i], unclip_ratio_ * unclip_scale_, kMinUnclippedSide,
                                 ratio_w, ratio_h, orig_w, orig_h))
      boxes.push_back(*box);
  }
  return boxes;
}

// GPU CCL + oriented rects (all-GPU up to the unclip).
// 1. CCL on the bitmap -> compact ids + per-component bbox, score sum, size
// 2. Label the components that pass the score/size filters
// 3. One PCA reduction per labelled component -> its oriented rectangle
// 4. Host: the shared unclip of each rectangle, scaled to the original image
std::vector<Box>
PaddleDet::run_gpu_ccl_fast(const float *d_pred, const uint8_t *d_bitmap,
                              int resize_h, int resize_w,
                              int orig_h, int orig_w,
                              cudaStream_t stream,
                              int content_h, int content_w) {
  if (content_h <= 0) content_h = resize_h;
  if (content_w <= 0) content_w = resize_w;
  float ratio_h = static_cast<float>(content_h) / orig_h;
  float ratio_w = static_cast<float>(content_w) / orig_w;

  int h_num_boxes = 0;
  int h_num_total = 0;
  turbo_ocr::kernels::cuda_gpu_ccl_detect(
      d_bitmap, d_pred, resize_w, resize_h,
      box_thresh_,
      d_ccl_labels_.get(), d_ccl_compact_ids_.get(), d_ccl_id_counter_.get(),
      d_ccl_bboxes_.get(), d_ccl_num_boxes_.get(),
      h_ccl_boxes_.get(), &h_num_boxes, stream, &h_num_total);

  std::vector<Box> boxes;
  if (h_num_boxes == 0) return boxes;

  // The PRE-filter per-component boxes (first half of d_ccl_bboxes_) are
  // indexed by compact id, which is what compact_ids[] stores.
  // LOAD-BEARING BOUND: every id stored in compact_ids[] is < kMaxGpuComponents
  // or -1 — ccl_buf_compact_assign_kernel (kernels.cu) writes -1 for roots past
  // the cap, so the [compact_id]-indexed rect slots can never be written out of
  // bounds even when h_num_total exceeds the cap. num_slots clamps only the COUNT.
  using turbo_ocr::kernels::GpuDetBox;
  using turbo_ocr::kernels::kMaxGpuComponents;
  int num_slots = std::min(h_num_total, (int)kMaxGpuComponents);
  if (num_slots == 0) return boxes;

  turbo_ocr::kernels::cuda_label_accepted_components(
      d_bitmap, d_ccl_compact_ids_.get(), d_ccl_bboxes_.get(), num_slots,
      box_thresh_, d_comp_labels_.get(), resize_w, resize_h, stream);

  GpuDetBox *rects = d_ccl_bboxes_.get() + kMaxGpuComponents;
  turbo_ocr::kernels::cuda_extract_oriented_rects(
      d_comp_labels_.get(), resize_w, resize_h, rects, num_slots,
      d_ccl_moments_.get(), d_ccl_orient_.get(), stream);

  CUDA_CHECK(cudaMemcpyAsync(h_rect_boxes_.get(), rects,
      num_slots * sizeof(GpuDetBox), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // pixel_count == 0: an empty slot or a rejected component.
  boxes.reserve(h_num_boxes);
  for (int i = 0; i < num_slots; i++) {
    const auto &r = h_rect_boxes_.get()[i];
    if (r.pixel_count == 0) continue;
    // Corners go around the rectangle: 0-1 is one side, 0-3 the other.
    const float w = std::hypot(r.ox[1] - r.ox[0], r.oy[1] - r.oy[0]);
    const float h = std::hypot(r.ox[3] - r.ox[0], r.oy[3] - r.oy[0]);
    if (std::min(w, h) < kMinBoxSide) continue;
    const cv::RotatedRect rect(
        cv::Point2f(0.25f * (r.ox[0] + r.ox[1] + r.ox[2] + r.ox[3]),
                    0.25f * (r.oy[0] + r.oy[1] + r.oy[2] + r.oy[3])),
        cv::Size2f(w, h),
        std::atan2(r.oy[1] - r.oy[0], r.ox[1] - r.ox[0]) * 180.0f / static_cast<float>(CV_PI));
    if (auto box = region_to_box(rect, unclip_ratio_ * unclip_scale_, kMinUnclippedSide,
                                 ratio_w, ratio_h, orig_w, orig_h))
      boxes.push_back(*box);
  }
  return boxes;
}

// CPU fallback path (original findContours)
std::vector<Box>
PaddleDet::run_cpu_contours(const float *d_pred, const uint8_t *d_bitmap,
                             int resize_h, int resize_w,
                             int orig_h, int orig_w,
                             cudaStream_t stream,
                             int content_h, int content_w) {
  if (content_h <= 0) content_h = resize_h;
  if (content_w <= 0) content_w = resize_w;
  // Download raw probability map for score filtering. Both D2H copies are queued
  // async and covered by a single stream sync — the minimal, and required, sync
  // for the host-side findContours below.
  cv::Mat pred_map(resize_h, resize_w, CV_32F);
  CUDA_CHECK(cudaMemcpyAsync(pred_map.data, d_pred,
                              resize_h * resize_w * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));

  cv::Mat bitmap(resize_h, resize_w, CV_8UC1);
  CUDA_CHECK(cudaMemcpyAsync(bitmap.data, d_bitmap, resize_w * resize_h,
                              cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // extract_boxes_from_bitmap takes the map extent from the Mats; the
  // resize params only feed the box->original ratios, so pass content dims.
  return extract_boxes_from_bitmap(pred_map, bitmap, orig_h, orig_w, content_h, content_w,
                                   box_thresh_, unclip_ratio_ * unclip_scale_, kMinBoxSide,
                                   kMinUnclippedSide, shifted_buf_, mask_buf_, contours_buf_,
                                   hierarchy_buf_);
}


} // namespace turbo_ocr::detection
