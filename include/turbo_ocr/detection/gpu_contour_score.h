#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "turbo_ocr/common/cuda/cuda_ptr.h"
#include "turbo_ocr/kernels/contour_score.h"

namespace turbo_ocr::detection {

// GPU means over exactly the filled polygons used by box_score_fast. A bounded
// staging area batches many small contours; large/overlapping ROIs flush it
// rather than allocating a mask per candidate. Not safe for concurrent calls.
class GpuContourScorer {
 public:
  explicit GpuContourScorer(size_t max_pixels);
  [[nodiscard]] std::vector<float> score(const float* d_pred, int width, int height,
                                         const std::vector<std::vector<cv::Point>>& contours,
                                         cudaStream_t stream);

 private:
  size_t capacity_;
  CudaPtr<uint8_t> d_masks_;
  CudaHostPtr<uint8_t> h_masks_;
  CudaPtr<kernels::ContourScoreRegion> d_regions_;
  CudaHostPtr<kernels::ContourScoreRegion> h_regions_;
  CudaPtr<float> d_scores_;
  CudaHostPtr<float> h_scores_;
  std::vector<cv::Point> shifted_;
  cv::Mat mask_;
};

}  // namespace turbo_ocr::detection
