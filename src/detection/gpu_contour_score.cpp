#include "turbo_ocr/detection/gpu_contour_score.h"

#include <array>
#include <climits>
#include <cmath>
#include <cstring>

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/detection/det_postprocess.h"

namespace turbo_ocr::detection {

GpuContourScorer::GpuContourScorer(size_t max_pixels) : capacity_(max_pixels) {
  if (max_pixels == 0 || max_pixels > INT_MAX)
    throw InferenceError("Invalid contour mask capacity");
  constexpr int n = kernels::kContourScoreBatchRegions;
  d_masks_ = CudaPtr<uint8_t>(max_pixels);
  h_masks_ = CudaHostPtr<uint8_t>(max_pixels);
  d_regions_ = CudaPtr<kernels::ContourScoreRegion>(n);
  h_regions_ = CudaHostPtr<kernels::ContourScoreRegion>(n);
  d_scores_ = CudaPtr<float>(n);
  h_scores_ = CudaHostPtr<float>(n);
}

std::vector<float> GpuContourScorer::score(const float* d_pred, int width, int height,
                                           const std::vector<std::vector<cv::Point>>& contours,
                                           cudaStream_t stream) {
  if (!d_pred || width <= 0 || height <= 0 ||
      static_cast<size_t>(width) > capacity_ / static_cast<size_t>(height))
    throw InferenceError("Contour probability map exceeds capacity");
  std::vector<float> scores(contours.size(), 0.0f);
  constexpr int limit = kernels::kContourScoreBatchRegions;
  std::array<size_t, limit> indexes{};
  int count = 0;
  size_t used = 0;
  auto flush = [&] {
    if (count == 0) return;
    // Drain even on an exception: queued readers must finish before staging
    // buffers may be reused, or before the enclosing object is destroyed.
    try {
      CUDA_CHECK(
          cudaMemcpyAsync(d_masks_.get(), h_masks_.get(), used, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_regions_.get(), h_regions_.get(),
                                 count * sizeof(kernels::ContourScoreRegion),
                                 cudaMemcpyHostToDevice, stream));
      kernels::cuda_contour_scores(d_pred, width, height, d_masks_.get(), static_cast<int>(used),
                                   d_regions_.get(), count, d_scores_.get(), stream);
      CUDA_CHECK(cudaMemcpyAsync(h_scores_.get(), d_scores_.get(), count * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) {
      (void)cudaStreamSynchronize(stream);
      throw;
    }
    for (int j = 0; j < count; ++j) {
      const float value = h_scores_.get()[j];
      if (!std::isfinite(value) || value < 0 || value > 1)
        throw InferenceError("Invalid GPU contour score");
      scores[indexes[j]] = value;
    }
    count = 0;
    used = 0;
  };
  for (size_t i = 0; i < contours.size(); ++i) {
    const cv::Rect r = filled_contour_mask(width, height, contours[i], shifted_, mask_);
    if (r.empty()) continue;
    const size_t area = static_cast<size_t>(r.width) * r.height;
    if (area > capacity_) throw InferenceError("Contour mask exceeds capacity");
    if (count == limit || area > capacity_ - used) flush();
    for (int y = 0; y < r.height; ++y)
      std::memcpy(h_masks_.get() + used + static_cast<size_t>(y) * r.width, mask_.ptr<uint8_t>(y),
                  r.width);
    h_regions_.get()[count] = {r.x, r.y, r.width, r.height, static_cast<int>(used)};
    indexes[count++] = i;
    used += area;
  }
  flush();
  return scores;
}

}  // namespace turbo_ocr::detection
