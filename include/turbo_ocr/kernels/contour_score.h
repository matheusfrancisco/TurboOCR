#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace turbo_ocr::kernels {

// Packed, row-major filled masks. The probabilities stay on device; only mask
// geometry goes H2D and one mean per region returns D2H.
struct ContourScoreRegion {
  int x, y, width, height, mask_offset;
};
inline constexpr int kContourScoreBatchRegions = 256;

// All buffers are caller-owned and preallocated. Enqueues work on stream,
// without synchronization. Invalid device descriptors yield NaN (the caller
// must reject it); an empty mask has the same zero mean as cv::mean.
void cuda_contour_scores(const float* d_pred, int width, int height, const uint8_t* d_masks,
                         int mask_bytes, const ContourScoreRegion* d_regions, int count,
                         float* d_scores, cudaStream_t stream);

}  // namespace turbo_ocr::kernels
