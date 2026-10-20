#include <cuda_runtime.h>

#include <cmath>

#include "turbo_ocr/common/cuda/cuda_check.h"
#include "turbo_ocr/kernels/contour_score.h"

namespace turbo_ocr::kernels {
namespace {
constexpr int kThreads = 256;

__global__ void contour_scores_kernel(const float* pred, int width, int height,
                                      const uint8_t* masks, int mask_bytes,
                                      const ContourScoreRegion* regions, float* scores) {
  const int region = blockIdx.x;
  const ContourScoreRegion r = regions[region];
  const long long area = static_cast<long long>(r.width) * r.height;
  if (r.width <= 0 || r.height <= 0 || r.x < 0 || r.y < 0 || r.width > width || r.height > height ||
      r.x > width - r.width || r.y > height - r.height || r.mask_offset < 0 ||
      area > static_cast<long long>(mask_bytes) - r.mask_offset) {
    if (threadIdx.x == 0) scores[region] = nanf("");
    return;
  }
  // cv::mean accumulates in double. Include all filled-mask pixels, not just
  // thresholded foreground: a hollow icon's low-score interior counts too.
  double sum = 0;
  unsigned int count = 0;
  for (long long i = threadIdx.x; i < area; i += kThreads) {
    if (masks[static_cast<long long>(r.mask_offset) + i]) {
      const long long y = i / r.width;
      const long long x = i % r.width;
      sum += pred[(static_cast<long long>(r.y) + y) * width + r.x + x];
      ++count;
    }
  }
  __shared__ double sums[kThreads];
  __shared__ unsigned int counts[kThreads];
  sums[threadIdx.x] = sum;
  counts[threadIdx.x] = count;
  __syncthreads();
  for (int stride = kThreads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) {
      sums[threadIdx.x] += sums[threadIdx.x + stride];
      counts[threadIdx.x] += counts[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) scores[region] = counts[0] ? static_cast<float>(sums[0] / counts[0]) : 0.0f;
}
}  // namespace

void cuda_contour_scores(const float* d_pred, int width, int height, const uint8_t* d_masks,
                         int mask_bytes, const ContourScoreRegion* d_regions, int count,
                         float* d_scores, cudaStream_t stream) {
  if (count == 0) return;
  if (count < 0 || count > kContourScoreBatchRegions || !d_pred || !d_masks || !d_regions ||
      !d_scores || width <= 0 || height <= 0 || mask_bytes <= 0)
    throw InferenceError("Invalid GPU contour scoring arguments");
  contour_scores_kernel<<<count, kThreads, 0, stream>>>(d_pred, width, height, d_masks, mask_bytes,
                                                        d_regions, d_scores);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbo_ocr::kernels
