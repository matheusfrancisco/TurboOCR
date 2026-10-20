#include <catch_amalgamated.hpp>
#include <cmath>
#include <limits>

#include "turbo_ocr/common/cv_geometry.h"
#include "turbo_ocr/detection/det_postprocess.h"
#include "turbo_ocr/detection/gpu_contour_score.h"

using turbo_ocr::CudaPtr;
using turbo_ocr::detection::box_score_fast;
using turbo_ocr::detection::GpuContourScorer;
using namespace turbo_ocr::kernels;

namespace {
void require_device() {
  if (cudaSetDevice(0) != cudaSuccess) {
    (void)cudaGetLastError();
    SKIP("no CUDA device");
  }
}
struct Stream {
  cudaStream_t value{};
  Stream() { CUDA_CHECK(cudaStreamCreate(&value)); }
  ~Stream() {
    (void)cudaStreamSynchronize(value);
    (void)cudaStreamDestroy(value);
  }
};
std::vector<cv::Point> rectangle(int x, int y, int w, int h) {
  return {{x, y}, {x + w - 1, y}, {x + w - 1, y + h - 1}, {x, y + h - 1}};
}
void check_scores(const cv::Mat& pred, const std::vector<std::vector<cv::Point>>& contours) {
  Stream stream;
  CudaPtr<float> device(pred.total());
  CUDA_CHECK(cudaMemcpyAsync(device.get(), pred.data, pred.total() * sizeof(float),
                             cudaMemcpyHostToDevice, stream.value));
  GpuContourScorer scorer(pred.total());
  const auto scores = scorer.score(device.get(), pred.cols, pred.rows, contours, stream.value);
  REQUIRE(scores.size() == contours.size());
  std::vector<cv::Point> shifted;
  cv::Mat mask;
  for (size_t i = 0; i < contours.size(); ++i) {
    INFO("contour=" << i);
    CHECK(scores[i] ==
          Catch::Approx(box_score_fast(pred, contours[i], shifted, mask)).margin(1e-6));
  }
  cv::Mat after(pred.size(), CV_32F);
  CUDA_CHECK(
      cudaMemcpy(after.data, device.get(), pred.total() * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cv::norm(pred, after, cv::NORM_INF) == 0);  // scoring never changes its source
}
}  // namespace

TEST_CASE("GPU filled-contour means match CPU on hollow and solid candidates",
          "[det_gpu][contour_score]") {
  require_device();
  cv::Mat pred(64, 96, CV_32F, cv::Scalar(.05f));
  pred(cv::Rect(8, 8, 24, 24)).setTo(.95f);
  pred(cv::Rect(10, 10, 20, 20)).setTo(.05f);
  pred(cv::Rect(48, 12, 40, 10)).setTo(.95f);
  pred(cv::Rect(85, 55, 9, 5)).setTo(.95f);  // tiny, short sensitive-ID-like region
  check_scores(pred, {rectangle(8, 8, 24, 24), rectangle(48, 12, 40, 10), rectangle(85, 55, 9, 5)});
}

TEST_CASE("GPU filled-contour means cover edges concavity and empty contours",
          "[det_gpu][contour_score]") {
  require_device();
  cv::Mat pred(50, 64, CV_32F);
  cv::RNG rng(42);
  rng.fill(pred, cv::RNG::UNIFORM, 0.0, 1.0);
  check_scores(pred, {{{0, 0}, {12, 0}, {12, 8}, {0, 8}},
                      {{20, 3}, {29, 13}, {18, 24}, {9, 14}},
                      {{2, 25}, {24, 25}, {24, 30}, {8, 30}, {8, 45}, {2, 45}},
                      {{-4, -3}, {7, -3}, {7, 6}, {-4, 6}},
                      rectangle(55, 45, 9, 5),
                      {},
                      {{3, 3}, {3, 3}}});
}

TEST_CASE("GPU contour staging flushes at both byte and descriptor limits",
          "[det_gpu][contour_score]") {
  require_device();
  cv::Mat pred(64, 64, CV_32F);
  cv::RNG rng(43);
  rng.fill(pred, cv::RNG::UNIFORM, 0.0, 1.0);
  SECTION("descriptor limit with many small masks") {
    std::vector<std::vector<cv::Point>> contours;
    for (int i = 0; i < 600; ++i) contours.push_back(rectangle(i % 60, (i * 7) % 60, 2, 2));
    check_scores(pred, contours);
  }
  SECTION("overlapping masks exceed the bounded byte capacity") {
    check_scores(pred, std::vector<std::vector<cv::Point>>(5, rectangle(0, 0, 64, 64)));
  }
}

TEST_CASE("GPU contour scoring rejects invalid scores and can be reused",
          "[det_gpu][contour_score]") {
  require_device();
  constexpr int n = 64;
  CudaPtr<float> device(n);
  GpuContourScorer scorer(n);
  const std::vector<std::vector<cv::Point>> contours{rectangle(0, 0, 8, 8)};
  std::vector<float> pred(n, std::numeric_limits<float>::quiet_NaN());
  CUDA_CHECK(cudaMemcpy(device.get(), pred.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_THROWS_AS(scorer.score(device.get(), 8, 8, contours, 0), turbo_ocr::InferenceError);
  CHECK_THROWS_AS(scorer.score(device.get(), 9, 8, contours, 0), turbo_ocr::InferenceError);
  CHECK_THROWS_AS(scorer.score(nullptr, 8, 8, contours, 0), turbo_ocr::InferenceError);
  pred.assign(n, .75f);
  CUDA_CHECK(cudaMemcpy(device.get(), pred.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  const auto recovered = scorer.score(device.get(), 8, 8, contours, 0);
  REQUIRE(recovered.size() == 1);
  CHECK(recovered[0] == .75f);  // inclusive threshold boundary
  REQUIRE(cudaGetLastError() == cudaSuccess);
}

TEST_CASE("GPU contour descriptor bounds fail without out-of-bounds access",
          "[det_gpu][contour_score]") {
  require_device();
  CudaPtr<float> pred(64), scores(6);
  CudaPtr<uint8_t> masks(64);
  CudaPtr<ContourScoreRegion> regions(6);
  CUDA_CHECK(cudaMemset(pred.get(), 0, 64 * sizeof(float)));
  CUDA_CHECK(cudaMemset(masks.get(), 0, 64));
  const ContourScoreRegion bad[] = {{0, 0, 8, 8, 0}, {-1, 0, 8, 8, 0}, {0, 0, 9, 8, 0},
                                    {0, 0, 8, 8, 1}, {0, 0, 8, 8, -1}, {0, 0, 0, 8, 0}};
  CUDA_CHECK(cudaMemcpy(regions.get(), bad, sizeof(bad), cudaMemcpyHostToDevice));
  cuda_contour_scores(pred.get(), 8, 8, masks.get(), 64, regions.get(), 6, scores.get(), 0);
  float result[6]{};
  CUDA_CHECK(cudaMemcpy(result, scores.get(), sizeof(result), cudaMemcpyDeviceToHost));
  CHECK(result[0] == 0);  // empty mask matches cv::mean
  for (int i = 1; i < 6; ++i) CHECK(std::isnan(result[i]));
  REQUIRE(cudaGetLastError() == cudaSuccess);
}
