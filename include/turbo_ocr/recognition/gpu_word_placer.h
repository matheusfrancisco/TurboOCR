#pragma once

// Word boxes (?words=1) with the pixel work on the GPU, for the GPU pipeline.
// The same words recognition::locate_words places on the CPU, to the pixel:
// the GPU runs the per-pixel stages (crop, threshold, the other lines' claims,
// connected components, column profiles) for every line of a page at once,
// and the decisions about the components are the CPU code itself.

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include "turbo_ocr/common/types.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/recognition/ctc_decode.h"

namespace turbo_ocr::recognition {

class GpuWordPlacer {
public:
  GpuWordPlacer();
  ~GpuWordPlacer();
  GpuWordPlacer(const GpuWordPlacer &) = delete;
  GpuWordPlacer &operator=(const GpuWordPlacer &) = delete;

  // TURBO_WORDS_GPU (default on); 0 places word boxes on the CPU instead.
  [[nodiscard]] static bool enabled();

  using Finish = std::function<void(std::vector<OCRResultItem> &)>;

  // Embedded device callers forbid source-pixel downloads. Unresolved words
  // then keep the entire detected line, never a guessed sub-line span.
  enum class Fallback { CpuPixels, WholeLine };

  // Word boxes of the lines `todo` -- (result index, box index) pairs -- on
  // the device page `page`: the pixel work runs on `stream` and is waited for
  // here. Returns the rest, which writes each line's words into its result:
  // it owns everything it needs, so it may run later on another thread. A
  // line the GPU pass hands back (light type on a dark ground, glyphs of two
  // lines touching) is placed there by the CPU code, from `host_page` or,
  // without one, from a copy of the page taken here.
  [[nodiscard]] Finish place(const GpuImage &page, cudaStream_t stream,
                             const std::vector<Box> &boxes,
                             const std::vector<std::vector<CtcWord>> &words,
                             std::vector<std::pair<std::size_t, std::size_t>> todo,
                             const cv::Mat *host_page,
                             Fallback fallback = Fallback::CpuPixels);

  // place() off the caller's GPU worker: copies the device page (the only
  // work done here, waited for on `stream`) and returns a Finish that runs the
  // GPU passes later, on whatever thread calls it, with a placer from a small
  // shared pool on its own stream -- so the pipeline takes its next page at
  // once. `host_page` may be empty. When the copies already waiting would hold
  // too much device memory with this one, it places now instead.
  [[nodiscard]] Finish place_later(const GpuImage &page, cudaStream_t stream,
                                   const std::vector<Box> &boxes,
                                   const std::vector<std::vector<CtcWord>> &words,
                                   std::vector<std::pair<std::size_t, std::size_t>> todo,
                                   cv::Mat host_page);

  // Lines of the last place() the GPU pass handed back to the CPU code.
  [[nodiscard]] std::size_t cpu_lines() const noexcept { return cpu_lines_; }

private:
  std::size_t cpu_lines_ = 0;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace turbo_ocr::recognition
