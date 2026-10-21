#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/common/types.h"        // OCRResultItem, Box, kDropScore
#include "turbo_ocr/pipeline/pipeline_result.h"  // OcrPipelineResult, finalize_deferred
#include "turbo_ocr/recognition/word_boxes.h"

#include <functional>

namespace turbo_ocr::pipeline::detail {

// No-silent-failure guard for the base OCR/recognition stage. Detection found
// `num_boxes` text regions but recognition produced no usable text — flag the
// result degraded so the response can never be a clean empty 200 that looks
// identical to a genuinely text-free page. A page with zero detections is NOT
// degraded (correctly text-free). Mirrors the formula/table degraded contract.
inline void flag_text_degraded(OcrPipelineResult &out, std::size_t num_boxes) {
  if (num_boxes > 0 && out.results.empty()) {
    out.text_degraded = true;
    out.text_warning =
        "text stage degraded: detection found " + std::to_string(num_boxes) +
        " text region(s) but recognition produced no usable text "
        "(all crops decoded empty/blank; not a genuinely blank page)";
  }
}

// Partial recognition drops (engine output exceeded the decode buffers, see
// PaddleRec::last_dropped_crops) surface as text_degraded even when the rest
// of the page decoded fine — a thinner page must never read as a clean one.
inline void flag_dropped_crops(OcrPipelineResult &out, int dropped) {
  if (dropped <= 0) return;
  out.text_degraded = true;
  const std::string w =
      "text stage degraded: recognition dropped " + std::to_string(dropped) +
      " crop(s) (engine output exceeded decode buffers)";
  out.text_warning = out.text_warning.empty() ? w : out.text_warning + "; " + w;
}

// Word boxes for the kept lines: todo holds (result index, box index) pairs.
// Each line reads only the page and the boxes and writes its own result, so
// a page is split over a few threads. The split follows the pixels each line's
// analysis covers (length x height), not the line count: sixteen slide
// headlines are more work than a hundred lines of small print.
inline void place_words(OcrPipelineResult &out, const std::vector<Box> &boxes,
                        const std::vector<std::vector<recognition::CtcWord>> &words,
                        const cv::Mat &img,
                        const std::vector<std::pair<std::size_t, std::size_t>> &todo) {
  static constexpr double kPixelsPerThread = 1 << 20;
  static constexpr std::size_t kMaxThreads = 8;
  const std::size_t n = todo.size();
  const auto run = [&](std::size_t lo, std::size_t hi) {
    for (std::size_t k = lo; k < hi; ++k) {
      const auto [ri, bi] = todo[k];
      out.results[ri].words = recognition::locate_words(img, boxes[bi], words[bi], boxes);
    }
  };
  // Pixels up to each line: the crop is the line's length by twice its height.
  std::vector<double> upto(n + 1, 0.0);
  for (std::size_t k = 0; k < n; ++k) {
    const Box &b = boxes[todo[k].second];
    const double e01 = std::hypot(b[1][0] - b[0][0], b[1][1] - b[0][1]);
    const double e03 = std::hypot(b[3][0] - b[0][0], b[3][1] - b[0][1]);
    upto[k + 1] = upto[k] + std::max(e01, e03) * 2.0 * std::min(e01, e03) + 1.0;
  }
  const std::size_t threads = std::min<std::size_t>(
      {kMaxThreads, std::max(1u, std::thread::hardware_concurrency()), n,
       static_cast<std::size_t>(upto[n] / kPixelsPerThread) + 1});
  if (threads <= 1) {
    run(0, n);
    return;
  }
  // Contiguous chunks of about equal pixels.
  std::vector<std::size_t> cut(threads + 1, n);
  cut[0] = 0;
  for (std::size_t t = 1; t < threads; ++t)
    cut[t] = static_cast<std::size_t>(
        std::lower_bound(upto.begin(), upto.end(), upto[n] * t / threads) - upto.begin());
  std::vector<std::exception_ptr> errors(threads);
  std::vector<std::thread> pool;
  pool.reserve(threads - 1);
  for (std::size_t t = 1; t < threads; ++t)
    pool.emplace_back([&, t] {
      try {
        run(std::min(cut[t], n), std::min(cut[t + 1], n));
      } catch (...) {
        errors[t] = std::current_exception();
      }
    });
  try {
    run(0, std::min(cut[1], n));
  } catch (...) {
    errors[0] = std::current_exception();
  }
  for (auto &th : pool) th.join();
  for (auto &e : errors)
    if (e) std::rethrow_exception(e);
}

// Hands the lines whose words to place -- (result index, box index) -- to a
// placer that does the pixel work and returns the rest (the GPU pipeline's
// GpuWordPlacer).
using WordPlaceFn = std::function<std::function<void(std::vector<OCRResultItem> &)>(
    std::vector<std::pair<std::size_t, std::size_t>>)>;

// The single combine step every pipeline path ends with: pair recognition
// output with its boxes, drop empty/below-kDropScore results, then apply the
// text-degraded guard. One implementation so the filter semantics can never
// drift between the CPU, cv::Mat, GpuImage and batch paths again. With the
// recognizer's word segmentation (`words`, one entry per box) and the host
// pixels, every kept line also gets its word boxes. `defer_words` leaves the
// placement to out.place_words, which the caller runs with finalize_deferred()
// off the GPU worker; it needs pixels the result can own (an allocated Mat).
// With `placer`, the words are placed by it instead of from `img`.
inline void combine_recognition(
    OcrPipelineResult &out, const std::vector<Box> &boxes,
    std::vector<std::pair<std::string, float>> &rec_results,
    const std::vector<std::vector<recognition::CtcWord>> *words = nullptr,
    const cv::Mat *img = nullptr, bool defer_words = false,
    const WordPlaceFn *placer = nullptr, float minimum_score = turbo_ocr::kDropScore) {
  out.results.reserve(out.results.size() + boxes.size());
  const std::size_t n = std::min(boxes.size(), rec_results.size());
  std::vector<std::pair<std::size_t, std::size_t>> todo;
  for (std::size_t i = 0; i < n; ++i) {
    if (rec_results[i].second < minimum_score) continue;
    if (rec_results[i].first.empty()) continue;
    out.results.push_back({
        .text = std::move(rec_results[i].first),
        .confidence = rec_results[i].second,
        .box = boxes[i],
    });
    if (words && (img || placer) && i < words->size())
      todo.emplace_back(out.results.size() - 1, i);
  }
  if (!todo.empty()) {
    if (placer && *placer) {
      auto finish = (*placer)(std::move(todo));
      if (defer_words)
        out.place_words = [finish = std::move(finish)](OcrPipelineResult &r) {
          finish(r.results);
        };
      else
        finish(out.results);
    } else if (defer_words && img->u != nullptr)
      out.place_words = [boxes, words = *words, page = *img,
                         todo = std::move(todo)](OcrPipelineResult &r) {
        place_words(r, boxes, words, page, todo);
      };
    else
      place_words(out, boxes, *words, *img, todo);
  }
  flag_text_degraded(out, boxes.size());
}

// Backend-independent table-region adjustment (kept out of the recognizers so
// the env knobs live in one place and backends receive an already-adjusted
// box):
//   TABLE_CROP_MODE=detunion — snap to the tight AABB of the det text boxes
//     inside the layout box (so the region can only tighten).
//   TABLE_CROP_MARGIN — expand by this fraction per side (default 0.03, the
//     measured best on the 117-table set; layout boxes tend to clip border
//     rows/cols and structure-TEDS is sensitive to missing edge cells).
Box adjust_table_region(const Box &in,
                        const std::vector<OCRResultItem> &results);

} // namespace turbo_ocr::pipeline::detail
