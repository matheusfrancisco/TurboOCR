// Single-image entry points (cv::Mat and GpuImage): upload + det + cls + rec
// + layout with the shared degenerate-input fault taxonomy. The router
// dispatch lives in ocr_pipeline_dispatch.cpp, batching in
// ocr_pipeline_batch.cpp, lifecycle/model loading in ocr_pipeline_init.cpp.

#include "turbo_ocr/pipeline/ocr/ocr_pipeline.h"
#include <unordered_map>
#include "infer_one.h"
#include "ocr_pipeline_detail.h"
#include "recognizer_registry.h"
#include "turbo_ocr/classification/cls_options.h"
#include "turbo_ocr/common/cuda/cuda_check.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/common/log/timing.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/recognition/gpu_word_placer.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/formula/routing/auto_cjk_formula.h"
#include "turbo_ocr/formula/formula_recognizer.h"
#include "turbo_ocr/layout/order/reading_order.h"
#include "turbo_ocr/router/cua_router.h"
#include "turbo_ocr/backend_routing/routing_config.h"
#include "turbo_ocr/table/table_recognizer.h"
#include "turbo_ocr/table/cell_matcher.h"
#include "turbo_ocr/table/html_reconstruct.h"
#include "turbo_ocr/table/slanext/slanext_enc_split.h"
#include "turbo_ocr/table/table_types.h"
#include "turbo_ocr/engine/trt/onnx_to_trt.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <format>

#include <opencv2/imgproc.hpp>

using namespace turbo_ocr::pipeline;
using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::GpuImage;
using turbo_ocr::PipelineTimer;
// is_vertical_box / sorted_boxes are called unqualified below; ADL resolves them
// to turbo_ocr:: from their Box / vector<Box> arguments, so no using-decl needed.
using turbo_ocr::detection::PaddleDet;
using turbo_ocr::classification::PaddleCls;
using turbo_ocr::recognition::PaddleRec;
using turbo_ocr::layout::PaddleLayout;
using turbo_ocr::pipeline::OcrPipelineResult;

using turbo_ocr::pipeline::detail::adjust_table_region;
#include "turbo_ocr/pipeline/reading_order_util.h"

namespace {
// Single fault taxonomy for the upload+detection stage, shared by the cv::Mat
// and GpuImage entry points. Returns true when the fault is attributable to a
// degenerate INPUT (genuinely no text -> empty result is the correct answer);
// throws InferenceError for everything else so a real fault — crucially OOM
// under VRAM pressure — is never a silent blank page. Callers run this only
// after abort_on_sticky_cuda_fault() has ruled out a poisoned context; the
// error is cleared either way so it does not poison subsequent requests.
[[nodiscard]] bool det_fault_is_degenerate_input(const turbo_ocr::CudaError &e,
                                                 int cols, int rows,
                                                 const char *where) {
  const cudaError_t err = cudaGetLastError();
  // InvalidPitchValue / InvalidConfiguration fire for dims no GPU op can
  // launch on (1x1 pages, corrupt-decoded Mats with broken pitch). The
  // generic InvalidValue is accepted as degenerate ONLY for genuinely tiny
  // inputs — on a full-size image it means a real launch bug and must be
  // loud, not an empty 200.
  const bool tiny = cols < 8 || rows < 8;
  const bool degenerate = err == cudaErrorInvalidPitchValue ||
                          err == cudaErrorInvalidConfiguration ||
                          (err == cudaErrorInvalidValue && tiny);
  if (degenerate) {
    TOCR_LOG_WARN_RL("degenerate input, returning empty result", "cols", cols,
                     "rows", rows, "where", where, "error", e.what(),
                     "cuda", cudaGetErrorString(err));
    return true;
  }
  TOCR_LOG_ERROR_RL("detection GPU fault, surfacing as inference error",
                    "cols", cols, "rows", rows, "where", where,
                    "error", e.what(), "cuda", cudaGetErrorString(err));
  throw turbo_ocr::InferenceError(std::string("detection GPU fault: ") + e.what());
}

// Host copy of a device page (interleaved BGR, pitched) for the host-side word
// boxes (recognition::locate_words). A failed copy throws: a response must not
// come back without the words it asked for.
cv::Mat download_page(const GpuImage &img, cudaStream_t stream) {
  cv::Mat host(img.rows, img.cols, CV_8UC3);
  CUDA_CHECK(cudaMemcpy2DAsync(host.data, host.step, img.data, img.step,
                               static_cast<size_t>(img.cols) * 3, img.rows,
                               cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return host;
}
} // namespace

turbo_ocr::recognition::GpuWordPlacer &OcrPipeline::word_placer() {
  if (!word_placer_) word_placer_ = std::make_unique<turbo_ocr::recognition::GpuWordPlacer>();
  return *word_placer_;
}

std::vector<OCRResultItem> OcrPipeline::run(const cv::Mat &img,
                                            cudaStream_t stream) {
  return run_with_layout(img, stream).results;
}

OcrPipelineResult OcrPipeline::run_with_layout(const cv::Mat &img,
                                               cudaStream_t stream,
                                               bool want_layout,
                                               bool want_reading_order,
                                               const backend_routing::RequestRouting &routing,
                                               bool defer_external,
                                               bool want_tables,
                                               bool want_formulas,
                                               bool want_words,
                                               bool defer_words) {
  UseGuard _ug{in_use_, "run_with_layout"};
  const bool layout_active = use_layout_ && want_layout;
  if (img.empty()) [[unlikely]] return OcrPipelineResult{};

  PipelineTimer timer;
  timer.init(stream);
  timer.reset();

  // Upload + detection wrapped: degenerate inputs (e.g. 1×1, corrupt-
  // decoded Mats with zero-aligned pitch) trip CUDA "invalid pitch" in
  // cudaMemcpy2DAsync or in the resize kernel. Reset the stream and
  // return an empty result instead of bubbling up a 500 — there is no
  // text to detect and the request shouldn't poison subsequent ones.
  GpuImage gpu_img;
  std::vector<Box> boxes;
  try {
    gpu_img = upload_image(img, stream, timer);
    timer.gpu_start("detection_inference");
    boxes = det_->run(gpu_img, img.rows, img.cols, stream);
    timer.gpu_stop();
  } catch (const turbo_ocr::CudaError &e) {
    // Surface any async fault, then: a STICKY fault poisons the context for every future
    // request, so fail fast and let the orchestrator restart a healthy pod (std::_Exit, not
    // std::abort — skip atexit/dtors that would issue poisoned CUDA calls and hang).
    cudaStreamSynchronize(stream);
    turbo_ocr::abort_on_sticky_cuda_fault("run_with_layout/upload+det");
    if (det_fault_is_degenerate_input(e, img.cols, img.rows,
                                      "run_with_layout/upload+det"))
      return OcrPipelineResult{};
  }

  // Sort boxes top-to-bottom, left-to-right (in-place)
  timer.cpu_start("box_postprocessing");
  sorted_boxes(boxes);
  timer.cpu_stop();

  // Optional layout detection — dispatched on a dedicated layout_stream_
  // that waits only on det (via det_only_event_), so layout TRT execute
  // overlaps with cls on `stream` AND with rec on `rec_stream_`. The
  // host-side decode happens in collect() at the very end of run().
  // collect() must only run when enqueue succeeded: on an enqueue failure
  // it would sync the stale d2h_event_ of the LAST successful execute and
  // hand this page the previous page's boxes.
  bool layout_enqueued = false;
  if (layout_active) {
    CUDA_CHECK(cudaEventRecord(det_only_event_, stream));
    CUDA_CHECK(cudaStreamWaitEvent(layout_stream_, det_only_event_, 0));
    timer.gpu_start("layout_enqueue");
    layout_enqueued = layout_->enqueue(gpu_img, img.rows, img.cols,
                                       layout_stream_);
    timer.gpu_stop();
  }

  // Optional angle classification (CLS_ALL_BOXES / vertical-only gate).
  classify_angles_(gpu_img, boxes, stream, &timer);

  // Recognition — launch on dedicated rec_stream_ so the caller's stream is
  // free for the next image's upload+detection (pipeline parallelism).
  // Record det_event_ on the caller's stream after det+cls, then make
  // rec_stream_ wait on it before launching recognition.
  CUDA_CHECK(cudaEventRecord(det_event_, stream));
  CUDA_CHECK(cudaStreamWaitEvent(rec_stream_, det_event_, 0));

  timer.gpu_start("recognition_inference");
  std::vector<std::vector<recognition::CtcWord>> words;
  auto rec_results =
      rec_->run(gpu_img, boxes, rec_stream_, want_words ? &words : nullptr);
  timer.gpu_stop();

  // Record rec_event_ so the NEXT run() can wait for this recognition to
  // finish before reusing the image buffer. Note: rec_->run() syncs
  // rec_stream_ internally (for D2H + CTC decode), so by the time we get
  // here rec_stream_ is idle and this event is immediately "done". The event
  // is still useful as a correctness guard and for future async recognition.
  CUDA_CHECK(cudaEventRecord(rec_event_, rec_stream_));

  // Combine (filter by drop_score, matching Python's behavior); word boxes
  // are placed from the uploaded page on the GPU (the caller's host pixels
  // only for the lines it hands back), or from the host pixels.
  OcrPipelineResult out;
  const bool gpu_words = want_words && turbo_ocr::recognition::GpuWordPlacer::enabled();
  const detail::WordPlaceFn placer =
      [&](auto todo) -> turbo_ocr::recognition::GpuWordPlacer::Finish {
    if (defer_words)  // off this worker: the caller finishes them
      return word_placer().place_later(gpu_img, rec_stream_, boxes, words, std::move(todo), img);
    return word_placer().place(gpu_img, rec_stream_, boxes, words, std::move(todo), &img);
  };
  detail::combine_recognition(out, boxes, rec_results,
                              want_words ? &words : nullptr, &img, defer_words,
                              gpu_words ? &placer : nullptr);
  detail::flag_dropped_crops(out, rec_->last_dropped_crops());

  // Layout collect waits on d2h_event_ recorded on layout_stream_. Because
  // layout and rec run on separate streams, total wall-clock is bounded by
  // max(layout, cls+rec); on typical pages rec dominates so the wait is a
  // no-op.
  if (layout_enqueued) {
    out.layout = layout_->collect();
  }

  // CUA router + table/formula dispatch. No-op on text-only pages (see
  // dispatch_router_'s short-circuits — plan 04 §7).
  dispatch_router_(out, gpu_img, boxes, timer, routing, defer_external,
                   want_tables, want_formulas);

  // Reading-order over layout regions, with synthetic XY-cut entries
  // for orphan results so unmatched detections (page numbers, headers
  // the layout model missed) land in their natural position instead of
  // trailing the entire document. Helper is shared with cpu_ocr_pipeline.
  maybe_assign_reading_order(want_reading_order, out.results, out.layout,
                             out.reading_order);

  timer.print_total();

  return out;
}

OcrPipelineResult OcrPipeline::run_layout_only(const cv::Mat &img,
                                                cudaStream_t stream) {
  UseGuard _ug{in_use_, "run_layout_only"};
  OcrPipelineResult out;
  // Fast no-op if layout isn't loaded: skip the upload entirely. Callers
  // in /ocr/pdf geometric/auto modes never need OCR text here — they fill
  // results from the PDFium text layer — so returning empty is correct.
  if (!use_layout_ || !layout_) return out;

  PipelineTimer timer;
  timer.init(stream);
  timer.reset();

  auto gpu_img = upload_image(img, stream, timer);

  // Layout runs on layout_stream_ for the same reason run_with_layout
  // does: overlap with whatever else the caller has in flight. We still
  // record det_only_event_ on `stream` so layout_stream_ waits for the
  // upload before reading the image buffer.
  CUDA_CHECK(cudaEventRecord(det_only_event_, stream));
  CUDA_CHECK(cudaStreamWaitEvent(layout_stream_, det_only_event_, 0));

  timer.gpu_start("layout_only");
  if (layout_->enqueue(gpu_img, img.rows, img.cols, layout_stream_))
    out.layout = layout_->collect();
  timer.gpu_stop();

  // Mirror the event bookkeeping of run_with_layout so the next
  // run_with_layout()/run_layout_only() call can sync correctly on its
  // turn. rec_event_ is recorded on rec_stream_ — we didn't touch
  // rec_stream_ at all, so record an already-completed event by pushing
  // a no-op into rec_stream_ after layout_stream_ is known to be done.
  CUDA_CHECK(cudaEventRecord(rec_event_, rec_stream_));

  timer.print_total();
  return out;
}

OcrPipelineResult OcrPipeline::run_layout_only(GpuImage gpu_img,
                                                cudaStream_t stream) {
  UseGuard _ug{in_use_, "run_layout_only(GpuImage)"};
  OcrPipelineResult out;
  if (!use_layout_ || !layout_) return out;

  PipelineTimer timer;
  timer.init(stream);
  timer.reset();

  // Caller-owned device image (see run_with_layout(GpuImage)): make sure no
  // earlier consumer is still reading the buffer before layout does.
  wait_prior_readers_();
  CUDA_CHECK(cudaEventRecord(det_only_event_, stream));
  CUDA_CHECK(cudaStreamWaitEvent(layout_stream_, det_only_event_, 0));

  timer.gpu_start("layout_only");
  if (layout_->enqueue(gpu_img, gpu_img.rows, gpu_img.cols, layout_stream_))
    out.layout = layout_->collect();
  timer.gpu_stop();

  // Same event bookkeeping as the host overload.
  CUDA_CHECK(cudaEventRecord(rec_event_, rec_stream_));

  timer.print_total();
  return out;
}

OcrPipelineResult OcrPipeline::run_layout_and_structure(
    const cv::Mat &img, cudaStream_t stream,
    std::vector<OCRResultItem> text_results, bool want_tables,
    bool want_formulas, const backend_routing::RequestRouting &routing) {
  UseGuard _ug{in_use_, "run_layout_and_structure"};
  OcrPipelineResult out;
  out.results = std::move(text_results);
  // No layout model -> nothing to route on; return the text unchanged. The
  // router also no-ops without table/formula backends (dispatch_router_ bails).
  if (!use_layout_ || !layout_) return out;

  PipelineTimer timer;
  timer.init(stream);
  timer.reset();

  auto gpu_img = upload_image(img, stream, timer);

  // Record on `stream` so both layout_stream_ and the router's table/formula
  // streams (which wait on det_only_event_) see the upload complete before
  // reading gpu_img — mirrors run_layout_only / run_with_layout.
  CUDA_CHECK(cudaEventRecord(det_only_event_, stream));
  CUDA_CHECK(cudaStreamWaitEvent(layout_stream_, det_only_event_, 0));

  timer.gpu_start("layout_only");
  if (layout_->enqueue(gpu_img, img.rows, img.cols, layout_stream_))
    out.layout = layout_->collect();
  timer.gpu_stop();

  // The router classifies regions and fills table cells using the text boxes as
  // the "detections" — here the text-layer boxes. Same call the OCR path makes.
  std::vector<Box> boxes;
  boxes.reserve(out.results.size());
  for (const auto &r : out.results) boxes.push_back(r.box);

  dispatch_router_(out, gpu_img, boxes, timer, routing, /*defer_external=*/false,
                   want_tables, want_formulas);

  // Event bookkeeping parity: we never touched rec_stream_, so record an
  // already-complete rec_event_ for the next run()'s wait.
  CUDA_CHECK(cudaEventRecord(rec_event_, rec_stream_));
  timer.print_total();
  return out;
}

std::vector<OCRResultItem> OcrPipeline::run(const GpuImage &gpu_img,
                                            cudaStream_t stream) {
  return run_with_layout(gpu_img, stream).results;
}

OcrPipelineResult OcrPipeline::run_with_layout(GpuImage gpu_img,
                                               cudaStream_t stream,
                                               bool want_layout,
                                               bool want_reading_order,
                                               const backend_routing::RequestRouting &routing,
                                               bool defer_external,
                                               bool want_tables,
                                               bool want_formulas,
                                               bool want_words,
                                               bool defer_words) {
  UseGuard _ug{in_use_, "run_with_layout(GpuImage)"};
  const bool layout_active = use_layout_ && want_layout;
  PipelineTimer timer;
  timer.init(stream);
  timer.reset();

  // No image_upload stage — the image is already on the GPU.
  // Wait for any previous consumers that might still be reading their source
  // image. For caller-owned GpuImage this is a correctness guard only.
  wait_prior_readers_();

  // Detection — same fault taxonomy as the cv::Mat entry point: degenerate
  // input -> empty result, everything else loud.
  std::vector<Box> boxes;
  try {
    timer.gpu_start("detection_inference");
    boxes = det_->run(gpu_img, gpu_img.rows, gpu_img.cols, stream);
    timer.gpu_stop();
  } catch (const turbo_ocr::CudaError &e) {
    cudaStreamSynchronize(stream);
    turbo_ocr::abort_on_sticky_cuda_fault("run_with_layout(GpuImage)/det");
    if (det_fault_is_degenerate_input(e, gpu_img.cols, gpu_img.rows,
                                      "run_with_layout(GpuImage)/det"))
      return OcrPipelineResult{};
  }

  // Sort boxes top-to-bottom, left-to-right (in-place)
  timer.cpu_start("box_postprocessing");
  sorted_boxes(boxes);
  timer.cpu_stop();

  // Optional layout detection (see run(cv::Mat, stream) for rationale,
  // including why collect() is gated on the enqueue result).
  bool layout_enqueued = false;
  if (layout_active) {
    CUDA_CHECK(cudaEventRecord(det_only_event_, stream));
    CUDA_CHECK(cudaStreamWaitEvent(layout_stream_, det_only_event_, 0));
    timer.gpu_start("layout_enqueue");
    layout_enqueued = layout_->enqueue(gpu_img, gpu_img.rows, gpu_img.cols,
                                       layout_stream_);
    timer.gpu_stop();
  }

  // Optional angle classification (CLS_ALL_BOXES / vertical-only gate) —
  // was vertical-only here while the cv::Mat path honored CLS_ALL_BOXES;
  // the shared helper ends that drift.
  classify_angles_(gpu_img, boxes, stream, &timer);

  // Recognition — use det_event_ for det→rec stream handoff.
  CUDA_CHECK(cudaEventRecord(det_event_, stream));
  CUDA_CHECK(cudaStreamWaitEvent(rec_stream_, det_event_, 0));

  timer.gpu_start("recognition_inference");
  std::vector<std::vector<recognition::CtcWord>> words;
  auto rec_results =
      rec_->run(gpu_img, boxes, rec_stream_, want_words ? &words : nullptr);
  timer.gpu_stop();

  // Record rec_event_ for the next run() to wait on.
  CUDA_CHECK(cudaEventRecord(rec_event_, rec_stream_));

  // Combine (filter by drop_score). Word boxes are placed on the host, so a
  // request for them costs one copy of the page.
  OcrPipelineResult out;
  const bool gpu_words = want_words && turbo_ocr::recognition::GpuWordPlacer::enabled();
  const detail::WordPlaceFn placer =
      [&](auto todo) -> turbo_ocr::recognition::GpuWordPlacer::Finish {
    if (defer_words)  // off this worker: the caller finishes them
      return word_placer().place_later(gpu_img, rec_stream_, boxes, words, std::move(todo),
                                       cv::Mat());
    return word_placer().place(gpu_img, rec_stream_, boxes, words, std::move(todo), nullptr);
  };
  cv::Mat host_img;
  if (want_words && !gpu_words) host_img = download_page(gpu_img, rec_stream_);
  detail::combine_recognition(out, boxes, rec_results,
                              want_words ? &words : nullptr, &host_img, defer_words,
                              gpu_words ? &placer : nullptr);
  detail::flag_dropped_crops(out, rec_->last_dropped_crops());

  // Layout collect — see run(cv::Mat, stream) above.
  if (layout_enqueued) {
    out.layout = layout_->collect();
  }

  // CUA router + table/formula dispatch (text-only path bails inside).
  dispatch_router_(out, gpu_img, boxes, timer, routing, defer_external,
                   want_tables, want_formulas);

  // Reading-order — see run(...) above for the contract; helper handles
  // orphan results (missing layout match) via synthetic XY-cut entries.
  maybe_assign_reading_order(want_reading_order, out.results, out.layout,
                             out.reading_order);

  timer.print_total();

  return out;
}

OcrPipelineResult OcrPipeline::run_device(GpuImage gpu_img, cudaStream_t stream,
                                          std::vector<Box> *unreadable, const DeviceOcrOptions *options,
                                          DeviceOcrDiagnostics *diagnostics) {
  if (unreadable) unreadable->clear();
  if (diagnostics) *diagnostics = {};
  UseGuard guard{in_use_, "run_device"};
  if (options && (!unreadable || !std::isfinite(options->minimum_confidence) ||
                  options->minimum_confidence<0 || options->minimum_confidence>1 || options->retry_limit>16))
    throw turbo_ocr::InferenceError("Invalid device OCR retry options");
  // Unlike the serving pipeline, keep every image reader on the caller's
  // stream. Its preceding writes are ordered before detection automatically.
  // On failure there may be kernels queued without a completion event yet.
  struct Drain {
    cudaStream_t stream;
    bool finished = false;
    ~Drain() {
      if (!finished) (void)cudaStreamSynchronize(stream);
    }
    void finish() {
      const auto status = cudaStreamSynchronize(stream);
      finished = true;
      CUDA_CHECK(status);
    }
  } drain{stream};
  wait_prior_readers_();
  auto boxes = det_->run(gpu_img, gpu_img.rows, gpu_img.cols, stream);
  sorted_boxes(boxes);
  classify_angles_(gpu_img, boxes, stream, nullptr);
  std::vector<std::vector<recognition::CtcWord>> words;
  auto recognized = rec_->run(gpu_img, boxes, stream, &words);
  if (recognized.size() != boxes.size() || words.size() != boxes.size())
    throw turbo_ocr::InferenceError("Device OCR result count mismatch");
  if (!boxes.empty() && rec_->last_dropped_crops() != 0)
    throw turbo_ocr::InferenceError("Device OCR crop dropped");
  auto validate_words=[](const std::vector<recognition::CtcWord> &ws) {
    for (const auto &w:ws)
      if (w.text.empty() || !std::isfinite(w.score) || w.score<0 || w.score>1 ||
          !std::isfinite(w.first) || !std::isfinite(w.last) || w.last<w.first)
        throw turbo_ocr::InferenceError("Device OCR word metadata invalid");
  };
  std::vector<DeviceOcrInfo> info(boxes.size());
  for (std::size_t i=0;i<recognized.size();++i) {
    const float confidence=recognized[i].second;
    if (!std::isfinite(confidence) || confidence<0 || confidence>1)
      throw turbo_ocr::InferenceError("Device OCR crop confidence invalid");
    validate_words(words[i]);
    info[i].source_confidence=confidence;
  }
  // Recognition reuses the immutable device image and caller stream. The
  // budget bounds extra work even for screens full of false text detections.
  if (unreadable && options && options->retry_limit) {
    std::vector<Box> first, second;
    std::vector<std::size_t> indexes;
    for (std::size_t i=0;i<boxes.size() && indexes.size()<options->retry_limit;++i) {
      bool uncertain=recognized[i].first.empty() || recognized[i].second<options->minimum_confidence;
      for (const auto &w:words[i]) uncertain|=w.score<options->minimum_confidence;
      if (!uncertain) continue;
      Box a{},b{};
      if (!device_retry_boxes(boxes[i],gpu_img.cols,gpu_img.rows,a,b)) continue;
      indexes.push_back(i);first.push_back(a);second.push_back(b);
    }
    if (!indexes.empty()) {
      std::vector<std::vector<recognition::CtcWord>> aw,bw;
      auto ar=rec_->run(gpu_img,first,stream,&aw);
      if (rec_->last_dropped_crops()!=0 || ar.size()!=indexes.size() || aw.size()!=indexes.size())
        throw turbo_ocr::InferenceError("Device OCR crop dropped");
      auto br=rec_->run(gpu_img,second,stream,&bw);
      if (rec_->last_dropped_crops()!=0 || br.size()!=indexes.size() || bw.size()!=indexes.size())
        throw turbo_ocr::InferenceError("Device OCR crop dropped");
      for (std::size_t j=0;j<indexes.size();++j) {
        for (float c:{ar[j].second,br[j].second})
          if (!std::isfinite(c) || c<0 || c>1) throw turbo_ocr::InferenceError("Device OCR crop confidence invalid");
        validate_words(aw[j]);validate_words(bw[j]);
        const auto i=indexes[j];info[i].retries=2;
        if (!device_retry_agrees(recognized[i].first,ar[j],br[j],aw[j],bw[j],options->minimum_confidence)) continue;
        boxes[i]=first[j];recognized[i]=std::move(ar[j]);words[i]=std::move(aw[j]);
      }
    }
  }
  const float minimum_text_score = options && options->preserve_low_confidence_text ? 0.0f : turbo_ocr::kDropScore;
  // A partially unreadable page is not a clean result for a redaction caller.
  for (std::size_t i = 0; i < recognized.size(); ++i) {
    const auto &[text, confidence] = recognized[i];
    if (!std::isfinite(confidence) || confidence < 0.0f || confidence > 1.0f)
      throw turbo_ocr::InferenceError("Device OCR crop confidence invalid");
    if (text.empty() || confidence < minimum_text_score) {
      if (!unreadable)
        throw turbo_ocr::InferenceError(text.empty() ? "Device OCR crop recognized empty"
                                                   : "Device OCR crop below threshold");
      unreadable->push_back(boxes[i]);
      info[i].reason=text.empty()?1:2;
      if (diagnostics) diagnostics->unreadable.push_back(info[i]);
      continue;
    }
    for (const auto &word : words[i])
      if (word.text.empty() || !std::isfinite(word.score) || word.score < 0.0f ||
          word.score > 1.0f || !std::isfinite(word.first) ||
          !std::isfinite(word.last) || word.last < word.first)
        throw turbo_ocr::InferenceError("Device OCR word metadata invalid");
  }
  // Compact all three vectors together; every omitted recognition has a
  // mandatory whole-line box above, never a successful no-text answer.
  if (unreadable && !unreadable->empty()) {
    std::size_t kept = 0;
    for (std::size_t i = 0; i < boxes.size(); ++i) {
      if (recognized[i].first.empty() || recognized[i].second < minimum_text_score)
        continue;
      if (kept != i) {
        boxes[kept] = boxes[i];
        recognized[kept] = std::move(recognized[i]);
        words[kept] = std::move(words[i]);
        info[kept] = info[i];
      }
      ++kept;
    }
    boxes.resize(kept); recognized.resize(kept); words.resize(kept); info.resize(kept);
  }
  OcrPipelineResult out;
  const detail::WordPlaceFn placer =
      [&](auto todo) -> turbo_ocr::recognition::GpuWordPlacer::Finish {
    return word_placer().place(gpu_img, stream, boxes, words, std::move(todo),
                               nullptr, recognition::GpuWordPlacer::Fallback::WholeLine);
  };
  detail::combine_recognition(out, boxes, recognized, &words, nullptr, false, &placer, minimum_text_score);
  if (out.text_degraded || out.results.size() != boxes.size())
    throw turbo_ocr::InferenceError("Device OCR result assembly degraded");
  if (diagnostics) diagnostics->readable = std::move(info);
  CUDA_CHECK(cudaEventRecord(rec_event_, stream));
  drain.finish();
  return out;
}
