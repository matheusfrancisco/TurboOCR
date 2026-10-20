#include "turbo_ocr/device_api.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>

#include "turbo_ocr/pipeline/ocr/ocr_pipeline.h"
#include "turbo_ocr/common/errors.h"

struct tocr_device_engine {
  int device;
  std::atomic_flag in_use = ATOMIC_FLAG_INIT;
  turbo_ocr::pipeline::OcrPipeline pipeline;

  explicit tocr_device_engine(int ordinal) : device(ordinal) {}
};

namespace {

int fail(char *error, size_t capacity, const char *message) noexcept {
  if (error && capacity) {
    const size_t n = std::min(capacity - 1, std::strlen(message));
    std::memcpy(error, message, n);
    error[n] = '\0';
  }
  return 1;
}

void clear_error(char *error, size_t capacity) noexcept {
  if (error && capacity) error[0] = '\0';
}

bool readable_file(const char *path) {
  if (!path || !*path) return false;
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  return input && input.tellg() > 0;
}

struct Rect {
  int x, y, width, height;
};

bool bounded_rect(const turbo_ocr::Box &box, int width, int height, Rect &out) noexcept {
  const auto bounds = turbo_ocr::aabb(box);
  if (bounds[2] < 0 || bounds[3] < 0 || bounds[0] >= width || bounds[1] >= height ||
      bounds[0] >= bounds[2] || bounds[1] >= bounds[3])
    return false;
  // Corners originate in integer-rounded perspective transforms. Expand one
  // pixel before clipping so the ABI rectangle cannot shrink the source quad.
  const auto x0 = std::clamp<std::int64_t>(std::int64_t{bounds[0]} - 1, 0, width);
  const auto y0 = std::clamp<std::int64_t>(std::int64_t{bounds[1]} - 1, 0, height);
  const auto x1 = std::clamp<std::int64_t>(std::int64_t{bounds[2]} + 1, 0, width);
  const auto y1 = std::clamp<std::int64_t>(std::int64_t{bounds[3]} + 1, 0, height);
  out = {static_cast<int>(x0), static_cast<int>(y0), static_cast<int>(x1 - x0),
         static_cast<int>(y1 - y0)};
  return out.width > 0 && out.height > 0;
}

bool valid_stream(cudaStream_t stream) {
  unsigned flags = 0;
  if (cudaStreamGetFlags(stream, &flags) != cudaSuccess) return false;
  cudaStreamCaptureStatus capture;
  if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
      capture != cudaStreamCaptureStatusNone)
    return false;
  if (!stream || stream == cudaStreamLegacy || stream == cudaStreamPerThread)
    return true;
  CUcontext current = nullptr, owner = nullptr;
  return cuCtxGetCurrent(&current) == CUDA_SUCCESS &&
         cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &owner) == CUDA_SUCCESS &&
         owner == current;
}

bool valid_image(const void *bgr, int device, int width, int height, size_t pitch) {
  // Kernels multiply row indices by an int pitch; enforce the complete span,
  // not just the pitch, to prevent signed overflow before any GPU launch.
  if (!bgr || width < 8 || height < 8 || width > INT_MAX / 3 ||
      pitch < static_cast<size_t>(width) * 3 || pitch > INT_MAX)
    return false;
  const size_t row = static_cast<size_t>(width) * 3;
  if (static_cast<size_t>(height - 1) > (INT_MAX - row) / pitch) return false;
  const size_t span = static_cast<size_t>(height - 1) * pitch + row;
  cudaPointerAttributes attrs{};
  if (cudaPointerGetAttributes(&attrs, bgr) != cudaSuccess ||
      attrs.type != cudaMemoryTypeDevice || attrs.device != device)
    return false;
  CUdeviceptr base = 0;
  size_t allocation = 0;
  const auto address = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(bgr));
  if (cuMemGetAddressRange(&base, &allocation, address) != CUDA_SUCCESS || address < base)
    return false;
  const auto offset = address - base;
  return offset <= allocation && span <= allocation - offset;
}

} // namespace

extern "C" tocr_device_engine *tocr_device_create(const tocr_device_config *config,
                                                  char *error, size_t error_capacity) {
  clear_error(error, error_capacity);
  try {
    if (!config || config->device < 0) {
      fail(error, error_capacity, "Invalid device OCR configuration");
      return nullptr;
    }
    if (!readable_file(config->det_model) || !readable_file(config->rec_model) ||
        !readable_file(config->rec_dict) ||
        (config->cls_model && *config->cls_model && !readable_file(config->cls_model))) {
      fail(error, error_capacity, "Device OCR model or dictionary file is missing or unreadable");
      return nullptr;
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || config->device >= count ||
        cudaSetDevice(config->device) != cudaSuccess) {
      fail(error, error_capacity, "Device OCR CUDA device is unavailable");
      return nullptr;
    }
    auto engine = std::make_unique<tocr_device_engine>(config->device);
    if (!engine->pipeline.init(config->det_model, config->rec_model, config->rec_dict,
                               config->cls_model ? config->cls_model : "")) {
      fail(error, error_capacity, "Device OCR model initialization failed");
      return nullptr;
    }
    return engine.release();
  } catch (...) {
    // Exception messages may contain model/user data. Never expose them through
    // this boundary; the startup failure remains explicit and non-successful.
    fail(error, error_capacity, "Device OCR engine creation failed");
    return nullptr;
  }
}

extern "C" void tocr_device_destroy(tocr_device_engine *engine) {
  if (!engine) return;
  try {
    (void)cudaSetDevice(engine->device);
    delete engine;
  } catch (...) {
    // Destruction has no error channel; callers must quiesce all other calls.
  }
}

static int recognize_impl(tocr_device_engine *engine, const void *bgr,
                                     int width, int height, size_t pitch,
                                     void *cuda_stream, tocr_device_emit_fn emit,
                                     void *user, char *error, size_t error_capacity,
                                     tocr_device_cover_fn cover, tocr_device_emit_ex_fn emit_ex = nullptr,
                                     const turbo_ocr::pipeline::DeviceOcrOptions *options = nullptr) {
  clear_error(error, error_capacity);
  try {
    if (!engine || (!emit && !emit_ex))
      return fail(error, error_capacity, "Invalid device OCR engine or callback");
    if (engine->in_use.test_and_set(std::memory_order_acquire))
      return fail(error, error_capacity, "Device OCR engine is already in use");
    struct Release {
      std::atomic_flag &flag;
      ~Release() { flag.clear(std::memory_order_release); }
    } release{engine->in_use};
    if (cudaSetDevice(engine->device) != cudaSuccess)
      return fail(error, error_capacity, "Device OCR CUDA device selection failed");
    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    if (!valid_stream(stream))
      return fail(error, error_capacity, "Invalid device OCR CUDA stream");
    if (!valid_image(bgr, engine->device, width, height, pitch))
      return fail(error, error_capacity, "Invalid device OCR BGR allocation, dimensions, or pitch");
    std::vector<turbo_ocr::Box> unreadable;
    turbo_ocr::pipeline::DeviceOcrDiagnostics metadata;
    auto result = engine->pipeline.run_device(
        {const_cast<void *>(bgr), pitch, height, width}, stream,
        (cover || emit_ex) ? &unreadable : nullptr, options,
        emit_ex ? &metadata : nullptr);
    if (result.text_degraded || result.results.size() > INT_MAX ||
        unreadable.size() > INT_MAX - result.results.size())
      return fail(error, error_capacity, "Device OCR recognition degraded");
    if (emit_ex && (metadata.readable.size()!=result.results.size() || metadata.unreadable.size()!=unreadable.size()))
      return fail(error,error_capacity,"Device OCR result assembly degraded");
    for (size_t line = 0; line < result.results.size(); ++line) {
      const auto &item = result.results[line];
      if (item.words.size() > INT_MAX)
        return fail(error, error_capacity, "Device OCR word count exceeds ABI limits");
      const auto deliver = [&](const std::string &text, float confidence,
                               const turbo_ocr::Box &box, int word) {
        Rect rect{};
        if (text.empty() || !std::isfinite(confidence) || confidence < 0.0f ||
            confidence > 1.0f || !bounded_rect(box, width, height, rect))
          return fail(error, error_capacity, "Device OCR output metadata invalid");
        if (emit_ex) {
          const auto &m=metadata.readable[line];
          if (emit_ex(user,text.data(),text.size(),confidence,rect.x,rect.y,rect.width,rect.height,
                      static_cast<int>(line),word,0,m.retries,box==item.box?1:0,m.source_confidence)!=0)
            return fail(error,error_capacity,"Device OCR callback aborted");
          return 0;
        }
        if (emit(user, text.data(), text.size(), confidence, rect.x, rect.y,
                  rect.width, rect.height, static_cast<int>(line), word) != 0)
          return fail(error, error_capacity, "Device OCR callback aborted");
        return 0;
      };
      if (item.words.empty()) {
        if (deliver(item.text, item.confidence, item.box, 0) != 0) return 1;
      } else {
        for (size_t word = 0; word < item.words.size(); ++word) {
          const auto &value = item.words[word];
          if (deliver(value.text, value.confidence, value.box, static_cast<int>(word)) != 0)
            return 1;
        }
      }
    }
    for (size_t i = 0; i < unreadable.size(); ++i) {
      Rect rect{};
      if (!bounded_rect(unreadable[i], width, height, rect))
        return fail(error, error_capacity, "Device OCR unreadable geometry invalid");
      if (emit_ex) {
        const auto &m=metadata.unreadable[i];
        if (emit_ex(user,nullptr,0,0,rect.x,rect.y,rect.width,rect.height,
                    static_cast<int>(result.results.size()+i),0,m.reason,m.retries,1,m.source_confidence)!=0)
          return fail(error,error_capacity,"Device OCR coverage callback aborted");
        continue;
      }
      if (cover(user, rect.x, rect.y, rect.width, rect.height,
                static_cast<int>(result.results.size() + i)) != 0)
        return fail(error, error_capacity, "Device OCR coverage callback aborted");
    }
    return 0;
  } catch (const turbo_ocr::InferenceError &e) {
    // Only these fixed diagnostics may cross the ABI. Never forward arbitrary
    // exception messages: they may contain model paths or recognized text.
    for (const char *reason : {"Device OCR result count mismatch",
                               "Device OCR crop dropped",
                               "Device OCR crop recognized empty",
                               "Device OCR crop below threshold",
                               "Device OCR crop confidence invalid",
                               "Device OCR word metadata invalid",
                               "Device OCR result assembly degraded"}) {
      if (std::strcmp(e.what(), reason) == 0)
        return fail(error, error_capacity, reason);
    }
    return fail(error, error_capacity, "Device OCR inference failed");
  } catch (const turbo_ocr::CudaError &) {
    return fail(error, error_capacity, "Device OCR CUDA operation failed");
  } catch (...) {
    return fail(error, error_capacity, "Device OCR recognition failed");
  }
}

extern "C" int tocr_device_recognize(tocr_device_engine *engine, const void *bgr,
    int width, int height, size_t pitch, void *stream, tocr_device_emit_fn emit,
    void *user, char *error, size_t capacity) {
  return recognize_impl(engine, bgr, width, height, pitch, stream, emit, user,
                        error, capacity, nullptr);
}
extern "C" int tocr_device_recognize_redaction(tocr_device_engine *engine,
    const void *bgr, int width, int height, size_t pitch, void *stream,
    tocr_device_emit_fn emit, tocr_device_cover_fn cover, void *user,
    char *error, size_t capacity) {
  if (!cover) return fail(error, capacity, "Missing device OCR coverage callback");
  return recognize_impl(engine, bgr, width, height, pitch, stream, emit, user,
                        error, capacity, cover);
}

extern "C" int tocr_device_recognize_redaction_v2(tocr_device_engine *engine,
    const void *bgr,int width,int height,size_t pitch,void *stream,
    float minimum,unsigned limit,tocr_device_emit_ex_fn emit,void *user,
    char *error,size_t capacity) {
  if (!emit || !std::isfinite(minimum) || minimum<0 || minimum>1 || limit>16)
    return fail(error,capacity,"Invalid device OCR retry options");
  const turbo_ocr::pipeline::DeviceOcrOptions options{minimum,limit};
  return recognize_impl(engine,bgr,width,height,pitch,stream,nullptr,user,error,capacity,nullptr,emit,&options);
}
