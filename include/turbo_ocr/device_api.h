#ifndef TURBO_OCR_DEVICE_API_H
#define TURBO_OCR_DEVICE_API_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tocr_device_engine tocr_device_engine;

typedef struct tocr_device_config {
  int device;
  const char *det_model;
  const char *rec_model;
  const char *rec_dict;
  const char *cls_model; /* NULL or empty disables angle classification. */
} tocr_device_config;

/* Text is UTF-8, borrowed only during the callback, and need not be copied when
 * text_len == 0. Each callback describes one word; absent word segmentation,
 * the entire recognized line is word 0. line and word are zero-based. Rectangles
 * are positive, image-bounded, axis-aligned pixel regions [x,x+width) x
 * [y,y+height). Unresolved word geometry conservatively covers the whole line.
 * Return nonzero to abort recognition. Do not destroy the engine in a callback.
 */
typedef int (*tocr_device_emit_fn)(void *user, const char *text, size_t text_len,
                                   float confidence, int x, int y, int width,
                                   int height, int line, int word);

/* Loads existing TensorRT engine files and a matching recognition dictionary.
 * Never downloads models, installs dependencies, or starts a server. Strings
 * are borrowed only during create. NULL means startup failed; errors contain
 * no recognized text. error may be NULL; nonzero capacity is NUL-terminated.
 */
tocr_device_engine *tocr_device_create(const tocr_device_config *config,
                                        char *error, size_t error_capacity);

/* NULL is harmless. The caller must stop all calls before destroying a handle.
 * The engine's CUDA device is selected even if the calling OS thread changed.
 */
void tocr_device_destroy(tocr_device_engine *engine);

/* Synchronous, in-process OCR of caller-owned interleaved BGR8 device memory.
 * cuda_stream is a CUDA stream handle (NULL means the default stream). Writes
 * already queued on this stream happen before inference; writes on other
 * streams must be ordered by the caller. Input must remain valid until return.
 * All source-image readers are drained even on failure. Source pixels never
 * download to host; only model outputs and placement metadata may do so.
 * Ordinary CUDA device allocations (including pitched allocations and ROIs)
 * are supported; managed/host memory is rejected. The addressed span must fit
 * INT_MAX because the inference kernels use signed int byte offsets.
 * Concurrent/reentrant use of one engine returns an error, not a race. Each
 * call selects config.device. Calls on different engines may run concurrently.
 * Returns 0 on success, nonzero on validation, inference/degradation, or callback
 * failure. Discard ALL callbacks from a failed call. No C++ exceptions escape.
 */
int tocr_device_recognize(tocr_device_engine *engine, const void *bgr,
                           int width, int height, size_t pitch,
                           void *cuda_stream, tocr_device_emit_fn emit,
                           void *user, char *error, size_t error_capacity);

/* Redaction-specific recognition. Every unreadable detected crop is delivered
 * to cover with a bounded whole-line rectangle and MUST be painted opaquely
 * before presentation. Both callbacks are mandatory. Discard ALL results on
 * failure. Ordinary recognize retains its strict behavior. */
typedef int (*tocr_device_cover_fn)(void *user, int x, int y, int width,
                                    int height, int line);
int tocr_device_recognize_redaction(tocr_device_engine *engine, const void *bgr,
    int width, int height, size_t pitch, void *cuda_stream,
    tocr_device_emit_fn emit, tocr_device_cover_fn cover, void *user,
    char *error, size_t error_capacity);

/* Metadata-bearing redaction API. reason: 0 readable, 1 empty crop, 2 native
 * low-confidence crop. retry_count is 0 or 2; whole_line is 0/1. Original
 * confidence is content-free, finite 0..1. Unreadable text/score remain empty/0.
 * Retry limit is 0..16. Two expanded device crops must agree at minimum_score;
 * conflicting results retain original protection. Old APIs do not retry. */
typedef int (*tocr_device_emit_ex_fn)(void *, const char *, size_t, float,
    int, int, int, int, int, int, int, int, int, float);
int tocr_device_recognize_redaction_v2(tocr_device_engine *, const void *, int,
    int, size_t, void *, float minimum_score, unsigned retry_limit,
    tocr_device_emit_ex_fn, void *, char *, size_t);

/* Match-driven policy API: emit every nonempty recognition, including scores
 * below kDropScore. Empty crops remain reason=1 metadata, without text.
 * Coordinates and CUDA failures keep the same validation as redaction_v2. */
int tocr_device_recognize_matches_v1(tocr_device_engine *, const void *, int,
    int, size_t, void *, float minimum_score, unsigned retry_limit,
    tocr_device_emit_ex_fn, void *, char *, size_t);

/* Optional batch capability: 1..8 immutable device ROIs, one revision at the
 * caller. Detection geometry is unchanged; recognition crops share GPU batches.
 * Every reader finishes before return. Discard ALL callbacks on any failure. */
typedef struct { const void *bgr; int width, height; size_t pitch; } tocr_device_image;
typedef int (*tocr_device_emit_batch_fn)(void *, size_t, const char *, size_t, float,
    int, int, int, int, int, int, int, int, int, float);
int tocr_device_recognize_matches_batch_v1(tocr_device_engine *,
    const tocr_device_image *, size_t, void *, float, unsigned,
    tocr_device_emit_batch_fn, void *, char *, size_t);

#ifdef __cplusplus
}
#endif

#endif
