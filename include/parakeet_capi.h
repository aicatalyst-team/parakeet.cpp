#ifndef PARAKEET_CAPI_H
#define PARAKEET_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

// Flat C-API for parakeet.cpp — designed for dlopen / cgo / purego (LocalAI).
//
// All functions are extern "C" and never let a C++ exception cross the
// boundary. The model is loaded ONCE into an opaque `parakeet_ctx` and reused
// across transcribe calls. Returned strings are malloc'd UTF-8 owned by the
// caller and must be released with parakeet_capi_free_string.

// Opaque transcription context (wraps a loaded model + last-error buffer).
typedef struct parakeet_ctx parakeet_ctx;

// ABI version of this header/implementation. Bump on any breaking change to the
// function signatures or semantics below.
int parakeet_capi_abi_version(void);

// Load a GGUF model. Returns an owning context, or NULL on failure.
// The returned context must be released with parakeet_capi_free.
parakeet_ctx* parakeet_capi_load(const char* gguf_path);

// Free a context obtained from parakeet_capi_load. Safe on NULL.
void parakeet_capi_free(parakeet_ctx* ctx);

// Transcribe a WAV file. `decoder` selects the head:
//   0 = default (by arch: transducer for tdt/rnnt/hybrid, CTC for ctc),
//   1 = ctc (force CTC head),
//   2 = tdt/rnnt (force the transducer head).
// On success returns a malloc'd, NUL-terminated UTF-8 transcript (free with
// parakeet_capi_free_string). On error returns NULL and sets the context's
// last error (see parakeet_capi_last_error).
char* parakeet_capi_transcribe_path(parakeet_ctx* ctx, const char* wav_path,
                                    int decoder);

// Transcribe in-memory mono float PCM (`samples`, length `n_samples`). If
// `sample_rate != 16000` the audio is linearly resampled to 16 kHz first.
// `decoder` is as in parakeet_capi_transcribe_path. On success returns a
// malloc'd UTF-8 transcript (free with parakeet_capi_free_string); on error
// returns NULL and sets the context's last error.
char* parakeet_capi_transcribe_pcm(parakeet_ctx* ctx, const float* samples,
                                   int n_samples, int sample_rate, int decoder);

// Free a string previously returned by parakeet_capi_transcribe_*. Safe on NULL.
void parakeet_capi_free_string(char* s);

// Human-readable description of the last error on `ctx`, or "" if none.
// The returned pointer is owned by the context and valid until the next call on
// it (or until parakeet_capi_free). Returns "" if `ctx` is NULL.
const char* parakeet_capi_last_error(parakeet_ctx* ctx);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // PARAKEET_CAPI_H
