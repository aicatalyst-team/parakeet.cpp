#ifndef PARAKEET_H
#define PARAKEET_H
#ifdef __cplusplus
extern "C" {
#endif
// Returns a static version string. Never null.
const char* parakeet_version(void);

// Transcribe a wav file with the given GGUF model using the CTC head.
// On success returns 0 and writes a malloc'd, NUL-terminated UTF-8 string to
// *out (the transcript; may be the empty string ""). The caller must release
// it with parakeet_free_string. On error returns nonzero and leaves *out
// unchanged. No C++ exceptions cross this boundary.
int parakeet_transcribe_file(const char* model_path, const char* wav_path, char** out);

// Frees a string previously returned via parakeet_transcribe_file. Safe on NULL.
void parakeet_free_string(char* s);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>
namespace pk {
// End-to-end CTC transcription: ModelLoader -> MelFrontend -> Encoder ->
// CTCDecoder -> ctc_greedy -> detokenize. Returns the decoded text (possibly
// empty). Throws std::runtime_error on failure (model/audio load, etc.).
std::string transcribe(const std::string& model_path, const std::string& wav_path);
} // namespace pk
#endif

#endif // PARAKEET_H
