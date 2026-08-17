// Copyright      2026  AXERA-TECH  (authors: Magnetar)
//
// Streaming zipformer ASR (encoder + decoder + joiner axmodels) inference.
//
// Mirrors the Python reference implementation
//   ax_pretrained_infer.py (icefall pruned_transducer_stateless7_streaming)
// with the same streaming parameters:
//   - 16 kHz mono PCM audio
//   - 80-dim fbank, 25 ms frame / 10 ms shift, dither=0, snip_edges=false
//   - encoder chunk: segment=103 frames, shift=96 frames (7-frame overlap)
//   - greedy search, context_size=2, blank_id=0, max_sym_per_frame=1
//   - 0.3 s tail padding

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ax_engine_api.h"
#include "ax_sys_api.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include "src/engine_wrapper.hpp"
#include "src/wav_reader.hpp"

#ifndef AXERA_TARGET_NAME
#define AXERA_TARGET_NAME "AXERA"
#endif
#ifndef DEFAULT_MODELS_DIR
#define DEFAULT_MODELS_DIR "inputs/axmodels_650N"
#endif

namespace {
constexpr int kSampleRate = 16000;
constexpr int kFeatureDim = 80;
constexpr int kEncoderDim = 512;   // joiner_dim
constexpr int kSegmentFrames = 103;  // decode_chunk_len(96) + 7 left context
constexpr int kChunkShift = 96;
constexpr int kContextSize = 2;
constexpr int kBlankId = 0;
constexpr float kTailSeconds = 0.3f;
constexpr int kStreamChunk = kSampleRate;  // feed waveform in 1 s chunks
using Clock = std::chrono::steady_clock;

double ElapsedSeconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

struct Args {
  std::string models_dir = DEFAULT_MODELS_DIR;
  std::string tokens = "inputs/lang_char_bpe/tokens.txt";
  std::string audio;
  std::string audio_dir;
};

void Usage(const char *program) {
  std::printf(
      "Usage: %s [--models-dir DIR] [--tokens FILE]\n"
      "           [--audio WAV | --audio-dir DIR]\n"
      "  --models-dir  directory containing encoder/decoder/joiner.axmodel\n"
      "  --tokens      path to tokens.txt (BPE symbol table)\n"
      "  --audio       single 16 kHz mono PCM WAV file\n"
      "  --audio-dir   decode all *.wav files in the directory\n",
      program);
}

Args ParseArgs(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("Missing value for " + key);
      return argv[i];
    };
    if (key == "--models-dir") {
      args.models_dir = value();
    } else if (key == "--tokens") {
      args.tokens = value();
    } else if (key == "--audio") {
      args.audio = value();
    } else if (key == "--audio-dir") {
      args.audio_dir = value();
    } else if (key == "-h" || key == "--help") {
      Usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }
  if (!args.audio.empty() && !args.audio_dir.empty()) {
    throw std::runtime_error("--audio and --audio-dir are mutually exclusive");
  }
  return args;
}

std::string Join(const std::string &left, const std::string &right) {
  return left.empty() || left.back() == '/' ? left + right
                                             : left + "/" + right;
}

class AxRuntime {
 public:
  AxRuntime() {
    if (AX_SYS_Init() != 0) throw std::runtime_error("AX_SYS_Init failed");
    sys_initialized_ = true;
    AX_ENGINE_NPU_ATTR_T attr{};
    if (AX_ENGINE_Init(&attr) != 0) {
      AX_SYS_Deinit();
      sys_initialized_ = false;
      throw std::runtime_error("AX_ENGINE_Init failed");
    }
    engine_initialized_ = true;
  }
  ~AxRuntime() {
    if (engine_initialized_) AX_ENGINE_Deinit();
    if (sys_initialized_) AX_SYS_Deinit();
  }

 private:
  bool sys_initialized_ = false;
  bool engine_initialized_ = false;
};

std::vector<std::string> LoadTokens(const std::string &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open tokens: " + path);
  std::vector<std::string> result;
  std::string line;
  int line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const std::size_t split = line.find_last_of(' ');
    if (split == std::string::npos) {
      throw std::runtime_error("Invalid token line " +
                               std::to_string(line_number));
    }
    const int id = std::stoi(line.substr(split + 1));
    if (id >= static_cast<int>(result.size())) {
      result.resize(static_cast<std::size_t>(id) + 1);
    }
    result[static_cast<std::size_t>(id)] = line.substr(0, split);
  }
  if (result.empty()) throw std::runtime_error("Empty token table: " + path);
  return result;
}

std::string TokensToText(const std::vector<std::string> &symbols,
                         const std::vector<int32_t> &hyp) {
  std::string text;
  for (std::size_t i = kContextSize; i < hyp.size(); ++i) {
    const int32_t id = hyp[i];
    if (id < 0 || static_cast<std::size_t>(id) >= symbols.size()) continue;
    text += symbols[static_cast<std::size_t>(id)];
  }
  // "\xE2\x96\x81" is the UTF-8 encoding of the BPE word-boundary symbol ▁
  const std::string marker = "\xE2\x96\x81";
  std::size_t pos = 0;
  while ((pos = text.find(marker, pos)) != std::string::npos) {
    text.replace(pos, marker.size(), " ");
  }
  const std::size_t begin = text.find_first_not_of(' ');
  if (begin == std::string::npos) return "";
  const std::size_t end = text.find_last_not_of(' ');
  return text.substr(begin, end - begin + 1);
}

// Mirrors OnnxModel::init_encoder_states / run_encoder / greedy_search of
// ax_pretrained_infer.py.
class ZipformerDecoder {
 public:
  ZipformerDecoder(EngineWrapper *encoder, EngineWrapper *decoder,
                   EngineWrapper *joiner, int vocab_size)
      : encoder_(encoder),
        decoder_(decoder),
        joiner_(joiner),
        vocab_size_(vocab_size) {
    encoder_output_frames_ =
        encoder_->GetOutputSizeByName("encoder_out") /
        (kEncoderDim * static_cast<int>(sizeof(float)));
    if (encoder_->GetInputSizeByName("x") !=
        kSegmentFrames * kFeatureDim * static_cast<int>(sizeof(float))) {
      throw std::runtime_error("Encoder input shape does not match segment=103");
    }
    if (encoder_->GetOutputSizeByName("encoder_out") %
            (kEncoderDim * static_cast<int>(sizeof(float))) !=
        0) {
      throw std::runtime_error("Unexpected encoder_out size");
    }
    if (decoder_->GetInputSizeByName("y") !=
        kContextSize * static_cast<int>(sizeof(int32_t))) {
      throw std::runtime_error("Decoder input shape does not match context=2");
    }
    if (decoder_->GetOutputSizeByName("decoder_out") !=
        kEncoderDim * static_cast<int>(sizeof(float))) {
      throw std::runtime_error("Unexpected decoder_out size");
    }
    if (joiner_->GetOutputSizeByName("logit") !=
        vocab_size_ * static_cast<int>(sizeof(float))) {
      throw std::runtime_error("Unexpected joiner logit size");
    }
    Reset();
  }

  // Reset encoder cached states and decoding context for a new utterance.
  void Reset() {
    for (std::size_t i = 0; i < encoder_->InputCount(); ++i) {
      const std::string &name = encoder_->InputName(i);
      if (name == "x") continue;
      if (encoder_->ZeroInputByName(name) != 0) {
        throw std::runtime_error("Failed to reset encoder state: " + name);
      }
    }
    hyp_.assign(kContextSize, kBlankId);
    decoder_out_.assign(kEncoderDim, 0.0f);
    RunDecoder();
  }

  // Run one encoder chunk over segment frames; returns the number of output
  // frames decoded (encoder_out frames).
  int AcceptChunk(const float *frames) {
    if (encoder_->SetInputByName(
            "x", frames,
            kSegmentFrames * kFeatureDim * sizeof(float)) != 0 ||
        encoder_->RunSync() != 0) {
      throw std::runtime_error("Encoder inference failed");
    }
    std::vector<float> encoder_out(
        static_cast<std::size_t>(encoder_output_frames_) * kEncoderDim);
    if (encoder_->GetOutputByName("encoder_out", encoder_out.data(),
                                  encoder_out.size() * sizeof(float)) != 0) {
      throw std::runtime_error("Failed to read encoder_out");
    }
    // Carry cached states: new_cached_* -> cached_*
    for (std::size_t i = 0; i < encoder_->InputCount(); ++i) {
      const std::string &input_name = encoder_->InputName(i);
      if (input_name == "x") continue;
      if (encoder_->CopyOutputToInputByName("new_" + input_name, input_name) !=
          0) {
        throw std::runtime_error("Failed to update encoder state: " +
                                 input_name);
      }
    }
    // Greedy search over the frames of this chunk (max_sym_per_frame=1).
    for (int frame = 0; frame < encoder_output_frames_; ++frame) {
      const float *cur_encoder_out = encoder_out.data() + frame * kEncoderDim;
      if (joiner_->SetInputByName("encoder_out", cur_encoder_out,
                                  kEncoderDim * sizeof(float)) != 0 ||
          joiner_->SetInputByName("decoder_out", decoder_out_.data(),
                                  kEncoderDim * sizeof(float)) != 0 ||
          joiner_->RunSync() != 0) {
        throw std::runtime_error("Joiner inference failed");
      }
      std::vector<float> logits(static_cast<std::size_t>(vocab_size_));
      if (joiner_->GetOutputByName("logit", logits.data(),
                                   logits.size() * sizeof(float)) != 0) {
        throw std::runtime_error("Failed to read joiner logit");
      }
      const int32_t y = static_cast<int32_t>(
          std::max_element(logits.begin(), logits.end()) - logits.begin());
      if (y != kBlankId) {
        hyp_.push_back(y);
        RunDecoder();
      }
    }
    return encoder_output_frames_;
  }

  const std::vector<int32_t> &hyp() const { return hyp_; }

 private:
  // Run the decoder with the last context_size tokens of hyp_.
  void RunDecoder() {
    std::array<int32_t, kContextSize> decoder_input{};
    for (int i = 0; i < kContextSize; ++i) {
      decoder_input[static_cast<std::size_t>(i)] =
          hyp_[hyp_.size() - kContextSize + i];
    }
    if (decoder_->SetInputByName(
            "y", decoder_input.data(),
            decoder_input.size() * sizeof(int32_t)) != 0 ||
        decoder_->RunSync() != 0) {
      throw std::runtime_error("Decoder inference failed");
    }
    if (decoder_->GetOutputByName("decoder_out", decoder_out_.data(),
                                  decoder_out_.size() * sizeof(float)) != 0) {
      throw std::runtime_error("Failed to read decoder_out");
    }
  }

  EngineWrapper *encoder_;
  EngineWrapper *decoder_;
  EngineWrapper *joiner_;
  int vocab_size_ = 0;
  int encoder_output_frames_ = 0;
  std::vector<int32_t> hyp_;
  std::vector<float> decoder_out_;
};

std::vector<std::string> ListWavFiles(const Args &args) {
  std::vector<std::string> files;
  if (!args.audio.empty()) {
    files.push_back(args.audio);
    return files;
  }
  const std::string dir = args.audio_dir.empty() ? "inputs/test_wavs"
                                                 : args.audio_dir;
  std::string listing =
      "find '" + dir + "' -maxdepth 1 -type f -name '*.wav' | sort";
  FILE *pipe = popen(listing.c_str(), "r");
  if (!pipe) throw std::runtime_error("Failed to list audio dir: " + dir);
  char buffer[4096];
  while (fgets(buffer, sizeof(buffer), pipe)) {
    std::string file = buffer;
    while (!file.empty() && (file.back() == '\n' || file.back() == '\r')) {
      file.pop_back();
    }
    if (!file.empty()) files.push_back(file);
  }
  pclose(pipe);
  if (files.empty()) throw std::runtime_error("No *.wav files found in " + dir);
  return files;
}

std::string DecodeFile(ZipformerDecoder *decoder,
                       const std::vector<std::string> &symbols,
                       const std::string &path, double *audio_seconds,
                       double *processing_seconds) {
  const auto audio_begin = Clock::now();
  const PcmWav wav = ReadPcmWav(path);
  if (wav.sample_rate != kSampleRate) {
    throw std::runtime_error("Input WAV must use 16 kHz sample rate: " + path);
  }
  std::vector<float> waveform(wav.samples.size() +
                              static_cast<std::size_t>(kTailSeconds *
                                                       kSampleRate));
  for (std::size_t i = 0; i < wav.samples.size(); ++i) {
    waveform[i] = static_cast<float>(wav.samples[i]) / 32768.0f;
  }
  const double audio_load_seconds = ElapsedSeconds(audio_begin, Clock::now());

  const auto process_begin = Clock::now();
  knf::FbankOptions options;
  options.frame_opts.samp_freq = kSampleRate;
  options.frame_opts.dither = 0.0f;
  options.frame_opts.frame_length_ms = 25.0f;
  options.frame_opts.frame_shift_ms = 10.0f;
  options.frame_opts.snip_edges = false;
  options.frame_opts.window_type = "povey";
  options.mel_opts.num_bins = kFeatureDim;
  options.mel_opts.low_freq = 20.0f;
  options.mel_opts.high_freq = -400.0f;
  options.energy_floor = 0.0f;
  knf::OnlineFbank fbank(options);

  decoder->Reset();
  int num_processed_frames = 0;
  std::vector<float> chunk(kSegmentFrames * kFeatureDim);
  for (std::size_t start = 0; start < waveform.size();
       start += kStreamChunk) {
    const std::size_t end = std::min(start + kStreamChunk, waveform.size());
    fbank.AcceptWaveform(kSampleRate, waveform.data() + start,
                         static_cast<int32_t>(end - start));
    while (fbank.NumFramesReady() - num_processed_frames >= kSegmentFrames) {
      for (int i = 0; i < kSegmentFrames; ++i) {
        std::memcpy(chunk.data() +
                        static_cast<std::size_t>(i) * kFeatureDim,
                    fbank.GetFrame(num_processed_frames + i),
                    kFeatureDim * sizeof(float));
      }
      num_processed_frames += kChunkShift;
      decoder->AcceptChunk(chunk.data());
    }
  }
  fbank.InputFinished();
  const double inference_seconds = ElapsedSeconds(process_begin, Clock::now());

  *audio_seconds = static_cast<double>(wav.samples.size()) / kSampleRate;
  *processing_seconds = audio_load_seconds + inference_seconds;
  return TokensToText(symbols, decoder->hyp());
}

void Run(const Args &args) {
  const std::vector<std::string> audio_files = ListWavFiles(args);
  const auto symbols = LoadTokens(args.tokens);

  const auto model_load_begin = Clock::now();
  AxRuntime runtime;
  EngineWrapper encoder;
  EngineWrapper decoder;
  EngineWrapper joiner;
  if (encoder.Init(Join(args.models_dir, "encoder.axmodel")) != 0 ||
      decoder.Init(Join(args.models_dir, "decoder.axmodel")) != 0 ||
      joiner.Init(Join(args.models_dir, "joiner.axmodel")) != 0) {
    throw std::runtime_error("Failed to load zipformer axmodels");
  }
  const int vocab_size =
      joiner.GetOutputSizeByName("logit") /
      static_cast<int>(sizeof(float));
  ZipformerDecoder zipformer_decoder(&encoder, &decoder, &joiner, vocab_size);
  const double model_load_seconds =
      ElapsedSeconds(model_load_begin, Clock::now());

  std::printf("target: %s\nmodel_load_seconds: %.3f\n\n", AXERA_TARGET_NAME,
              model_load_seconds);

  double total_rtf = 0.0;
  for (std::size_t i = 0; i < audio_files.size(); ++i) {
    double audio_seconds = 0.0;
    double processing_seconds = 0.0;
    const std::string text = DecodeFile(&zipformer_decoder, symbols,
                                        audio_files[i], &audio_seconds,
                                        &processing_seconds);
    const double rtf = audio_seconds > 0 ? processing_seconds / audio_seconds
                                         : 0.0;
    total_rtf += rtf;
    std::printf("[%zu/%zu] %s\n", i + 1, audio_files.size(),
                audio_files[i].c_str());
    std::printf("%s\n", text.c_str());
    std::printf(" Audio duration: %.3f s\n", audio_seconds);
    std::printf(" Audio processing time: %.3f s\n", processing_seconds);
    std::printf(" RTF (total_time/audio_duration): %.3f\n\n", rtf);
  }
  if (audio_files.size() > 1) {
    std::printf("Average RTF over %zu files: %.3f\n", audio_files.size(),
                total_rtf / audio_files.size());
  }
}
}  // namespace

int main(int argc, char **argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    Run(args);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
  }
}
