// by thibaut LOMBARD
// (Lombard Web) @lombardweb
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <thread>
#include <chrono>
#include <atomic>
#include <fstream>
#include <iomanip> // For std::setw and std::setfill
#include <fftw3.h>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavdevice/avdevice.h>
#include <libswresample/swresample.h> // Added for resampling
}

// Structure to hold analysis results for SRT generation
struct AnalysisResult {
    double start_time;
    double end_time;
    std::string result;
    int confidence;
};

// FFT wrapper class using FFTW
class FFTAnalyzer {
    unsigned int size;
    fftw_plan plan;
    double* in;
    fftw_complex* out;

public:
    FFTAnalyzer(unsigned int frame_size) : size(frame_size) {
 in = fftw_alloc_real(size);
 out = fftw_alloc_complex(size);
 plan = fftw_plan_dft_r2c_1d(size, in, out, FFTW_ESTIMATE);
    }

    ~FFTAnalyzer() {
 fftw_destroy_plan(plan);
 fftw_free(in);
 fftw_free(out);
    }

    void compute_fft(double* result, const double* data) {
 for (unsigned int i = 0; i < size; ++i) {
     in[i] = data[i];
 }
 fftw_execute(plan);
 for (unsigned int i = 0; i < size / 2; ++i) {
     result[i] = sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
 }
    }
};

// Analyze microtremors for stress detection, return result for SRT
AnalysisResult analyze_stress(const std::vector<double>& window, int sample_rate, double start_time, bool print_to_console) {
    const int frame_size = 16384;
    FFTAnalyzer fft(frame_size);
    double data[frame_size] = {0};
    double result[frame_size / 2];

    size_t i = 0;
    for (; i < window.size() && i < frame_size; ++i) {
 data[i] = window[i];
    }
    for (; i < frame_size; ++i) {
 data[i] = 0.0; // Zero-pad
    }

    fft.compute_fft(result, data);

    double bin_width = static_cast<double>(sample_rate) / frame_size;
    int normal_start = static_cast<int>(8.0 / bin_width);
    int normal_end = static_cast<int>(9.0 / bin_width);
    int stress_start = static_cast<int>(11.0 / bin_width);
    int stress_end = static_cast<int>(12.0 / bin_width);

    double normal_power = 0.0, stress_power = 0.0;
    for (int j = normal_start; j <= normal_end && j < frame_size / 2; ++j) {
 normal_power += result[j] * result[j];
    }
    for (int j = stress_start; j <= stress_end && j < frame_size / 2; ++j) {
 stress_power += result[j] * result[j];
    }

    normal_power /= (normal_end - normal_start + 1);
    stress_power /= (stress_end - stress_start + 1);

    double total_power = normal_power + stress_power;
    double stress_ratio = (total_power > 0) ? (stress_power / total_power) : 0.0;
    double confidence = std::min(stress_ratio * 200.0, 100.0);

    std::string timestamp = "[" + std::to_string(static_cast<int>(start_time)) + "s]";
    std::string result_str;
    if (stress_power > normal_power && stress_ratio > 0.5) {
 result_str = "PROBABLE LIE DETECTED";
    } else if (stress_power > normal_power) {
 result_str = "POSSIBLE LIE DETECTED";
    } else {
 result_str = "PROBABLE TRUTH DETECTED";
    }

    if (print_to_console) {
 std::cout << "\r" << timestamp << " Analysis Result: " << result_str;
 std::cout << " Confidence: [";
 int bar_length = static_cast<int>(confidence / 2);
 for (int k = 0; k < 50; ++k) {
     std::cout << (k < bar_length ? "=" : " ");
 }
 std::cout << "] " << static_cast<int>(confidence) << "%";
 std::cout.flush();
    }

    return {start_time, start_time + 1.0, result_str, static_cast<int>(confidence)};
}

// Generate SRT subtitle file
void generate_srt(const char* filename, const std::vector<AnalysisResult>& results) {
    std::string srt_filename = std::string(filename) + ".srt";
    std::ofstream srt_file(srt_filename);
    if (!srt_file.is_open()) {
 std::cerr << "Failed to create SRT file: " << srt_filename << std::endl;
 return;
    }

    for (size_t i = 0; i < results.size(); ++i) {
 int start_sec = static_cast<int>(results[i].start_time);
 int start_ms = static_cast<int>((results[i].start_time - start_sec) * 1000);
 int end_sec = static_cast<int>(results[i].end_time);
 int end_ms = static_cast<int>((results[i].end_time - end_sec) * 1000);

 srt_file << i + 1 << "\n";
 srt_file << std::setw(2) << std::setfill('0') << start_sec / 3600 << ":"
   << std::setw(2) << std::setfill('0') << (start_sec % 3600) / 60 << ":"
   << std::setw(2) << std::setfill('0') << start_sec % 60 << ","
   << std::setw(3) << std::setfill('0') << start_ms << " --> "
   << std::setw(2) << std::setfill('0') << end_sec / 3600 << ":"
   << std::setw(2) << std::setfill('0') << (end_sec % 3600) / 60 << ":"
   << std::setw(2) << std::setfill('0') << end_sec % 60 << ","
   << std::setw(3) << std::setfill('0') << end_ms << "\n";
 srt_file << "Analysis Result: " << results[i].result << " (" << results[i].confidence << "%)\n\n";
    }

    srt_file.close();
    std::cout << "Generated SRT file: " << srt_filename << std::endl;
}

// Analysis thread: Decode and analyze audio
void analysis_thread(const char* filename, std::atomic<bool>& running, bool realtime, bool generate_srt_flag, std::vector<AnalysisResult>& results) {
    AVFormatContext* format_ctx = avformat_alloc_context();
    AVCodecContext* codec_ctx = nullptr;
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    bool error = false;

    if (!format_ctx || !packet || !frame) {
 std::cerr << "Analysis: Memory allocation failed" << std::endl;
 error = true;
    } else if (avformat_open_input(&format_ctx, filename, nullptr, nullptr) < 0) {
 std::cerr << "Analysis: Could not open file: " << filename << std::endl;
 error = true;
    } else if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
 std::cerr << "Analysis: Could not find stream info" << std::endl;
 error = true;
    } else {
 int audio_stream_idx = -1;
 for (unsigned int i = 0; i < format_ctx->nb_streams; ++i) {
     if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
  audio_stream_idx = i;
  break;
     }
 }

 if (audio_stream_idx == -1) {
     std::cerr << "Analysis: No audio stream found" << std::endl;
     error = true;
 } else {
     const AVCodec* codec = avcodec_find_decoder(format_ctx->streams[audio_stream_idx]->codecpar->codec_id);
     if (!codec) {
  std::cerr << "Analysis: Codec not found" << std::endl;
  error = true;
     } else {
  codec_ctx = avcodec_alloc_context3(codec);
  if (!codec_ctx) {
      std::cerr << "Analysis: Could not allocate codec context" << std::endl;
      error = true;
  } else if (avcodec_parameters_to_context(codec_ctx, format_ctx->streams[audio_stream_idx]->codecpar) < 0) {
      std::cerr << "Analysis: Failed to copy codec parameters" << std::endl;
      error = true;
  } else if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
      std::cerr << "Analysis: Could not open codec" << std::endl;
      error = true;
  } else {
      int sample_rate = codec_ctx->sample_rate;
      const double update_interval = 1.0;
      const size_t samples_per_update = sample_rate * update_interval;
      std::vector<double> window;
      double elapsed_time = 0.0;
      auto start_time = std::chrono::steady_clock::now();

      while (running && av_read_frame(format_ctx, packet) >= 0) {
   if (packet->stream_index == audio_stream_idx) {
       if (avcodec_send_packet(codec_ctx, packet) >= 0) {
    while (running && avcodec_receive_frame(codec_ctx, frame) >= 0) {
        int num_samples = frame->nb_samples;
        float* data = (float*)frame->data[0];

        for (int i = 0; i < num_samples; ++i) {
     window.push_back(static_cast<double>(data[i]));
        }

        double chunk_duration = static_cast<double>(num_samples) / sample_rate;
        if (realtime) {
     auto now = std::chrono::steady_clock::now();
     auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time).count() / 1e6;
     double expected_time = elapsed_time + chunk_duration;
     if (elapsed < expected_time) {
         std::this_thread::sleep_for(std::chrono::microseconds(
      static_cast<long>((expected_time - elapsed) * 1e6)
         ));
     }
        }
        elapsed_time += chunk_duration;

        while (window.size() >= samples_per_update) {
     std::vector<double> analysis_window(window.begin(), window.begin() + samples_per_update);
     AnalysisResult res = analyze_stress(analysis_window, sample_rate, elapsed_time - update_interval, realtime);
     if (generate_srt_flag) {
         results.push_back(res);
     }
     window.erase(window.begin(), window.begin() + samples_per_update);
        }
    }
       }
   }
   av_packet_unref(packet);
      }

      if (!window.empty() && running) {
   elapsed_time += window.size() / static_cast<double>(sample_rate);
   AnalysisResult res = analyze_stress(window, sample_rate, elapsed_time - (window.size() / static_cast<double>(sample_rate)), realtime);
   if (generate_srt_flag) {
       results.push_back(res);
   }
      }
  }
     }
 }
    }

    if (frame) av_frame_free(&frame);
    if (packet) av_packet_free(&packet);
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (format_ctx) avformat_close_input(&format_ctx);
}

// Playback thread: Play audio using FFmpeg with PulseAudio and resampling
void playback_thread(const char* filename, std::atomic<bool>& running) {
    AVFormatContext* input_ctx = avformat_alloc_context();
    AVFormatContext* output_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    SwrContext* swr_ctx = nullptr;
    bool error = false;

    if (!input_ctx || !packet || !frame) {
 std::cerr << "Playback: Memory allocation failed" << std::endl;
 error = true;
    } else if (avformat_open_input(&input_ctx, filename, nullptr, nullptr) < 0) {
 std::cerr << "Playback: Could not open file: " << filename << std::endl;
 error = true;
    } else if (avformat_find_stream_info(input_ctx, nullptr) < 0) {
 std::cerr << "Playback: Could not find stream info" << std::endl;
 error = true;
    } else {
 int audio_stream_idx = -1;
 for (unsigned int i = 0; i < input_ctx->nb_streams; ++i) {
     if (input_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
  audio_stream_idx = i;
  break;
     }
 }

 if (audio_stream_idx == -1) {
     std::cerr << "Playback: No audio stream found" << std::endl;
     error = true;
 } else {
     const AVCodec* codec = avcodec_find_decoder(input_ctx->streams[audio_stream_idx]->codecpar->codec_id);
     if (!codec) {
  std::cerr << "Playback: Codec not found" << std::endl;
  error = true;
     } else {
  codec_ctx = avcodec_alloc_context3(codec);
  if (!codec_ctx) {
      std::cerr << "Playback: Could not allocate codec context" << std::endl;
      error = true;
  } else if (avcodec_parameters_to_context(codec_ctx, input_ctx->streams[audio_stream_idx]->codecpar) < 0) {
      std::cerr << "Playback: Failed to copy codec parameters" << std::endl;
      error = true;
  } else if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
      std::cerr << "Playback: Could not open codec" << std::endl;
      error = true;
  } else {
      const AVOutputFormat* output_format = av_guess_format("pulse", nullptr, nullptr);
      if (!output_format) {
   std::cerr << "Playback: Could not find PulseAudio output format" << std::endl;
   error = true;
      } else if (avformat_alloc_output_context2(&output_ctx, output_format, nullptr, nullptr) < 0) {
   std::cerr << "Playback: Could not allocate output context" << std::endl;
   error = true;
      } else {
   AVStream* out_stream = avformat_new_stream(output_ctx, nullptr);
   if (!out_stream) {
       std::cerr << "Playback: Failed to create output stream" << std::endl;
       error = true;
   } else {
       out_stream->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
       out_stream->codecpar->codec_id = AV_CODEC_ID_PCM_S16LE;
       out_stream->codecpar->sample_rate = codec_ctx->sample_rate;
       out_stream->codecpar->ch_layout.nb_channels = codec_ctx->ch_layout.nb_channels;
       av_channel_layout_default(&out_stream->codecpar->ch_layout, out_stream->codecpar->ch_layout.nb_channels);
       out_stream->codecpar->format = AV_SAMPLE_FMT_S16;

       AVDictionary* opts = nullptr;
       av_dict_set(&opts, "server", nullptr, 0); // Default PulseAudio server
       if (avio_open2(&output_ctx->pb, "default", AVIO_FLAG_WRITE, nullptr, &opts) < 0) {
    std::cerr << "Playback: Could not open PulseAudio default output" << std::endl;
    error = true;
       }
       av_dict_free(&opts);

       if (!error && avformat_write_header(output_ctx, nullptr) < 0) {
    std::cerr << "Playback: Could not write header" << std::endl;
    error = true;
       } else {
    // Set up resampling context
    swr_ctx = swr_alloc_set_opts(
        nullptr,
        av_get_default_channel_layout(codec_ctx->ch_layout.nb_channels), // Out layout
        AV_SAMPLE_FMT_S16,      // Out format
        codec_ctx->sample_rate,        // Out sample rate
        av_get_default_channel_layout(codec_ctx->ch_layout.nb_channels), // In layout
        codec_ctx->sample_fmt,         // In format
        codec_ctx->sample_rate,        // In sample rate
        0, nullptr
    );
    if (!swr_ctx || swr_init(swr_ctx) < 0) {
        std::cerr << "Playback: Failed to initialize resampling context" << std::endl;
        error = true;
    } else {
        auto start_time = std::chrono::steady_clock::now();
        double elapsed_time = 0.0;
        int64_t sample_count = 0;

        while (running && !error && av_read_frame(input_ctx, packet) >= 0) {
     if (packet->stream_index == audio_stream_idx) {
         if (avcodec_send_packet(codec_ctx, packet) >= 0) {
      while (running && avcodec_receive_frame(codec_ctx, frame) >= 0) {
          int num_samples = frame->nb_samples;

          // Calculate output samples with potential delay
          int out_samples = av_rescale_rnd(
       swr_get_delay(swr_ctx, codec_ctx->sample_rate) + num_samples,
       codec_ctx->sample_rate, codec_ctx->sample_rate, AV_ROUND_UP
          );

          std::vector<int16_t> pcm_buffer(out_samples * codec_ctx->ch_layout.nb_channels);
          uint8_t* out_data[1] = { (uint8_t*)pcm_buffer.data() };
          int converted = swr_convert(
       swr_ctx, out_data, out_samples,
       (const uint8_t**)frame->data, num_samples
          );
          if (converted < 0) {
       std::cerr << "Playback: Resampling failed" << std::endl;
       error = true;
       break;
          }
          int pcm_data_size = converted * codec_ctx->ch_layout.nb_channels * sizeof(int16_t);

          AVPacket* out_packet = av_packet_alloc();
          if (!out_packet) {
       std::cerr << "Playback: Failed to allocate output packet" << std::endl;
       error = true;
       break;
          }
          out_packet->data = (uint8_t*)pcm_buffer.data();
          out_packet->size = pcm_data_size;
          out_packet->stream_index = 0;
          out_packet->pts = sample_count;
          sample_count += converted;

          double chunk_duration = static_cast<double>(converted) / codec_ctx->sample_rate;
          auto now = std::chrono::steady_clock::now();
          auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time).count() / 1e6;
          double expected_time = elapsed_time + chunk_duration;
          if (elapsed < expected_time) {
       std::this_thread::sleep_for(std::chrono::microseconds(
           static_cast<long>((expected_time - elapsed) * 1e6)
       ));
          }
          elapsed_time += chunk_duration;

          if (output_ctx->pb && av_write_frame(output_ctx, out_packet) < 0) {
       std::cerr << "Playback: Failed to write frame" << std::endl;
       error = true;
          }
          av_packet_free(&out_packet);
      }
         }
     }
     av_packet_unref(packet);
        }

        if (running && !error) {
     av_write_trailer(output_ctx);
        }
    }
       }
   }
      }
  }
     }
 }
    }

    if (swr_ctx) swr_free(&swr_ctx);
    if (frame) av_frame_free(&frame);
    if (packet) av_packet_free(&packet);
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (output_ctx) {
 if (output_ctx->pb) avio_closep(&output_ctx->pb);
 avformat_free_context(output_ctx);
    }
    if (input_ctx) avformat_close_input(&input_ctx);
}

// Display help message
void display_help() {
    std::cout << "Usage: ./howdareyou [options] <audio_file.mp3/mp4>\n";
    std::cout << "Options:\n";
    std::cout << "  --audiolib <driver>       Specify audio driver (ignored, uses PulseAudio)\n";
    std::cout << "  --option disableplayback  Disable audio playback\n";
    std::cout << "  --option realtime Enable real-time playback and console output (default)\n";
    std::cout << "  --option subtitle Generate SRT subtitle file with analysis results\n";
    std::cout << "  --help     Display this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  ./howdareyou test.mp3     # Real-time playback and console output\n";
    std::cout << "  ./howdareyou --option disableplayback test.mp3 # Analysis only, no playback\n";
    std::cout << "  ./howdareyou --option subtitle test.mp3 # Generate test.mp3.srt with results\n";
}

// Forward declaration
void generate_srt(const char* filename, const std::vector<AnalysisResult>& results);

// Main function to launch threads
void process_audio(const char* filename, bool enable_playback, bool realtime, bool generate_srt_flag) {
    AVFormatContext* format_ctx = avformat_alloc_context();
    if (!format_ctx) {
 std::cerr << "Could not allocate format context" << std::endl;
 return;
    }

    if (avformat_open_input(&format_ctx, filename, nullptr, nullptr) < 0) {
 std::cerr << "Could not open file for initial check: " << filename << std::endl;
 avformat_free_context(format_ctx);
 return;
    }

    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
 std::cerr << "Could not find stream info" << std::endl;
 avformat_free_context(format_ctx);
 return;
    }

    int audio_stream_idx = -1;
    for (unsigned int i = 0; i < format_ctx->nb_streams; ++i) {
 if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
     audio_stream_idx = i;
     break;
 }
    }

    if (audio_stream_idx == -1) {
 std::cerr << "No audio stream found" << std::endl;
 avformat_free_context(format_ctx);
 return;
    }

    AVCodecContext* codec_ctx = avcodec_alloc_context3(nullptr);
    if (!codec_ctx) {
 std::cerr << "Could not allocate codec context" << std::endl;
 avformat_free_context(format_ctx);
 return;
    }

    if (avcodec_parameters_to_context(codec_ctx, format_ctx->streams[audio_stream_idx]->codecpar) < 0) {
 std::cerr << "Failed to copy codec parameters" << std::endl;
 avcodec_free_context(&codec_ctx);
 avformat_free_context(format_ctx);
 return;
    }

    int sample_rate = codec_ctx->sample_rate;
    int channels = codec_ctx->ch_layout.nb_channels;
    std::cout << "Processing " << filename << " (Sample Rate: " << sample_rate << " Hz, Channels: " << channels << ")\n";

    avcodec_free_context(&codec_ctx);
    avformat_free_context(format_ctx);

    std::atomic<bool> running(true);
    std::vector<AnalysisResult> results;
    std::thread analysis(analysis_thread, filename, std::ref(running), realtime, generate_srt_flag, std::ref(results));
    std::thread playback;

    if (enable_playback) {
 playback = std::thread(playback_thread, filename, std::ref(running));
 std::cout << "Press 'q' and Enter to quit...\n";
    } else {
 std::cout << "Playback disabled. Press 'q' and Enter to quit...\n";
    }

    char input;
    while (running) {
 std::cin.get(input);
 if (input == 'q' || input == 'Q') {
     running = false;
     break;
 }
    }

    analysis.join();
    if (enable_playback) {
 playback.join();
    }

    if (generate_srt_flag && !results.empty()) {
 generate_srt(filename, results);
    }

    std::cout << "\nFinished processing " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    const char* filename = nullptr;
    const char* audio_driver = nullptr;
    bool enable_playback = true; // Default: playback enabled
    bool realtime = true; // Default: real-time mode
    bool generate_srt = false;   // Default: no SRT generation

    for (int i = 1; i < argc; ++i) {
 if (strcmp(argv[i], "--help") == 0) {
     display_help();
     return 0;
 } else if (strcmp(argv[i], "--audiolib") == 0) {
     if (i + 1 < argc) {
  audio_driver = argv[++i]; // Ignored, but parsed
     } else {
  std::cerr << "Error: --audiolib requires a driver name (ignored)\n";
  return 1;
     }
 } else if (strcmp(argv[i], "--option") == 0) {
     if (i + 1 < argc) {
  i++;
  if (strcmp(argv[i], "disableplayback") == 0) {
      enable_playback = false;
  } else if (strcmp(argv[i], "realtime") == 0) {
      realtime = true;
  } else if (strcmp(argv[i], "subtitle") == 0) {
      generate_srt = true;
  } else {
      std::cerr << "Error: Unknown option value '" << argv[i] << "' for --option\n";
      display_help();
      return 1;
  }
     } else {
  std::cerr << "Error: --option requires a value\n";
  return 1;
     }
 } else if (argv[i][0] != '-') {
     filename = argv[i];
 } else {
     std::cerr << "Unknown option: " << argv[i] << "\n";
     std::cerr << "Use --help for usage information\n";
     return 1;
 }
    }

    if (!filename) {
 std::cerr << "Error: No audio file specified\n";
 std::cerr << "Usage: ./howdareyou [options] <audio_file.mp3/mp4>\n";
 std::cerr << "Use --help for more information\n";
 return 1;
    }

    avdevice_register_all();
    process_audio(filename, enable_playback, realtime, generate_srt);
    return 0;
}
