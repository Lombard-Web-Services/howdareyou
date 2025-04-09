# howdareyou
Simple micro-tremor VSA tool , to detect voice stress, using Fast Fourier Transform.
This script is an experimental tool that permit to assess veracity of a sound file based on stress analysis of micro-tremors.

## Dependencies
Here are the following dependencies to be installed.
```sh
sudo apt install -y libavcodec-dev libavformat-dev libavutil-dev libfftw3-dev libao-dev libavdevice-dev libsdl2-dev
```

## Compilation
Use g++ to compile the cpp file.
```
g++ -o howdareyou howdareyou.cpp -lavformat -lavcodec -lavutil -lfftw3 -lavdevice -lswresample -lm -pthread
chmod +x howdareyou
```

## Usage
```sh
./howdareyou --help
Usage: ./howdareyou [options] <audio_file.mp3/mp4>
Options:
  --audiolib <driver>       Specify audio driver (ignored, uses PulseAudio)
  --option disableplayback  Disable audio playback
  --option realtime Enable real-time playback and console output (default)
  --option subtitle Generate SRT subtitle file with analysis results
  --help     Display this help message

Examples:
  ./howdareyou test.mp3     # Real-time playback and console output
  ./howdareyou --option disableplayback test.mp3 # Analysis only, no playback
  ./howdareyou --option subtitle test.mp3 # Generate test.mp3.srt with results

```
