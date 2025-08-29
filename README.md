# 🎤 howdareyou 🗣️

Simple VSA tool, to detect voice stress, using Fast Fourier Transform.
This script is an experimental tool that permit to assess veracity of a speech, on a sound file and perform stress analysis of micro-tremors.

---

## 💻 Dependencies 🛠️

Here are the following dependencies to be installed.

```sh
sudo apt install -y libavcodec-dev libavformat-dev libavutil-dev libfftw3-dev libavdevice-dev libswresample-dev
```

---

## ⚙️ Compilation ✨

Use g++ to compile the cpp file.

```
g++ -o howdareyou howdareyou.cpp -lavformat -lavcodec -lavutil -lfftw3 -lavdevice -lswresample -lm -pthread
chmod +x howdareyou
```

---

## 🚀 Usage 📖

**howdareyou** is a commandline tool :

```sh
./howdareyou --help
Usage: ./howdareyou [options] <audio_file.mp3/mp4>
Options:
  --audiolib <driver>      Specify audio driver (ignored, uses PulseAudio)
  --option disableplayback  Disable audio playback
  --option realtime Enable real-time playback and console output (default)
  --option subtitle Generate SRT subtitle file with analysis results
  --help     Display this help message

Examples:
  ./howdareyou test.mp3      # Real-time playback and console output
  ./howdareyou --option disableplayback test.mp3 # Analysis only, no playback
  ./howdareyou --option subtitle test.mp3 # Generate test.mp3.srt with results
```

---

## 📜 License & Author 🧑‍💻

**License:** CC BY-NC-ND
![Logo de la licence CC BY-NC-ND](CC_BY-NC-ND.png)
**Author:** Thibaut Lombard
**LinkedIn:** [https://www.linkedin.com/in/thibautlombard/](https://www.linkedin.com/in/thibautlombard/)
**X:** [https://x.com/lombardweb](https://x.com/lombardweb)
**Repository:** [https://github.com/Lombard-Web-Services/howdareyou](https://github.com/Lombard-Web-Services/howdareyou)
