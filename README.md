# howdareyou
Simple micro-tremor VSA tool , to detect voice stress, using Fast Fourier Transform.
This script is an experimental tool that permit to assess veracity of a sound file based on stress analysis.

## Dependencies
Here are the following dependencies : 
```sh

```

## Compilation
Use g++ to compile the cpp file.
```
g++ -o howdareyou howdareyou.cpp -lavformat -lavcodec -lavutil -lfftw3 -lavdevice -lswresample -lm -pthread
```

## Usage
