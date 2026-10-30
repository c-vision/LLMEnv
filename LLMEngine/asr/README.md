# asr

Speech-to-text with the **Qwen3-ASR** model. This is a C port of
[antirez/qwen-asr](https://github.com/antirez/qwen-asr) (MIT license). The audio front end and the
numerical kernels follow that project so results match.

What it contains:

- audio loading: WAV reading, resampling, log-mel spectrogram, silence-based segmentation (`gmb_asr_audio.c`);
- the model: encoder, decoder, tokenizer and the transcription pipeline;
- CPU kernels (matrix products, convolution, attention, normalization) and runtime 8-bit quantization.

Use it from the command line with `fastllm asr <model_dir> <file.wav>` (see FastLLM).

## Parakeet-TDT (GGUF)

`llm_parakeet.h` runs [NVIDIA Parakeet-TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) from a GGUF file
(`general.architecture = parakeet`), a multilingual speech recognizer for 25 European languages. It is written from the
model description, in C, for the CPU (matrix products through Accelerate on macOS), and it is separate from the Qwen3-ASR
code above.

- Front end: pre-emphasis, STFT, the mel filter bank stored in the GGUF, log, per-feature normalization.
- Encoder: 8x convolutional subsampling, then 24 Conformer blocks (relative-position attention, GLU convolution, batch norm).
- Decoder: two-layer LSTM prediction network and a joint network. Each step picks a token and a duration (how many frames
  to skip), so decoding is fast.
- Audio longer than four minutes is cut at the quietest point near the end of each segment.

Measured on an Apple Silicon Mac: a 216 s recording is transcribed in 7.6 s. The JFK sample (11 s) comes out word for
word, and Italian and English text-to-speech samples are recognized correctly.

```sh
cmake -S LLMEngine/asr -B build/asr && cmake --build build/asr --target llm_parakeet_test
build/asr/llm_parakeet_test ~/ai/Models/gguf/parakeet-tdt-0.6b-v3-q8_0.gguf speech.wav [expected.txt]
```

The model weights are NVIDIA's, under the CC-BY-4.0 license of the original model.
