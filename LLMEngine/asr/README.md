# asr

Speech-to-text with the **Qwen3-ASR** model. This is a C port of
[antirez/qwen-asr](https://github.com/antirez/qwen-asr) (MIT license). The audio front end and the
numerical kernels follow that project so results match.

What it contains:

- audio loading: WAV reading, resampling, log-mel spectrogram, silence-based segmentation (`gmb_asr_audio.c`);
- the model: encoder, decoder, tokenizer and the transcription pipeline;
- CPU kernels (matrix products, convolution, attention, normalization) and runtime 8-bit quantization.

Use it from the command line with `fastllm asr <model_dir> <file.wav>` (see FastLLM).
