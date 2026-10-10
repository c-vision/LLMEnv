/* gmb_asr_audio.c — WAV + resample + log-mel per Qwen3-ASR.
 * Porting VERBATIM (parametri e formule) da antirez/qwen-asr `qwen_asr_audio.c` (MIT), per
 * fedeltà bit-per-bit del mel. Estratto: niente stdin/thread live (non serve al runtime). */
#include "gmb_asr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SR        GMB_ASR_SR
#define N_MEL     GMB_ASR_N_MEL
#define HOP       GMB_ASR_HOP
#define WIN       GMB_ASR_WIN
#define N_FFT     GMB_ASR_N_FFT
#define N_FREQ    GMB_ASR_N_FREQ

static uint16_t rd_u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd_u32(const uint8_t* p) { return (uint32_t)(p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24)); }

float* gmb_asr_wav_decode(const uint8_t* data, size_t file_size, int* out_n_samples) {
    if (data == NULL || file_size < 44 || memcmp(data, "RIFF", 4) != 0 || memcmp(data + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "gmb_asr_wav_decode: non e' un WAV valido\n");
        return NULL;
    }
    int channels = 0, sample_rate = 0, bits = 0, fmt = 0;
    const uint8_t* pcm = NULL; int pcm_size = 0;
    const uint8_t* p = data + 12; const uint8_t* end = data + file_size;
    while (p + 8 <= end) {
        uint32_t cs = rd_u32(p + 4);
        if (p + 8 + cs > end) break;
        if (memcmp(p, "fmt ", 4) == 0 && cs >= 16) {
            fmt = rd_u16(p + 8); channels = rd_u16(p + 10); sample_rate = (int)rd_u32(p + 12); bits = rd_u16(p + 22);
        } else if (memcmp(p, "data", 4) == 0) {
            pcm = p + 8; pcm_size = (int)cs;
            if (pcm + pcm_size > end) pcm_size = (int)(end - pcm);
        }
        p += 8 + cs; if (cs & 1) p++;
    }
    if (fmt != 1 || bits != 16 || pcm == NULL || channels < 1) {
        fprintf(stderr, "gmb_asr_wav_decode: formato non supportato (serve PCM 16-bit; fmt=%d bits=%d)\n", fmt, bits);
        return NULL;
    }
    int n_frames = pcm_size / (channels * 2);
    float* samples = (float*)malloc((size_t)(n_frames > 0 ? n_frames : 1) * sizeof(float));
    if (!samples) return NULL;
    const int16_t* src = (const int16_t*)pcm;
    for (int i = 0; i < n_frames; i++) {
        if (channels == 1) samples[i] = src[i] / 32768.0f;
        else { float sum = 0; for (int c = 0; c < channels; c++) { int16_t v; memcpy(&v, &src[i * channels + c], 2); sum += v; } samples[i] = (sum / channels) / 32768.0f; }
    }
    if (sample_rate != SR) {
        int nn = 0; float* rs = gmb_asr_resample(samples, n_frames, sample_rate, &nn);
        free(samples);
        if (!rs) return NULL;
        samples = rs; n_frames = nn;
    }
    *out_n_samples = n_frames;
    return samples;
}

float* gmb_asr_wav_load(const char* path, int* out_n_samples) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "gmb_asr_wav_load: impossibile aprire %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); if (sz <= 0) { fclose(f); return NULL; } fseek(f, 0, SEEK_SET);
    uint8_t* data = (uint8_t*)malloc((size_t)sz);
    if (!data || fread(data, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(data); return NULL; }
    fclose(f);
    float* s = gmb_asr_wav_decode(data, (size_t)sz, out_n_samples);
    free(data);
    return s;
}

/* I0 (Bessel modificata, I serie di potenze). */
static double bessel_i0(double x) {
    double sum = 1.0, term = 1.0, xx = x * x;
    for (int k = 1; k <= 20; k++) { term *= xx / (4.0 * (double)k * (double)k); sum += term; }
    return sum;
}

float* gmb_asr_resample(const float* in, int n_in, int sr_in, int* out_n) {
    if (in == NULL || n_in <= 0 || sr_in <= 0) return NULL;
    if (sr_in == SR) { float* c = (float*)malloc((size_t)n_in * sizeof(float)); if (c) memcpy(c, in, (size_t)n_in * sizeof(float)); *out_n = n_in; return c; }
    int new_n = (int)((long long)n_in * SR / sr_in);
    if (new_n <= 0) { *out_n = 0; return NULL; }
    float* out = (float*)malloc((size_t)new_n * sizeof(float));
    if (!out) return NULL;
    const int SINC_HALF = 16; const double BETA = 6.0;
    double ratio = (double)SR / (double)sr_in;
    double cutoff = (ratio < 1.0) ? ratio : 1.0;
    double inv_i0 = 1.0 / bessel_i0(BETA);
    for (int i = 0; i < new_n; i++) {
        double src_pos = (double)i / ratio;
        int center = (int)src_pos; double acc = 0.0, wsum = 0.0;
        for (int j = center - SINC_HALF + 1; j <= center + SINC_HALF; j++) {
            double d = (double)j - src_pos, x = d * cutoff;
            double s = (fabs(x) < 1e-9) ? 1.0 : sin(M_PI * x) / (M_PI * x);
            double npos = d / SINC_HALF, w;
            if (npos <= -1.0 || npos >= 1.0) w = 0.0; else w = bessel_i0(BETA * sqrt(1.0 - npos * npos)) * inv_i0;
            double coeff = s * w * cutoff;
            if (j >= 0 && j < n_in) acc += in[j] * coeff;
            wsum += coeff;
        }
        out[i] = (wsum > 1e-9) ? (float)(acc / wsum) : 0.0f;
    }
    *out_n = new_n;
    return out;
}

static float hz_to_mel(float f) { const float min_hz = 1000.0f, min_mel = 15.0f, step = 27.0f / logf(6.4f); float m = 3.0f * f / 200.0f; if (f >= min_hz) m = min_mel + logf(f / min_hz) * step; return m; }
static float mel_to_hz(float m) { const float min_hz = 1000.0f, min_mel = 15.0f, step = logf(6.4f) / 27.0f; float f = 200.0f * m / 3.0f; if (m >= min_mel) f = min_hz * expf(step * (m - min_mel)); return f; }

static float* build_mel_filters(void) {
    float* filters = (float*)calloc((size_t)N_MEL * N_FREQ, sizeof(float));
    if (!filters) return NULL;
    float fft_freqs[N_FREQ];
    for (int i = 0; i < N_FREQ; i++) fft_freqs[i] = (float)i * ((float)SR / 2.0f) / (float)(N_FREQ - 1);
    float mel_min = hz_to_mel(0.0f), mel_max = hz_to_mel((float)SR / 2.0f);
    float ff[N_MEL + 2], fd[N_MEL + 1];
    for (int i = 0; i < N_MEL + 2; i++) ff[i] = mel_to_hz(mel_min + (mel_max - mel_min) * (float)i / (float)(N_MEL + 1));
    for (int i = 0; i < N_MEL + 1; i++) { fd[i] = ff[i + 1] - ff[i]; if (fd[i] == 0.0f) fd[i] = 1e-6f; }
    for (int m = 0; m < N_MEL; m++) {
        float enorm = 2.0f / (ff[m + 2] - ff[m]);
        for (int f = 0; f < N_FREQ; f++) {
            float down = (fft_freqs[f] - ff[m]) / fd[m], up = (ff[m + 2] - fft_freqs[f]) / fd[m + 1];
            float val = fminf(down, up); if (val < 0.0f) val = 0.0f;
            filters[(size_t)m * N_FREQ + f] = val * enorm;
        }
    }
    return filters;
}

float* gmb_asr_mel(const float* samples, int n_samples, int* out_frames) {
    int pad_len = N_FFT / 2;
    int padded_len = n_samples + 2 * pad_len;
    float* padded = (float*)malloc((size_t)padded_len * sizeof(float));
    if (!padded) return NULL;
    for (int i = 0; i < pad_len; i++) { int s = pad_len - i; padded[i] = (s < n_samples) ? samples[s] : 0.0f; }
    memcpy(padded + pad_len, samples, (size_t)n_samples * sizeof(float));
    for (int i = 0; i < pad_len; i++) { int s = n_samples - 2 - i; padded[pad_len + n_samples + i] = (s >= 0) ? samples[s] : 0.0f; }
    int n_frames_total = (padded_len - N_FFT) / HOP + 1;
    int n_frames = n_frames_total - 1;
    if (n_frames <= 0) { fprintf(stderr, "gmb_asr_mel: audio troppo corto (%d campioni)\n", n_samples); free(padded); return NULL; }
    float* filters = build_mel_filters();
    if (!filters) { free(padded); return NULL; }
    float window[WIN];
    for (int i = 0; i < WIN; i++) window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)WIN));
    float* dft_cos = (float*)malloc((size_t)N_FREQ * N_FFT * sizeof(float));
    float* dft_sin = (float*)malloc((size_t)N_FREQ * N_FFT * sizeof(float));
    if (!dft_cos || !dft_sin) { free(dft_cos); free(dft_sin); free(filters); free(padded); return NULL; }
    for (int k = 0; k < N_FREQ; k++)
        for (int n = 0; n < N_FFT; n++) { float a = 2.0f * (float)M_PI * (float)k * (float)n / (float)N_FFT; dft_cos[k * N_FFT + n] = cosf(a); dft_sin[k * N_FFT + n] = sinf(a); }
    float* mel_tmp = (float*)calloc((size_t)n_frames * N_MEL, sizeof(float));
    float windowed[N_FFT], power[N_FREQ];
    float global_max = -1e30f;
    for (int t = 0; t < n_frames; t++) {
        int start = t * HOP;
        for (int i = 0; i < N_FFT; i++) windowed[i] = padded[start + i] * window[i];
        for (int k = 0; k < N_FREQ; k++) {
            float re = 0, im = 0; const float* cr = dft_cos + k * N_FFT; const float* sr = dft_sin + k * N_FFT;
            for (int n = 0; n < N_FFT; n++) { re += windowed[n] * cr[n]; im += windowed[n] * sr[n]; }
            power[k] = re * re + im * im;
        }
        for (int m = 0; m < N_MEL; m++) {
            float sum = 0.0f; const float* filt = filters + (size_t)m * N_FREQ;
            for (int k = 0; k < N_FREQ; k++) sum += filt[k] * power[k];
            if (sum < 1e-10f) sum = 1e-10f;
            float val = log10f(sum); mel_tmp[t * N_MEL + m] = val;
            if (val > global_max) global_max = val;
        }
    }
    float* mel = (float*)malloc((size_t)N_MEL * n_frames * sizeof(float));
    float min_val = global_max - 8.0f;
    for (int t = 0; t < n_frames; t++)
        for (int m = 0; m < N_MEL; m++) { float val = mel_tmp[t * N_MEL + m]; if (val < min_val) val = min_val; mel[m * n_frames + t] = (val + 4.0f) / 4.0f; }
    free(mel_tmp); free(dft_cos); free(dft_sin); free(padded); free(filters);
    *out_frames = n_frames;
    return mel;
}
