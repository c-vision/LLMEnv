/* llm_mlx_kvdisk_io_test.c — round-trip MLX -> blob -> disco -> blob -> MLX (§G).
 * Verifica: serializzazione f16 e f32 bit-exact, shape/dtype preservati, e che il blob
 * passi per llm_kv_disk (put/get) senza alterazioni. Linka mlx-c + libmlx. */
#include "llm_mlx_kvdisk_io.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

int main(void) {
    mlx_device dev = mlx_device_new_type(MLX_CPU, 0);
    mlx_stream stream = mlx_stream_new_device(dev);

    const int sh[4] = { 1, 2, 8, 64 };
    const size_t nel = (size_t)sh[0] * sh[1] * sh[2] * sh[3];
    float* fdata = (float*)malloc(nel * sizeof(float));
    for (size_t i = 0; i < nel; i++) fdata[i] = (float)((int)(i % 97) - 48) * 0.125f;
    mlx_array a32 = mlx_array_new_data(fdata, sh, 4, MLX_FLOAT32);
    mlx_array a16 = mlx_array_new();
    CHECK(mlx_astype(&a16, a32, MLX_FLOAT16, stream) == 0);
    CHECK(mlx_array_eval(a16) == 0);
    CHECK(mlx_array_eval(a32) == 0);

    /* blob = [f16 layer3 k] + [f32 layer3 v] */
    uint8_t* buf = NULL; size_t len = 0, cap = 0;
    CHECK(llm_kvdisk_blob_append_tensor(&buf, &len, &cap, a16, 3, 0) == 0);
    CHECK(llm_kvdisk_blob_append_tensor(&buf, &len, &cap, a32, 3, 1) == 0);
    CHECK(len > 0);

    /* passa per il tier disco (scratch temporaneo) */
    char path[] = "/tmp/llm_kvdisk_io.XXXXXX.scratch";
    KvDisk kd = llm_kv_disk_open(path, 64, 0);
    CHECK(kd.f != NULL);
    uint64_t key = llm_kv_disk_hash(buf, len);
    CHECK(llm_kv_disk_put(&kd, key, buf, len) == 0);
    uint8_t* buf2 = (uint8_t*)malloc(len);
    size_t got = 0;
    CHECK(llm_kv_disk_get(&kd, key, buf2, len, &got) == 0);
    CHECK(got == len && memcmp(buf, buf2, len) == 0);
    llm_kv_disk_close(&kd);

    /* decodifica e confronto bit-exact */
    size_t off = 0; int layer = -1, kind = -1;
    mlx_array r16 = mlx_array_new(), r32 = mlx_array_new();
    CHECK(llm_kvdisk_blob_read_tensor(buf2, got, &off, &r16, &layer, &kind) == 0);
    CHECK(layer == 3 && kind == 0);
    CHECK(llm_kvdisk_blob_read_tensor(buf2, got, &off, &r32, &layer, &kind) == 0);
    CHECK(layer == 3 && kind == 1);
    CHECK(off == got);
    CHECK(mlx_array_eval(r16) == 0 && mlx_array_eval(r32) == 0);

    CHECK(mlx_array_dtype(r16) == MLX_FLOAT16 && mlx_array_size(r16) == nel);
    CHECK(mlx_array_dtype(r32) == MLX_FLOAT32 && mlx_array_size(r32) == nel);
    CHECK(memcmp(mlx_array_data_float16(a16), mlx_array_data_float16(r16), nel * 2) == 0);
    CHECK(memcmp(mlx_array_data_float32(a32), mlx_array_data_float32(r32), nel * 4) == 0);

    mlx_array_free(r16); mlx_array_free(r32);
    mlx_array_free(a16); mlx_array_free(a32);
    free(fdata); free(buf); free(buf2);
    mlx_stream_free(stream);
    mlx_device_free(dev);
    if (fails == 0) printf("llm_mlx_kvdisk_io_test: OK (%zu byte blob)\n", len);
    return fails ? 1 : 0;
}
