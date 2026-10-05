/* moe_ep_socket.c — ESECUZIONE DISTRIBUITA EP cross-NODO via SOCKET (§4.6). Il padre ascolta su
 * TCP; ogni rank (fork) calcola il contributo dei suoi esperti e lo INVIA via socket; il padre
 * raccoglie tutti i parziali e fa l'ALL-REDUCE. Stesso meccanismo su loopback (test) o tra nodi.
 * Uso: moe_ep_socket [world] [port]   (default 3 rank, porta 0 = effimera su 127.0.0.1) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
extern int llmengine_moe_diverge_plan(const int*, int, int, int*, int*);
extern int llmengine_moe_diverge_forward(const float*, int, int, int, const float*, int, const int*, const float*, int, float*);

static int recv_all(int fd, void* buf, size_t n) { size_t off = 0; while (off < n) { ssize_t r = recv(fd, (char*)buf+off, n-off, 0); if (r <= 0) return -1; off += (size_t)r; } return 0; }
static int send_all(int fd, const void* buf, size_t n) { size_t off = 0; while (off < n) { ssize_t r = send(fd, (const char*)buf+off, n-off, 0); if (r <= 0) return -1; off += (size_t)r; } return 0; }

int main(int argc, char** argv) {
    int world = argc > 1 ? atoi(argv[1]) : 3; if (world < 1) world = 1;
    int port = argc > 2 ? atoi(argv[2]) : 0;
    int T = 8, H = 6, O = 5, E = 6, K = 2, n = T*K, NO = T*O;
    float* x = malloc(sizeof(float)*T*H); for (int i = 0; i < T*H; i++) x[i] = sinf(i*0.3f);
    float* W = malloc(sizeof(float)*(size_t)E*H*O); for (int i = 0; i < E*H*O; i++) W[i] = cosf(i*0.17f);
    int* aex = malloc(sizeof(int)*n); float* aw = malloc(sizeof(float)*n); int sd = 12345;
    for (int a = 0; a < n; a++) { sd = (sd*1103515245+12345)&0x7fffffff; aex[a] = sd % E; aw[a] = 0.25f + (float)((sd>>8)%4)/4.0f; }
    int* start = malloc(sizeof(int)*(E+1)); int* sorted = malloc(sizeof(int)*n);
    if (llmengine_moe_diverge_plan(aex, n, E, start, sorted) != 0) { printf("FAIL plan\n"); return 1; }
    float* ref = calloc((size_t)NO, sizeof(float));
    llmengine_moe_diverge_forward(x, T, H, O, W, E, aex, aw, K, ref);

    int ls = socket(AF_INET, SOCK_STREAM, 0); int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in sa; memset(&sa, 0, sizeof sa); sa.sin_family = AF_INET; sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK); sa.sin_port = htons((uint16_t)port);
    if (bind(ls, (struct sockaddr*)&sa, sizeof sa) != 0 || listen(ls, world) != 0) { printf("FAIL bind/listen\n"); return 1; }
    socklen_t sl = sizeof sa; getsockname(ls, (struct sockaddr*)&sa, &sl); int p = ntohs(sa.sin_port);
    printf("# EP distribuito SOCKET: world=%d 127.0.0.1:%d\n", world, p);

    for (int r = 0; r < world; r++) {
        pid_t pid = fork();
        if (pid == 0) {
            float* part = calloc((size_t)NO, sizeof(float));
            for (int e = r; e < E; e += world)
                for (int k = start[e]; k < start[e+1]; k++) { int a = sorted[k], ti = a/K; float w = aw[a];
                    for (int oo = 0; oo < O; oo++) { float acc = 0; for (int h = 0; h < H; h++) acc += x[ti*H+h]*W[((size_t)e*H+h)*O+oo]; part[ti*O+oo] += w*acc; } }
            int fd = socket(AF_INET, SOCK_STREAM, 0);
            if (connect(fd, (struct sockaddr*)&sa, sizeof sa) != 0) _exit(2);
            send_all(fd, part, (size_t)NO*sizeof(float)); close(fd); _exit(0);
        }
        if (pid < 0) { printf("FAIL fork\n"); return 1; }
    }
    float* out = calloc((size_t)NO, sizeof(float));
    for (int r = 0; r < world; r++) { int fd = accept(ls, NULL, NULL); if (fd < 0) { printf("FAIL accept\n"); return 1; }
        float* part = malloc((size_t)NO*sizeof(float)); if (recv_all(fd, part, (size_t)NO*sizeof(float)) != 0) { printf("FAIL recv\n"); return 1; }
        for (int i = 0; i < NO; i++) out[i] += part[i]; free(part); close(fd); }
    close(ls);
    for (int r = 0; r < world; r++) wait(NULL);
    float md = 0; for (int i = 0; i < NO; i++) { float d = fabsf(out[i]-ref[i]); if (d > md) md = d; }
    if (md > 1e-4f) { printf("FAIL EP socket vs single-process: max|diff|=%.6f\n", md); return 1; }
    printf("PASS moe_ep_socket: EP su SOCKET (%d rank, all-reduce di rete) == single-process (max|diff|=%.2e)\n", world, md);
    return 0;
}
