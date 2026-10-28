#!/usr/bin/env python3
"""gen_dsv41_compool_official_ref.py — gate: `llm_dsv41_compressor_pool` == espressione UFFICIALE di
DeepSeek-V4.1 `inference/model.py` (Compressor.__call__: (kv * score.softmax(dim=2)).sum(dim=2),
softmax ATTRAVERSO il gruppo, un peso per canale). Cattura l'errore "softmax per-canale" (sbagliato).
Uso: python3 tools/gen_dsv41_compool_official_ref.py > LLMEnv/LLMEngine/mlx/llm_dsv41_compool_official_test.c
"""
import numpy as np
rng = np.random.default_rng(20260923)
cases = [(4, 6, 3), (2, 8, 4), (8, 4, 2)]   # (ratio, H, m)
out = ['/* AUTOGENERATO da tools/gen_dsv41_compool_official_ref.py — NON editare a mano. */',
       '#include <stdio.h>', '#include <math.h>', '#include "llm_dsv41_math.h"', '']
for ci, (r, H, m) in enumerate(cases):
    kv = rng.standard_normal((m, r, H)).astype(np.float32)
    g = rng.standard_normal((m, r, H)).astype(np.float32)
    mx = g.max(axis=1, keepdims=True); e = np.exp(g - mx); w = e / e.sum(axis=1, keepdims=True)
    pooled = (kv * w).sum(axis=1)     # espressione ufficiale
    A = lambda a: ', '.join(f'{v:.8g}f' for v in np.asarray(a).ravel())
    out += [f'static const float KV{ci}[] = {{{A(kv)}}};',
            f'static const float G{ci}[] = {{{A(g)}}};',
            f'static const float Y{ci}[] = {{{A(pooled)}}};',
            f'static const int R{ci} = {r}, H{ci} = {H}, M{ci} = {m};']
arr = ', '.join('{KV%d,G%d,Y%d,R%d,H%d,M%d}' % ((i,)*6) for i in range(len(cases)))
out += [f'static const struct {{ const float *kv,*g,*y; int r,H,m; }} C[] = {{ {arr} }};',
        'int main(void){ int bad=0;',
        '  for(int i=0;i<(int)(sizeof(C)/sizeof(C[0]));i++){',
        '    float out[64];',
        '    for(int j=0;j<C[i].m;j++){ llm_dsv41_compressor_pool(C[i].kv+j*C[i].r*C[i].H, C[i].g+j*C[i].r*C[i].H, C[i].r, C[i].H, 1, out);',
        '      for(int d=0;d<C[i].H;d++) if(fabsf(out[d]-C[i].y[j*C[i].H+d])>3e-5f){ if(bad<3) fprintf(stderr,"  case%d j%d d%d got=%.7g exp=%.7g\\n",i,j,d,out[d],C[i].y[j*C[i].H+d]); bad++; } }',
        '    printf("%s compool case%d (ratio=%d,H=%d,m=%d)\\n", bad?"FAIL":"PASS", i, C[i].r, C[i].H, C[i].m); }',
        '  printf(bad?"FAIL (%d)\\n":"ALL PASS\\n", bad); return bad?1:0; }']
open('/dev/stdout','w').write('\n'.join(out) + '\n')
