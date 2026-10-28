#!/usr/bin/env python3
"""gen_dsv41_hcpost_official_ref.py — genera il gate `llm_dsv41_hcpost_official_test.c`:
confronta `llm_dsv41_hc_post` (llm_dsv41_math.h) con l'ESPRESSIONE UFFICIALE letterale di
DeepSeek-V4.1 `inference/model.py` Block.hc_post (5-D, sum axis=2). Serve a catturare la classe di
bug "matrice di mixing TRASPOSTA" (avvenuta in Cachalot) contro la FONTE AUTOREVOLE, non solo vs oMLX.
Uso: python3 tools/gen_dsv41_hcpost_official_ref.py > LLMEnv/LLMEngine/mlx/llm_dsv41_hcpost_official_test.c
"""
import numpy as np
rng = np.random.default_rng(20260923)
cases = [(4, 8, 3), (4, 5, 4), (2, 6, 3), (1, 4, 2)]   # (hc_mult, D, S)

def official(x, r, p, c):
    # ESPRESSIONE UFFICIALE (model.py Block.hc_post), invariata
    return p[..., None] * x[..., None, :] + np.sum(c[..., None] * r[..., None, :], axis=2)

lines = ['/* AUTOGENERATO da tools/gen_dsv41_hcpost_official_ref.py — NON editare a mano.',
         ' * Gate: `llm_dsv41_hc_post` == espressione UFFICIALE di inference/model.py (Block.hc_post).',
         ' * Verifica contro la FONTE AUTOREVOLE (non solo oMLX): cattura la classe "mixing trasposto". */',
         '#include <stdio.h>', '#include <math.h>', '#include "llm_dsv41_math.h"', '']
body = []
for ci, (n, D, S) in enumerate(cases):
    x = rng.standard_normal((S, D)).astype(np.float32)
    r = rng.standard_normal((S, n, D)).astype(np.float32)
    p = rng.standard_normal((S, n)).astype(np.float32)
    c = rng.standard_normal((S, n, n)).astype(np.float32)
    y = official(x[None], r[None], p[None], c[None])[0]
    def arr(a): return ', '.join(f'{v:.8g}f' for v in np.asarray(a).ravel())
    body.append(f'/* caso {ci}: hc_mult={n} D={D} S={S} */')
    body.append(f'static const float X{ci}[] = {{{arr(x)}}};')
    body.append(f'static const float R{ci}[] = {{{arr(r)}}};')
    body.append(f'static const float P{ci}[] = {{{arr(p)}}};')
    body.append(f'static const float C{ci}[] = {{{arr(c)}}};')
    body.append(f'static const float Y{ci}[] = {{{arr(y)}}};')
    body.append(f'static const int N{ci} = {n}, D{ci} = {D}, S{ci} = {S};')
cases_arr = ', '.join('{X%d,R%d,P%d,C%d,Y%d,N%d,D%d,S%d}' % ((i,)*8) for i in range(len(cases)))
body.append(f'static const struct {{ const float *x,*res,*post,*comb,*exp; int n,D,S; }} CASES[] = {{ {cases_arr} }};')
body.append('int main(void){ int bad=0;')
body.append('  for (int i=0;i<(int)(sizeof(CASES)/sizeof(CASES[0]));i++){')
body.append('    const float *x=CASES[i].x,*res=CASES[i].res,*po=CASES[i].post,*co=CASES[i].comb,*ex=CASES[i].exp;')
body.append('    int n=CASES[i].n,D=CASES[i].D,S=CASES[i].S; float out[256]; int b=0;')
body.append('    for(int t=0;t<S;t++){ llm_dsv41_hc_post(x+t*D,D,res+t*n*D,po+t*n,co+t*n*n,n,out);')
body.append('      for(int k=0;k<n*D;k++) if(fabsf(out[k]-ex[t*n*D+k])>3e-5f){ if(b<2) fprintf(stderr,"  case%d t%d k%d got=%.7g exp=%.7g\\n",i,t,k,out[k],ex[t*n*D+k]); b++; } }')
body.append('    printf("%s hc_post case%d (hc_mult=%d,D=%d,S=%d)\\n", b?"FAIL":"PASS", i,n,D,S); bad+=b; }')
body.append('  printf(bad?"FAIL (%d)\\n":"ALL PASS\\n", bad); return bad?1:0; }')
open('/dev/stdout','w').write('\n'.join(lines + body) + '\n')
