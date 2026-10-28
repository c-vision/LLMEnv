import math
_MASK64=(1<<64)-1; G=0x9E3779B97F4A7C15; M1=0xBF58476D1CE4E5B9; M2=0x94D049BB133111EB; P1=10007
def sm64(v):
    v=(v+G)&_MASK64; v=((v^(v>>30))*M1)&_MASK64; v=((v^(v>>27))*M2)&_MASK64; return (v^(v>>31))&_MASK64
def build_mult(vocab,ngram,li,seed):
    mmax=((1<<63)-1)//max(vocab,1); hb=max(1,mmax//2); bs=seed+P1*li
    return [2*(sm64((bs+G*(i+1))&_MASK64)%hb)+1 for i in range(ngram)]
def isprime(v):
    if v<2: return False
    if v%2==0: return v==2
    for d in range(3,math.isqrt(v)+1,2):
        if v%d==0: return False
    return True
def nth_prime(start,count):
    p=start
    for _ in range(count):
        p+=1
        while not isprime(p): p+=1
    return p
# --- parametri piccoli ---
VOCAB=64; NG=3; HPN=2; LI=0; SEED=42; BASE=1000; EOS=1
NH=(NG-1)*HPN
sizes=[nth_prime(BASE-1, LI*NH+h+1) for h in range(NH)]
offs=[]; t=0
for s in sizes: offs.append(t); t+=s
mult=build_mult(VOCAB,NG,LI,SEED)
def shift_right_ignore_eos(hist,shift):
    n=len(hist); eospos=[i if t==EOS else -1 for i,t in enumerate(hist)]
    pe=[]; m=-1
    for x in eospos: m=max(m,x); pe.append(m)
    prev=[-1]+pe[:-1]
    out=[]
    for i in range(n):
        seg=prev[i]+1; pin=i-seg; src=i-shift
        if pin>=shift and src>=0: out.append(hist[src])
        else: out.append(EOS)
    return out
hist=[5,6,7,8,9]
sh=[shift_right_ignore_eos(hist,s) for s in range(NG)]
per=[]
for ng in range(2,NG+1):
    st=(ng-2)*HPN; en=st+HPN
    mixed=[sh[0][i]*mult[0] for i in range(len(hist))]
    for pos in range(1,ng):
        mixed=[mixed[i]^(sh[pos][i]*mult[pos]) for i in range(len(hist))]
    per.append([[ (mixed[i] % sizes[h]) + offs[h] for h in range(st,en) ] for i in range(len(hist))])
full=[[v for blk in per for v in blk[i]] for i in range(len(hist))]
IL=3
print("MULT", mult)
print("SIZES", sizes)
print("OFFS", offs)
print("HIST", hist)
print("IDS", [v for i in range(len(hist)-IL,len(hist)) for v in full[i]])
