#!/usr/bin/env python3
"""df2_xctrace_agg.py — aggrega una traccia xctrace (Metal System Trace) di un run DFlash2.

Perche': i profiler a `mlx_array_eval` del motore hanno dato attribuzioni false (il primo eval
flushsa lavoro pendente). Qui si legge la traccia GPU vero, senza alcun sync nel motore.

Uso:
    xctrace record --template 'Metal System Trace' --output /tmp/df2.trace \
      --time-limit 100s --no-prompt \
      --env GMB_DF2_TARGET=<dir> --env GMB_DF2_DRAFT=<dir> \
      --launch -- /tmp/dflash2_bench 96
    python3 tools/df2_xctrace_agg.py /tmp/df2.trace [nome-processo]

Risponde a: (1) quanto tempo GPU e' occupata dal NOSTRO processo e quanto e' FERMA (-> se e'
CPU/launch-bound); (2) su quali canali (Compute/Vertex/...); (3) totale per processo.
"""
import sys, os, subprocess, collections, xml.etree.ElementTree as ET

def export(trace, schema, out):
    subprocess.run([
        'xctrace', 'export', '--input', trace,
        '--xpath', f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]',
        '--output', out], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out

def main():
    if len(sys.argv) < 2:
        print(__doc__); return 1
    trace = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else None
    xml = os.path.join('/tmp', 'df2_metal_intervals.xml')
    export(trace, 'metal-gpu-intervals', xml)
    per_proc = collections.Counter()
    per_proc_ch = collections.Counter()
    counts = collections.Counter()
    for _, el in ET.iterparse(xml, events=('end',)):
        if el.tag != 'row':
            continue
        ch = el.findtext('gpu-channel-name') or '(?)'
        d = el.findtext('duration')
        proc = None
        p = el.find('process')
        if p is not None:
            proc = p.findtext('pid') or p.text
        if d and d.strip().isdigit():
            dur = int(d)
            key = proc or '(?)'
            per_proc[key] += dur
            per_proc_ch[(key, ch)] += dur
            counts[key] += 1
        el.clear()
    if not per_proc:
        print('nessun intervallo GPU nella traccia'); return 1
    print('=== tempo GPU per processo (pid) ===')
    for k, v in per_proc.most_common(10):
        print('  %10.1f ms  x%-8d pid=%s%s' % (v / 1e6, counts[k], k,
              '   <-- NOSTRO' if (want and want in str(k)) else ''))
    if want:
        print('\n=== canali del processo %s ===' % want)
        for (k, ch), v in per_proc_ch.most_common(12):
            if want in str(k):
                print('  %10.1f ms  %s' % (v / 1e6, ch))
    return 0

def per_op(trace):
    """Aggrega il tempo GPU per NOME di operazione Metal (metal-application-encoders-list).
    E' la metrica per SCEGLIERE le fusioni: dice quali op costano GPU (qui: Broadcast/Add/
    elementwise a ~0,2-0,3 ms l'una, non i GEMM)."""
    import re
    xml = os.path.join('/tmp', 'df2_encoders.xml')
    export(trace, 'metal-application-encoders-list', xml)
    root = ET.parse(xml).getroot()
    ids = {}
    for el in root.iter():
        if el.tag == 'metal-object-label':
            i, t = el.get('id'), ''.join(el.itertext()).strip()
            if i and t:
                ids[i] = t
    a, c, tot = collections.Counter(), collections.Counter(), 0
    for row in root.iter('row'):
        d = row.findtext('duration')
        if not (d and d.strip().isdigit()):
            continue
        u = int(d); tot += u
        lab = None
        for l in row.findall('metal-object-label'):
            r = l.get('ref')
            t = ids.get(r) if r else ''.join(l.itertext()).strip()
            if t and not re.match(r'^\[\d+\]', t):
                lab = t; break
        a[lab or '(none)'] += u; c[lab or '(none)'] += 1
    print('=== tempo GPU per operazione Metal (tot %.1f ms) ===' % (tot / 1e6))
    for k, v in a.most_common(25):
        print('%10.1f ms  x%-7d %s' % (v / 1e6, c[k], k[:70]))
    return 0

if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[2] == '--per-op':
        sys.exit(per_op(sys.argv[1]))
    sys.exit(main())
