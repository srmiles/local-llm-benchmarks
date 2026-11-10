import json,glob,sys
D='/data/llm/benchmarks/20261031-5047748f1'
print(f"{'run/model':34} {'arm':6} {'dec_med':>8} {'dec_min':>8} {'acc%':>6} {'pp2k':>7} {'pp_max':>12} {'vram':>6}")
for f in sorted(glob.glob(D+'/*/*.json')):
    d=json.load(open(f)); r=f.split('/')[-2]; n=f.split('/')[-1][:-5]; m,arm=n.rsplit('-',1)
    pf=d.get('prefill',{}); big=max(pf,key=lambda k:int(k)) if pf else None
    if 'decode' not in d: print(f"{r+'/'+m:34} {arm:6} NO DECODE (run aborted)"); continue
    print(f"{r+'/'+m:34} {arm:6} {d['decode'].get('median',0):8.1f} {d['decode'].get('min',0):8.1f} {d.get('spec',{}).get('acceptance_pct',0):6.1f} {pf.get('2000',{}).get('tps_median',0):7.0f} {str(big)+':'+str(round(pf[big]['tps_median'])) if big else '-':>12} {d.get('peak_vram_gib',0):6.2f}")
