"""Aggregate completed evaluations; optional paired candidate-minus-baseline comparisons."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from analysis import SIG,cluster_ci

def records(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def completed(path):
    if not Path(str(path)+'.complete.json').exists():raise ValueError(f'Incomplete/unverified evaluation: {path}')
def key(r):return (r['image'],r['query'],r['label'],r.get('split','all'))
def comparison(base,candidate):
    completed(base);completed(candidate);b={key(r):r for r in records(base)};rr=records(candidate)
    if len(b)!=len(rr) or set(b)!=set(map(key,rr)):raise ValueError('Runs must contain exactly the same unique query keys')
    out={}
    for split in sorted({r.get('split','all') for r in rr}):
        for label in ['yes','no']:
            rows=[r for r in rr if r.get('split','all')==split and r['label']==label]
            if not rows:continue
            images=[r['image'] for r in rows]
            for e in [k for k in rows[0] if k.startswith('d_') and k!='d_full']:
                if any(e not in b[key(r)] for r in rows):raise ValueError('Endpoints differ')
                dp=[];direction=[]
                for r in rows:
                    z=b[key(r)]
                    dp.append((SIG(r['d_full'])-SIG(r[e]))-(SIG(z['d_full'])-SIG(z[e])))
                    direction.append(int(r['d_full']>r[e])-int(z['d_full']>z[e]))
                out[f'{split}:{label}:{e}']={'n':len(rows),'images':len(set(images)),
                    'candidate_minus_base_dp':float(np.mean(dp)),'dp_change_ci':cluster_ci(dp,images),
                    'candidate_minus_base_direction':float(np.mean(direction)),'direction_change_ci':cluster_ci(direction,images)}
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('results'))
    p.add_argument('--baseline',type=Path);p.add_argument('--candidate',type=Path);a=p.parse_args()
    if bool(a.baseline)!=bool(a.candidate):p.error('Provide both --baseline and --candidate')
    if a.baseline:
        print(json.dumps(comparison(a.baseline,a.candidate),indent=2));return
    table=[]
    for raw in sorted(a.results.glob('*.jsonl')):
        if not Path(str(raw)+'.complete.json').exists():continue
        rr=records(raw)
        for split in sorted({r.get('split','all') for r in rr}):
            for label in ['yes','no']:
                rows=[r for r in rr if r.get('split','all')==split and r['label']==label]
                if not rows:continue
                f=np.array([r['d_full'] for r in rows]);ims=[r['image'] for r in rows]
                for e in [k for k in rows[0] if k.startswith('d_') and k!='d_full']:
                    b=np.array([r[e] for r in rows]);dp=SIG(f)-SIG(b);lo,hi=cluster_ci(dp,ims)
                    table.append({'run':raw.stem,'split':split,'label':label,'endpoint':e[2:],'n':len(rows),'images':len(set(ims)),
                        'dp':float(dp.mean()),'ci_low':lo,'ci_high':hi,'positive_direction':float((f>b).mean()),
                        'class_accuracy':float((f>0).mean()),'class_abs_margin':float(abs(f).mean()),
                        'p_full':float(SIG(f).mean()),'p_endpoint':float(SIG(b).mean())})
    if not table:raise SystemExit('No completed evaluation files found')
    path=a.results/'report.csv'
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    md=['# Completed evaluation results','',
        '95% image-cluster percentile CIs; 5,000 resamples. Conditional on each checkpoint; not a training-seed CI. No multiple-comparison correction. Class means are not scale/bias invariant.','',
        '| Run | split | class | endpoint | n | DeltaP [95% CI] | positive direction |',
        '|---|---|---|---|---:|---|---:|']
    for r in table:md.append(f"| {r['run']} | {r['split']} | {r['label']} | {r['endpoint']} | {r['n']} | {r['dp']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['positive_direction']:.3f} |")
    (a.results/'report.md').write_text('\n'.join(md)+'\n');print(path);print(a.results/'report.md')
if __name__=='__main__':main()
