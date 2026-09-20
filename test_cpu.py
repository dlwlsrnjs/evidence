"""CPU-only manifest, statistics, CLI and non-overwrite checks (no pretrained model needed)."""
import json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from analysis import SIG,cluster_ci
from experiment import parser,write_new
from report import comparison
ROOT=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(ROOT/'prepare_data.py')],check=True)
read=lambda name:[json.loads(s) for s in (ROOT/'manifests'/name).read_text().splitlines()]
train=read('matched_train.jsonl');ev=read('original_250.jsonl');pool=read('train_pool.jsonl');pope=read('pope_random_252.jsonl')
assert len(ev)==250 and len(pope)==252
assert not ({r['image'] for r in train}&{r['image'] for r in ev})
assert not ({r['image'] for r in pool}&{r['image'] for r in pope})
assert sum(r['label']=='yes' for r in pope)==126
f=np.array([1.,9.]);b=np.array([0.,10.])
assert (SIG(f)-SIG(b)).mean()>0 and (SIG(f-10)-SIG(b-10)).mean()<0
for a in [.1,1.,10.]:
 for c in [-10.,0.,10.]:assert np.array_equal(a*f+c>a*b+c,f>b)
assert cluster_ci([1.,1.],['a','b'],100)==[1.,1.]
for method in ['pec','dpo','ipo','nats']:
 a=parser().parse_args(['train','--method',method,'--model','dummy','--data','x','--out','y'])
 assert a.steps==200 and a.lr==3e-5
with tempfile.TemporaryDirectory() as d:
 d=Path(d);base=d/'base.jsonl';target=d/'target.jsonl'
 x=[{'image':'a','query':'q','label':'no','split':'random','d_full':-1.,'d_gray':0.}]
 y=[dict(x[0],d_full=1.)]
 for p,rr in [(base,x),(target,y)]:
  p.write_text(json.dumps(rr[0])+'\n');Path(str(p)+'.complete.json').write_text('{}')
 z=comparison(base,target)['random:no:d_gray'];assert z['candidate_minus_base_direction']==1.
 subprocess.run([sys.executable,str(ROOT/'experiment.py'),'summarize','--data',str(target),'--out',str(d/'summary.json')],check=True,stdout=subprocess.DEVNULL)
 try:write_new(d/'summary.json',{})
 except FileExistsError:pass
 else:raise AssertionError('Existing results must be preserved')
print('PASS: manifest integrity/disjointness, affine-sign check, cluster CI, matched defaults, paired comparison, summary and overwrite guard. GPU execution is a separate check.')
