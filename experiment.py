"""Rebuttal experiments with immutable manifests, shared training schedules and raw scores.
CPU commands: prepare, check, summarize. GPU commands: train, evaluate.
Existing research scripts/checkpoints are never modified. See EXPERIMENTS.md.
"""
import argparse, collections, hashlib, json, math, os, random, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE

def read(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def key(r):return (r['image'],r['query'],r['label'])
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_new(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,indent=2)
def manifest(p,rows):
    with Path(p).open('x') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
def find_image(fn,dirs):
    p=Path(fn)
    if p.is_absolute() and p.is_file():return p
    for root in dirs:
        for sub in ['','coco','nocaps','vizwiz','amber']:
            q=Path(root)/sub/fn
            if q.is_file():return q
    raise FileNotFoundError(fn)
def check(a):
    rows=read(a.data);missing=[]
    for im in sorted({r['image'] for r in rows}):
        try:find_image(im,a.images)
        except FileNotFoundError:missing.append(im)
    print(json.dumps({'rows':len(rows),'images':len({r['image'] for r in rows}),'missing_count':len(missing),'missing_first20':missing[:20],'sha256':digest(a.data)},indent=2))
    if missing:raise SystemExit(2)
PINNED_REVISIONS={
 'unsloth/llava-1.5-7b-hf':'b34271dd19f43ed5f03ac2fa29cfbe50c798bd29',
 'unsloth/Qwen2.5-VL-7B-Instruct':'cbb81dca0b8c1a0c6827913deda2709388acadde'}

def load_runtime(model_path,train=False,seed=0,engine='unsloth',cpu_offload=False,device='cuda:0'):
    os.environ.setdefault('USE_TF','0')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    if engine=='hf':
        if train:raise ValueError('HF backend is for inference only; train with unsloth on a sufficiently large GPU')
        import torch
        from transformers import AutoModelForImageTextToText,AutoProcessor
        from peft import PeftModel
        adapter=Path(model_path)/'adapter_config.json'
        base=json.loads(adapter.read_text())['base_model_name_or_path'] if adapter.is_file() else model_path
        revision=PINNED_REVISIONS.get(base)
        m=AutoModelForImageTextToText.from_pretrained(base,revision=revision,dtype=torch.bfloat16,
              device_map='cpu' if cpu_offload else device,attn_implementation='sdpa')
        if adapter.is_file():m=PeftModel.from_pretrained(m,model_path,is_trainable=False)
        p=AutoProcessor.from_pretrained(model_path if adapter.is_file() else base,
               **({} if adapter.is_file() else {'revision':revision}))
        m.eval()
        if cpu_offload:
            from accelerate import cpu_offload as enable_offload
            enable_offload(m,execution_device=torch.device(device),offload_buffers=True)
        m._score_device=torch.device(device)
    else:
        from unsloth import FastVisionModel
        import torch
        random.seed(seed);torch.manual_seed(seed)
        m,p=FastVisionModel.from_pretrained(model_path,load_in_4bit=False,dtype=torch.bfloat16)
        if train:
            m=FastVisionModel.get_peft_model(m,r=16,lora_alpha=32,lora_dropout=0.05,random_state=seed,
                 finetune_vision_layers=False,finetune_language_layers=True,finetune_attention_modules=True,finetune_mlp_modules=False)
            FastVisionModel.for_training(m);m.train()
        else:FastVisionModel.for_inference(m);m.eval()
    if 'llava' in str(m.config.model_type).lower():
        p.patch_size=14;p.vision_feature_select_strategy='default';p.num_additional_image_tokens=1
    m.config.use_cache=False
    return m,p,torch

def score(m,p,im,q,ans,torch):
    msg=[{'role':'user','content':[{'type':'image'},{'type':'text','text':q}]}]
    prompt=p.apply_chat_template(msg,add_generation_prompt=True)
    enc=p(images=[im],text=prompt+' '+ans,return_tensors='pt').to(getattr(m,'_score_device',m.device))
    plen=p(images=[im],text=prompt,return_tensors='pt')['input_ids'].shape[1]
    # Same answer-only, summed token likelihood convention as submitted code.
    out=m(**enc,use_cache=False); logits=out.logits[:,:-1];target=enc['input_ids'][:,1:]
    # Slice answer positions before float softmax; same summed likelihood, lower workspace memory.
    logits=logits[:,plen-1:];target=target[:,plen-1:]
    return torch.log_softmax(logits.float(),-1).gather(-1,target.unsqueeze(-1)).sum()

def train(a):
    total_started=time.monotonic()
    if a.out.exists():raise FileExistsError(a.out)
    rows=read(a.data)
    if a.held_out:
        held=read(a.held_out)
        if {r['image'] for r in rows}&{r['image'] for r in held}:raise ValueError('Train/test image overlap')
    # Validate all images before loading a model.
    paths={r['image']:find_image(r['image'],a.images) for r in rows}
    a.out.mkdir(parents=True);config=vars(a).copy();config.pop('func')
    config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}
    config['data_sha256']=digest(a.data);config['implementation_sha256']=digest(__file__)
    if a.held_out:config['held_out_sha256']=digest(a.held_out)
    rng=random.Random(a.seed);indices=[rng.randrange(len(rows)) for _ in range(a.steps)]
    write_new(a.out/'config.json',config)
    write_new(a.out/'training_trace.json',{'indices':indices,'queries':[key(rows[i]) for i in indices]})
    m,p,torch=load_runtime(a.model,True,a.seed)
    import torch.nn.functional as F
    from PIL import Image
    opt=torch.optim.AdamW([x for x in m.parameters() if x.requires_grad],lr=a.lr)
    gray=Image.new('RGB',(448,448),(128,128,128));cache={}
    def im(r):
        if r['image'] not in cache:
            x=Image.open(paths[r['image']]).convert('RGB');x.thumbnail((512,512));cache[r['image']]=x
        return cache[r['image']]
    # IPO and DPO use a deterministic frozen reference (dropout disabled), cached before training.
    refs={};reference_started=time.monotonic()
    if a.method in ['dpo','ipo']:
        m.eval()
        with torch.no_grad(),m.disable_adapter():
            for i in sorted(set(indices)):
                r=rows[i];g,o=('Yes.','No.') if r['label']=='yes' else ('No.','Yes.')
                refs[i]=float(score(m,p,im(r),r['query'],g,torch)-score(m,p,im(r),r['query'],o,torch))
        m.train()
        write_new(a.out/'reference_margins.json',refs)
    reference_seconds=time.monotonic()-reference_started
    started=time.monotonic()
    with (a.out/'losses.jsonl').open('x') as log:
        for step,i in enumerate(indices,1):
            r=rows[i];g,o=('Yes.','No.') if r['label']=='yes' else ('No.','Yes.')
            opt.zero_grad();d=score(m,p,im(r),r['query'],g,torch)-score(m,p,im(r),r['query'],o,torch)
            if a.method=='dpo':loss=-F.logsigmoid(a.beta*(d-refs[i]))
            elif a.method=='ipo':loss=(d-refs[i]-1/(2*a.beta))**2
            elif a.method=='supervised':loss=F.softplus(-d)
            else:
                b=score(m,p,gray,r['query'],g,torch)-score(m,p,gray,r['query'],o,torch)
                gap=torch.sigmoid(d)-torch.sigmoid(b) if a.method=='pec' else d-b
                contrast=(gap-a.delta)**2 if a.method=='nats_sq' else F.softplus(a.delta-gap)
                loss=contrast+a.lam*b.abs()
            loss.backward();opt.step()
            log.write(json.dumps({'step':step,'row_index':i,'loss':float(loss.detach()),'seconds':time.monotonic()-started})+'\n');log.flush()
            if step%10==0:print(f'{a.method} step {step}/{a.steps} loss={float(loss):.5f}',flush=True)
    m.save_pretrained(str(a.out/'adapter'));p.save_pretrained(str(a.out/'adapter'))
    write_new(a.out/'complete.json',{'steps':a.steps,'training_seconds':time.monotonic()-started,'reference_seconds':reference_seconds,'total_wall_seconds':time.monotonic()-total_started,
                                   'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated()})

def evaluate(a):
    eval_started=time.monotonic()
    if a.out.exists():raise FileExistsError(a.out)
    rows=read(a.data)
    # Preserve all strata; --limit is only for smoke tests, never final reporting.
    if a.limit:rows=rows[:a.limit]
    paths={r['image']:find_image(r['image'],a.images) for r in rows}
    if 'other' in a.endpoints and len(paths)<2:raise ValueError('At least two distinct images required')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    write_new(str(a.out)+'.metadata.json',{'model':a.model,'data':str(a.data),'sha256':digest(a.data),'endpoints':a.endpoints,
        'other_repeats':a.other_repeats,'seed':a.seed,'n':len(rows),'engine':a.engine,'cpu_offload':a.cpu_offload,'status':'started','implementation_sha256':digest(__file__)})
    m,p,torch=load_runtime(a.model,engine=a.engine,cpu_offload=a.cpu_offload,device=a.device)
    from PIL import Image
    import numpy as np
    cache={};names=sorted(paths)
    def im(name):
        if name not in cache:
            x=Image.open(paths[name]).convert('RGB');x.thumbnail((512,512));cache[name]=x
        return cache[name]
    with torch.no_grad(),a.out.open('x') as out:
        for i,r in enumerate(rows):
            x=im(r['image']);sd=int(hashlib.sha256((str(a.seed)+str(key(r))).encode()).hexdigest()[:8],16)
            ims={'full':x};donors={}
            for e in a.endpoints:
                if e=='gray':ims[e]=Image.new('RGB',x.size,(128,128,128))
                elif e=='gray448':ims[e]=Image.new('RGB',(448,448),(128,128,128))
                elif e=='black':ims[e]=Image.new('RGB',x.size,(0,0,0))
                elif e=='noise':ims[e]=Image.fromarray(np.random.default_rng(sd).integers(0,256,(x.height,x.width,3),dtype=np.uint8))
                elif e=='other':
                    candidates=[n for n in names if n!=r['image']];g=random.Random(sd)
                    for j in range(a.other_repeats):
                        n=g.choice(candidates);donors[f'other{j}']=n;ims[f'other{j}']=im(n).resize(x.size)
            z=dict(r);z['donors']=donors;direction=1 if r['label']=='yes' else -1
            for name,img in ims.items():
                z['d_'+name]=direction*float(score(m,p,img,r['query'],'Yes.',torch)-score(m,p,img,r['query'],'No.',torch))
            out.write(json.dumps(z)+'\n');out.flush()
            if (i+1)%10==0 or i==0:print(f'{i+1}/{len(rows)}',flush=True)
    write_new(str(a.out)+'.complete.json',{'n':len(rows),'sha256':digest(a.out),'total_wall_seconds':time.monotonic()-eval_started})

def summarize(a):
    import numpy as np
    sys.path.insert(0,str(HERE));from analysis import cluster_ci,SIG
    rows=read(a.data)
    endpoints=[k for k in rows[0] if k.startswith('d_') and k!='d_full']
    report={}
    for split in sorted({r.get('split','all') for r in rows}):
        for label in ['yes','no']:
            rr=[r for r in rows if r.get('split','all')==split and r['label']==label]
            if not rr:continue
            f=np.array([r['d_full'] for r in rr]);images=[r['image'] for r in rr];z={}
            for e in endpoints:
                b=np.array([r[e] for r in rr]);dp=SIG(f)-SIG(b)
                z[e]={'dp':float(dp.mean()),'dp_ci':cluster_ci(dp,images),'positive_direction':float((f>b).mean()),
                      'positive_direction_ci':cluster_ci(f>b,images),'p_endpoint':float(SIG(b).mean()),
                      'endpoint_distance_half':float(abs(SIG(b)-.5).mean())}
            report[split+':'+label]={'n':len(rr),'acc':float((f>0).mean()),'margin':float(abs(f).mean()),'p_full':float(SIG(f).mean()),'endpoints':z}
    write_new(a.out,report);print(json.dumps(report,indent=2))

def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    default_images=[os.environ.get('PEC_IMAGES',str(HERE/'images')),os.environ.get('POPE_IMAGES',str(HERE/'pope_images'))]
    for name,func in [('check',check),('train',train),('evaluate',evaluate),('summarize',summarize)]:
        s=sub.add_parser(name);s.add_argument('--data',type=Path,required=True);s.set_defaults(func=func)
        if name!='summarize':s.add_argument('--images',nargs='+',default=default_images)
        if name in ['train','evaluate']:
            s.add_argument('--model',required=True);s.add_argument('--out',type=Path,required=True);s.add_argument('--seed',type=int,default=0)
        if name=='train':
            s.add_argument('--method',choices=['pec','nats','nats_sq','dpo','ipo','supervised'],required=True)
            s.add_argument('--steps',type=int,default=200);s.add_argument('--lr',type=float,default=3e-5)
            s.add_argument('--beta',type=float,default=.1);s.add_argument('--delta',type=float,default=.6);s.add_argument('--lam',type=float,default=.1)
            s.add_argument('--held-out',type=Path)
        if name=='evaluate':
            s.add_argument('--engine',choices=['unsloth','hf'],default='unsloth')
            s.add_argument('--cpu-offload',action='store_true',help='HF inference only; stream BF16 weights from CPU without quantization')
            s.add_argument('--device',default='cuda:0')
            s.add_argument('--endpoints',nargs='+',choices=['gray','gray448','black','noise','other'],default=['gray','black','noise','other'])
            s.add_argument('--other-repeats',type=int,default=3);s.add_argument('--limit',type=int,default=0)
        if name=='summarize':s.add_argument('--out',type=Path,required=True)
    return p
if __name__=='__main__':
    a=parser().parse_args()
    if getattr(a,'cpu_offload',False) and a.engine!='hf':raise ValueError('--cpu-offload requires --engine hf')
    a.func(a)
