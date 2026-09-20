"""Decode supplied question manifests and create fixed, score-independent small controls."""
import argparse,base64,gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def decode(name,metadata):
    raw=gzip.decompress(base64.b64decode((ROOT/'data'/f'{name}.jsonl.gz.b64').read_bytes()))
    assert hashlib.sha256(raw).hexdigest()==metadata[name]['decoded_sha256']
    return [json.loads(s) for s in raw.decode().splitlines()]
def write(path,rows):
    text=''.join(json.dumps(r)+'\n' for r in rows)
    if path.exists():
        if path.read_text()!=text:raise RuntimeError(f'Refusing to replace a different manifest: {path}')
    else:path.write_text(text)
    print(path.name,len(rows),len({r['image'] for r in rows}),'images')
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'manifests');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);meta=json.loads((ROOT/'data/manifest_metadata.json').read_text())
    rows=decode('queries',meta);train=[r for r in rows if r['split'] in ['rand','pop']]
    original=[r for r in rows if r['split']=='adv'][:250];train_images={r['image'] for r in train}
    write(a.out/'train_pool.jsonl',train);write(a.out/'original_250.jsonl',original)
    write(a.out/'original_image_disjoint.jsonl',[r for r in original if r['image'] not in train_images])
    # A bounded training-setting control uses the historical pool with evaluation images removed.
    # This is an add-on protocol; do not label the old PEC adapter as trained on this filtered pool.
    eval_images={r['image'] for r in original}
    write(a.out/'matched_train.jsonl',[r for r in train if r['image'] not in eval_images])
    pope=decode('pope_subset',meta)
    for split in ['random','popular','adversarial']:
        write(a.out/f'pope_{split}_252.jsonl',[r for r in pope if r['split']==split])
    write(a.out/'pope_all_756.jsonl',pope)
    assert not ({r['image'] for r in pope}&train_images)
    assert not ({r['image'] for r in train if r['image'] not in eval_images}&eval_images)
if __name__=='__main__':main()
