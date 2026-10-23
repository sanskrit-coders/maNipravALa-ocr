#!/usr/bin/env python3
import argparse,json,math,pathlib,random,sys,time
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from PIL import Image,ImageDraw,ImageFont,ImageFilter
IN=pathlib.Path('/kaggle/input')
def find(n):
 h=sorted(IN.rglob(n)); return h[0] if h else None
jsonl=find('train.jsonl'); fonts=find('fonts'); fontD=str(fonts/'NotoSansDevanagari-Regular.ttf'); fontT=str(fonts/'NotoSansTamil-Regular.ttf')
words=[]
for l in jsonl.read_text().splitlines():
 r=json.loads(l)
 if r.get('label')=='mixed' and r.get('n_runs')==2 and [x['script'] for x in r['runs']]==['dev','tam']:
  words.append((r['runs'][0]['text'],r['runs'][1]['text']))
print('words',len(words),flush=True)
H,W=64,256
cache={}
def font(path,size):
 k=(path,size)
 if k not in cache: cache[k]=ImageFont.truetype(path,size)
 return cache[k]
def render(d,t,rng):
 fs=rng.randint(28,48); fd=font(fontD,fs); ft=font(fontT,fs)
 gap=rng.choice([0,1,2,3])
 tmp=Image.new('L',(2400,160),255); dr=ImageDraw.Draw(tmp); base=120
 wd=dr.textlength(d,font=fd); wt=dr.textlength(t,font=ft)
 dr.text((20,base),d,font=fd,fill=0,anchor='ls')
 dr.text((20+wd+gap,base),t,font=ft,fill=0,anchor='ls')
 bbox=tmp.getbbox(); crop=tmp.crop(bbox); cw,ch=crop.size
 th=rng.randint(24,48); sc=min(th/ch,(W-12)/max(1,cw)); nw=max(1,int(cw*sc)); nh=max(1,int(ch*sc))
 crop=crop.resize((nw,nh),Image.BILINEAR)
 im=Image.new('L',(W,H),255); x0=rng.randint(0,max(0,W-nw)); y0=rng.randint(0,max(0,H-nh)); im.paste(crop,(x0,y0))
 b=(x0+wd*sc)/W
 return np.asarray(im,np.uint8),max(.02,min(.98,b))
class DS(torch.utils.data.Dataset):
 def __init__(s,items,seed): s.items=items;s.seed=seed
 def __len__(s): return len(s.items)
 def __getitem__(s,i):
  rng=random.Random((s.seed<<20)+i); im,b=render(*s.items[i],rng); return torch.from_numpy(1-im.astype(np.float32)/255).unsqueeze(0),int(round(b*(W-1)))
def aug(x):
 if random.random()<.5: x=(x*random.uniform(.75,1.2)+random.uniform(-.1,.1)).clamp(0,1)
 if random.random()<.35: x=(x+torch.randn_like(x)*random.uniform(.005,.03)).clamp(0,1)
 return x
class Net(nn.Module):
 def __init__(s):
  super().__init__()
  def b(a,c): return nn.Sequential(nn.Conv2d(a,c,3,1,1),nn.BatchNorm2d(c),nn.ReLU(),nn.Conv2d(c,c,3,1,1),nn.BatchNorm2d(c),nn.ReLU())
  s.e=nn.Sequential(b(1,32),nn.MaxPool2d((2,1)),b(32,64),nn.MaxPool2d((2,1)),b(64,128),nn.MaxPool2d((2,1)),b(128,128),nn.AdaptiveAvgPool2d((1,W)))
  s.o=nn.Sequential(nn.Conv1d(128,64,5,1,2),nn.ReLU(),nn.Conv1d(64,1,1))
 def forward(s,x): return s.o(s.e(x).squeeze(2)).squeeze(1)
def ev(net,items,dev):
 net.eval(); errs=[];ps=[];rs=[];ok5=[]
 for i in range(0,len(items),64):
  rng=random.Random(777+i); xs=[];ys=[]
  for d,t in items[i:i+64]:
   im,b=render(d,t,rng); xs.append(torch.from_numpy(1-im.astype(np.float32)/255).unsqueeze(0)); ys.append(int(round(b*(W-1))))
  with torch.no_grad(): p=net(torch.stack(xs).to(dev)).argmax(1).cpu().numpy()
  for q,t in zip(p,ys):
   errs.append(abs(q-t)/W); ok5.append(abs(q-t)<=round(.05*W))
   dev_gt=set(range(t+1));tam_gt=set(range(t+1,W));dev_p=set(range(q+1));tam_p=set(range(q+1,W))
   for g,o in ((dev_gt,dev_p),(tam_gt,tam_p)):
    ps.append(len(g&o)/max(1,len(o)));rs.append(len(g&o)/max(1,len(g)))
 net.train(); return float(np.mean(errs)),float(np.mean(ok5)),float(np.mean(ps)),float(np.mean(rs))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=2500); ap.add_argument('--batch',type=int,default=128); ap.add_argument('--out',default='/kaggle/working/render_boundary.pt'); a=ap.parse_args()
 rng=random.Random(5); order=list(range(len(words))); rng.shuffle(order); nv=max(1000,len(order)//10); vi=[words[i] for i in order[:nv]]; ti=[words[i] for i in order[nv:]]
 dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); net=Net().to(dev); opt=torch.optim.AdamW(net.parameters(),2e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=1e-5)
 dl=torch.utils.data.DataLoader(DS(ti,1),batch_size=a.batch,shuffle=True,num_workers=4,drop_last=True,persistent_workers=True)
 def cyc():
  while True:
   for b in dl: yield b
 it=cyc(); best=9; out=pathlib.Path(a.out); t0=time.time()
 for st in range(1,a.steps+1):
  x,y=next(it); x=torch.stack([aug(q) for q in x]); loss=F.cross_entropy(net(x.to(dev)),y.to(dev),label_smoothing=.02); opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();sch.step()
  if st%500==0 or st==a.steps:
   mae,p5,sp,sr=ev(net,vi,dev); print(f'step {st} loss {loss.item():.3f} mae {mae:.4f} p@5 {p5:.3f} segP {sp:.3f} segR {sr:.3f} {time.time()-t0:.0f}s',flush=True)
   if mae<best: best=mae; torch.save({'state_dict':net.state_dict(),'arch':'boundary_cnn','img_h':H,'img_w':W,'mae':mae,'p@5':p5},out)
 (out.parent/'render_boundary_metrics.json').write_text(json.dumps({'best_mae':best},indent=2)); print('saved',out)
if __name__=='__main__': main()
