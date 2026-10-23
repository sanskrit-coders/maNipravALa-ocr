#!/usr/bin/env python3
import argparse,json,math,pathlib,random,sys,time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset,DataLoader
IN=pathlib.Path('/kaggle/input')
def find(n):
 h=sorted(IN.rglob(n)); return h[0] if h else None
code=find('train_classifier.py'); sys.path.insert(0,str(code.parent)); import train_classifier as T
fonts=find('fonts'); texts=find('texts')
if fonts:
 T.DEV_FONTS=T.localize_fonts(T.DEV_FONTS,str(fonts)); T.TAM_FONTS=T.localize_fonts(T.TAM_FONTS,str(fonts)); T.OTHER_FONTS=T.localize_fonts(T.OTHER_FONTS,str(fonts))
words=T.load_words(str(texts)); print('words',len(words),flush=True)
CLASSES=['dev','tam','other']; C={c:i for i,c in enumerate(CLASSES)}; H,W=64,256; M=8

def crop(im):
 a=np.asarray(im); cs=np.where((a<200).any(0))[0]; rs=np.where((a<200).any(1))[0]
 if not len(cs): return None
 return im.crop((max(0,int(cs[0])-M),max(0,int(rs[0])-M),min(a.shape[1],int(cs[-1])+M+1),min(a.shape[0],int(rs[-1])+M+1)))
def fit(im,rng):
 im=crop(im)
 if im is None: return None
 w,h=im.size; nw=max(1,int(w*H/h)); im=im.resize((nw,H),Image.BILINEAR); c=Image.new('L',(W,H),255); c.paste(im,(rng.randint(0,max(0,W-nw)),0)); return np.asarray(c,np.uint8)
def tgt(runs):
 t=np.full(W,C['other'],np.int64)
 for r in runs:
  a=int(round(float(r.get('x0_frac',0))*W)); b=int(round(float(r.get('x1_frac',1))*W)); t[max(0,a):min(W,b)]=C.get(r['script'],2)
 return t
def lab(p):
 s={x for x in p if x!='other'}; return next(iter(s)) if len(s)==1 else None
def sw(word,prof,seed,rng):
 r=T.render_word((word,prof,seed)); return fit(Image.fromarray(r['img']),rng) if r else None
def so(seed,rng):
 r=T.render_other(seed); return fit(Image.fromarray(r['img']),rng) if r else None
def synth(rng,wl):
 k=rng.choices(['dev','tam','other','mixed'],[30,30,10,30])[0]
 if k=='other':
  im=so(rng.randrange(1<<30),rng); return (im,np.full(W,C['other'])) if im is not None else None
 if k in ('dev','tam'):
  p=[(w,q) for w,q in wl if lab(q)==k]
  if not p: return None
  w,q=rng.choice(p); im=sw(w,q,rng.randrange(1<<30),rng); return (im,np.full(W,C[k])) if im is not None else None
 ds=[(w,q) for w,q in wl if lab(q)=='dev']; ts=[(w,q) for w,q in wl if lab(q)=='tam']
 if not ds or not ts: return None
 a=sw(*rng.choice(ds),rng.randrange(1<<30),rng); b=sw(*rng.choice(ts),rng.randrange(1<<30),rng)
 if a is None or b is None: return None
 wa=int(rng.uniform(.35,.65)*W); c=np.full((H,W),255,np.uint8); c[:,:wa]=a[:,:wa]; c[:,wa:]=b[:,wa:]; t=np.full(W,C['dev']); t[wa:]=C['tam']; return c,t
def load_real():
 j=find('train.jsonl'); recs=[json.loads(l) for l in j.read_text('utf-8').splitlines() if l.strip()]; xs=[];ys=[];ps=[]
 for r in recs:
  try:
   rel=pathlib.Path(r['image']); r2=pathlib.Path(*rel.parts[1:]) if rel.parts and rel.parts[0]=='data' else rel
   impl=[j.parent/r2, j.parent.parent/r2, j.parent.parent.parent/r2, j.parent.parent/'data'/r2]
   img=next((p for p in impl if p.exists()), None)
   if img is None:
    hits=[p for p in IN.rglob(rel.name) if p.parent.name==rel.parent.name and p.parent.parent.name==rel.parent.parent.name]
    img=hits[0]
   im=Image.open(img).convert('L').crop(r['box']); im=crop(im)
   if im is None: continue
   xs.append(np.asarray(im.resize((W,H),Image.BILINEAR),np.uint8)); ys.append(tgt(r['runs'])); ps.append(rel.stem)
  except Exception: pass
 print('real',len(xs),flush=True); return np.stack(xs),np.stack(ys),np.array(ps)
class DS(Dataset):
 def __init__(s,x,y): s.x=x;s.y=y
 def __len__(s): return len(s.y)
 def __getitem__(s,i): return torch.from_numpy(1-s.x[i].astype(np.float32)/255).unsqueeze(0),torch.from_numpy(s.y[i]).long()
def aug(x):
 H,W=x.shape[-2:]
 if random.random()<.5: x=(x*random.uniform(.75,1.2)+random.uniform(-.1,.1)).clamp(0,1)
 if random.random()<.35: x=(x+torch.randn_like(x)*random.uniform(.005,.03)).clamp(0,1)
 return x
class Net(nn.Module):
 def __init__(s):
  super().__init__()
  def b(a,b): return nn.Sequential(nn.Conv2d(a,b,3,1,1),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,1,1),nn.BatchNorm2d(b),nn.ReLU())
  s.e=nn.Sequential(b(1,32),nn.MaxPool2d((2,1)),b(32,64),nn.MaxPool2d((2,1)),b(64,128),nn.MaxPool2d((2,1)),b(128,128),nn.AdaptiveAvgPool2d((1,W)))
  s.o=nn.Sequential(nn.Conv1d(128,64,5,1,2),nn.ReLU(),nn.Conv1d(64,3,1))
 def forward(s,x): return s.o(s.e(x).squeeze(2))
def ev(net,x,y,dev):
 net.eval(); cor=0;n=0;berr=[]
 with torch.no_grad():
  for i in range(0,len(x),128):
   p=net(torch.from_numpy(1-x[i:i+128].astype(np.float32)/255).unsqueeze(1).to(dev)).argmax(1).cpu().numpy(); t=y[i:i+128]; cor+=(p==t).sum(); n+=t.size
   for q,z in zip(p,t):
    cz=np.where(np.diff(z)!=0)[0]; cp=np.where(np.diff(q)!=0)[0]
    if len(cz)==1 and len(cp): berr.append(min(abs(cp-cz[0]))/W)
 net.train(); return cor/max(1,n),(float(np.mean(berr)) if berr else 0)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=2500); ap.add_argument('--batch',type=int,default=128); ap.add_argument('--out',default='/kaggle/working/column_sep.pt'); a=ap.parse_args()
 rng=random.Random(33); order=list(range(len(words))); rng.shuffle(order); nv=max(1000,len(order)//20); tw=[words[i] for i in order[nv:]]
 sx=[];sy=[]
 while len(sx)<20000:
  s=synth(rng,tw)
  if s: sx.append(s[0]);sy.append(s[1])
 sx=np.stack(sx);sy=np.stack(sy); rx,ry,rp=load_real()
 up=sorted(set(rp)); vp=set(up[::10]); vm=np.array([p in vp for p in rp]); xv,yv=rx[vm],ry[vm]; xt,yt=rx[~vm],ry[~vm]
 print('real train/val',len(xt),len(xv),flush=True)
 dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); net=Net().to(dev); print('params',sum(p.numel() for p in net.parameters())/1e6,flush=True)
 opt=torch.optim.AdamW(net.parameters(),2e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=1e-5)
 dl=DataLoader(DS(sx,sy),batch_size=a.batch,shuffle=True,num_workers=4,drop_last=True,persistent_workers=True)
 def cyc():
  while True:
   for b in dl: yield b
 it=cyc(); best=0; out=pathlib.Path(a.out); t0=time.time()
 for st in range(1,a.steps+1):
  x,y=next(it); x=torch.stack([aug(q) for q in x]); k=min(a.batch//2,len(xt)); idx=torch.randint(0,len(xt),(k,)); xr=torch.from_numpy(1-xt[idx.numpy()].astype(np.float32)/255).unsqueeze(1); yr=torch.from_numpy(yt[idx.numpy()]).long()
  x=torch.cat([x[:a.batch-k],xr]); y=torch.cat([y[:a.batch-k],yr]); loss=F.cross_entropy(net(x.to(dev)),y.to(dev),label_smoothing=.02); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5); opt.step(); sch.step()
  if st%500==0 or st==a.steps:
   a1,b1=ev(net,sx[:2000],sy[:2000],dev); a2,b2=ev(net,xv,yv,dev); print(f'step {st} loss {loss.item():.3f} synth {a1:.4f}/{b1:.3f} real {a2:.4f}/{b2:.3f} {time.time()-t0:.0f}s',flush=True)
   if a2>best: best=a2; torch.save({'state_dict':net.state_dict(),'classes':CLASSES,'arch':'column_cnn','img_h':H,'img_w':W,'synth_acc':a1,'real_acc':a2,'boundary_mae':b2},out)
 (out.parent/'column_sep_metrics.json').write_text(json.dumps({'best_real_acc':best},indent=2)); print('saved',out,flush=True)
if __name__=='__main__': main()
