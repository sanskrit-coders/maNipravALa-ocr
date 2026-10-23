#!/usr/bin/env python3
import argparse, json, math, os, pathlib, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IN = pathlib.Path('/kaggle/input')
def find(name, kind='file'):
    hits = sorted(IN.rglob(name))
    return hits[0] if hits else None
code = find('train_classifier.py')
if code is None: raise SystemExit('train_classifier.py missing')
sys.path.insert(0, str(code.parent))
import train_classifier as T
fonts_dir = find('fonts', 'dir')
texts_dir = find('texts', 'dir')
if fonts_dir:
    T.DEV_FONTS = T.localize_fonts(T.DEV_FONTS, str(fonts_dir))
    T.TAM_FONTS = T.localize_fonts(T.TAM_FONTS, str(fonts_dir))
    T.OTHER_FONTS = T.localize_fonts(T.OTHER_FONTS, str(fonts_dir))
words = T.load_words(str(texts_dir))
print('words',len(words),'fonts',len(T.DEV_FONTS),len(T.TAM_FONTS),len(T.OTHER_FONTS),flush=True)
CLASSES=['dev','tam','other','mixed']; CLS={c:i for i,c in enumerate(CLASSES)}
H,W=64,256; MARGIN=8

def crop_ink(im):
 a=np.asarray(im); cols=np.where((a<200).any(axis=0))[0]; rows=np.where((a<200).any(axis=1))[0]
 if len(cols)==0 or len(rows)==0: return None
 return im.crop((max(0,int(cols[0])-MARGIN),max(0,int(rows[0])-MARGIN),min(a.shape[1],int(cols[-1])+MARGIN+1),min(a.shape[0],int(rows[-1])+MARGIN+1)))
def fit(im,rng,target_h=None):
 if target_h is None: target_h=rng.choice([24,28,32,36,40,48])
 w,h=im.size; nw=max(1,int(round(w*target_h/max(1,h))))
 if nw>W-8: nw=W-8; target_h=max(1,int(round(h*nw/max(1,w))))
 im=im.resize((nw,target_h),Image.BILINEAR); c=Image.new('L',(W,H),255); c.paste(im,(rng.randint(0,W-nw),rng.randint(0,H-target_h))); return np.asarray(c,dtype=np.uint8)
def pure_img(word,prof,seed,rng):
 r=T.render_word((word,prof,seed))
 if r is None: return None
 im=crop_ink(Image.fromarray(r['img'])); return fit(im,rng) if im else None
def other_img(seed,rng):
 r=T.render_other(seed)
 if r is None: return None
 im=crop_ink(Image.fromarray(r['img'])); return fit(im,rng) if im else None
def pure_label(prof):
 s={p for p in prof if p!='other'}; return next(iter(s)) if len(s)==1 else None

def make_synth(word_list,rng):
 kind=rng.choices(['dev','tam','other','mixed'],weights=[35,35,10,20])[0]
 if kind=='other':
  im=other_img(rng.randrange(1<<30),rng); return (im,CLS['other']) if im is not None else None
 if kind in ('dev','tam'):
  pool=[(w,p) for w,p in word_list if pure_label(p)==kind]
  if not pool: return None
  w,p=rng.choice(pool); im=pure_img(w,p,rng.randrange(1<<30),rng)
  return (im,CLS[kind]) if im is not None else None
 ds=[(w,p) for w,p in word_list if pure_label(p)=='dev']; ts=[(w,p) for w,p in word_list if pure_label(p)=='tam']
 if not ds or not ts: return None
 d=pure_img(*rng.choice(ds),rng.randrange(1<<30),rng); t=pure_img(*rng.choice(ts),rng.randrange(1<<30),rng)
 if d is None or t is None: return None
 a=np.asarray(d); b=np.asarray(t); ca=np.where((a<200).any(axis=0))[0]; cb=np.where((b<200).any(axis=0))[0]
 if len(ca)==0 or len(cb)==0: return None
 da=a[:,ca[0]:ca[-1]+1]; db=b[:,cb[0]:cb[-1]+1]; gap=rng.choice([0,1,2,3,5])
 canvas=Image.new('L',(da.shape[1]+db.shape[1]+gap,max(da.shape[0],db.shape[0])),255)
 canvas.paste(Image.fromarray(da),(0,0)); canvas.paste(Image.fromarray(db),(da.shape[1]+gap,0))
 return fit(crop_ink(canvas),rng),CLS['mixed']

def make_pool(word_list,n,seed):
 rng=random.Random(seed); xs=[];ys=[]
 while len(xs)<n:
  s=make_synth(word_list,rng)
  if s is None: continue
  xs.append(s[0]); ys.append(s[1])
 print('synth pool',len(xs),np.bincount(ys,minlength=4).tolist(),flush=True); return np.stack(xs),np.array(ys)
class DS(Dataset):
 def __init__(self,x,y): self.x=x; self.y=y
 def __len__(self): return len(self.y)
 def __getitem__(self,i): return torch.from_numpy(1-self.x[i].astype(np.float32)/255).unsqueeze(0),int(self.y[i])
def augment(x):
 H,W=x.shape[-2:]
 if random.random()<0.8:
  ang=random.uniform(-2.5,2.5)*math.pi/180; sc=random.uniform(0.88,1.12); tx=random.uniform(-3,3)*2/W; ty=random.uniform(-2,2)*2/H
  th=torch.tensor([[sc*math.cos(ang),-sc*math.sin(ang),tx],[sc*math.sin(ang),sc*math.cos(ang),ty]])
  grid=F.affine_grid(th.unsqueeze(0),(1,1,H,W),align_corners=False); x=F.grid_sample(x.unsqueeze(0),grid,mode='bilinear',padding_mode='border',align_corners=False)[0]
 if random.random()<0.5: x=(x*random.uniform(0.7,1.2)+random.uniform(-0.15,0.15)).clamp(0,1)
 if random.random()<0.4: x=(x+torch.randn_like(x)*random.uniform(0.005,0.04)).clamp(0,1)
 return x
class Net(nn.Module):
 def __init__(self):
  super().__init__()
  def block(a,b): return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.MaxPool2d(2))
  self.f=nn.Sequential(block(1,32),block(32,64),block(64,128),nn.AdaptiveAvgPool2d(1)); self.o=nn.Sequential(nn.Flatten(),nn.Dropout(0.2),nn.Linear(128,4))
 def forward(self,x): return self.o(self.f(x))
def load_real():
 j=find('p0036.jsonl'); recs=[json.loads(l) for l in j.read_text(encoding='utf-8').splitlines() if l.strip()]
 imgs=[];ys=[]
 for r in recs:
  img=next(IN.rglob(pathlib.Path(r['image']).name)); im=Image.open(img).convert('L').crop(r['box']); im=crop_ink(im)
  if im is None: continue
  imgs.append(fit(im,random.Random(r['gt_index']))); ys.append(CLS['mixed'])
 print('real',len(imgs),flush=True); return np.stack(imgs),np.array(ys)
def evaluate(net,x,y,dev):
 net.eval(); ps=[]
 with torch.no_grad():
  for i in range(0,len(x),256): ps.append(net(torch.from_numpy(1-x[i:i+256].astype(np.float32)/255).unsqueeze(1).to(dev)).argmax(1).cpu())
 p=torch.cat(ps); yt=torch.from_numpy(y); acc=float((p==yt).float().mean()); per={c:float((p[yt==CLS[c]]==yt[yt==CLS[c]]).float().mean()) if (yt==CLS[c]).any() else 0 for c in CLASSES}; net.train(); return acc,per
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=2500); ap.add_argument('--batch',type=int,default=128); ap.add_argument('--out',default='/kaggle/working/script_sep.pt'); args=ap.parse_args()
 rng=random.Random(22); order=list(range(len(words))); rng.shuffle(order); nv=max(1000,len(order)//20); tw=[words[i] for i in order[nv:]]; vw=[words[i] for i in order[:nv]]
 xt,yt=make_pool(tw,20000,1); xv,yv=make_pool(vw,3000,2); xr,yr=load_real(); xrv,yrv=xr[::5],yr[::5]
 dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); net=Net().to(dev); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps,eta_min=1e-5)
 loader=DataLoader(DS(xt,yt),batch_size=args.batch,shuffle=True,num_workers=4,drop_last=True,persistent_workers=True)
 def cyc():
  while True:
   for b in loader: yield b
 it=cyc(); best=0; out=pathlib.Path(args.out); t0=time.time()
 for step in range(1,args.steps+1):
  x,y=next(it); x=torch.stack([augment(a) for a in x]);
  k=min(32,len(xr)); idx=torch.randint(0,len(xr),(k,)); xr_b=torch.from_numpy(1-xr[idx.numpy()].astype(np.float32)/255).unsqueeze(1); yr_b=torch.from_numpy(yr[idx.numpy()]).long()
  x=torch.cat([x[:args.batch-k],xr_b]); y=torch.cat([y[:args.batch-k],yr_b])
  loss=F.cross_entropy(net(x.to(dev)),y.to(dev),label_smoothing=0.02); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5); opt.step(); sched.step()
  if step%500==0 or step==args.steps:
   a1,p1=evaluate(net,xv,yv,dev); a2,p2=evaluate(net,xrv,yrv,dev) if len(xrv) else (0,{})
   print(f'step {step} loss {loss.item():.3f} synth {a1:.4f} real {a2:.4f} '+ ' '.join(f'{c}:{p1[c]:.3f}' for c in CLASSES)+f' {time.time()-t0:.0f}s',flush=True)
   score=(a1+a2)/2
   if score>best: best=score; torch.save({'state_dict':net.state_dict(),'classes':CLASSES,'arch':'word_cnn','img_h':H,'img_w':W,'synth_acc':a1,'real_acc':a2},out)
 (out.parent/'script_sep_metrics.json').write_text(json.dumps({'best':best},indent=2))
 print('saved',out)
if __name__=='__main__': main()
