#!/usr/bin/env python3
import argparse, json, math, os, pathlib, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader

IN = pathlib.Path(os.environ.get('WORD_INPUT', '/kaggle/input'))
def find(name, kind='file'):
    hits = sorted(IN.rglob(name))
    return hits[0] if hits else None

code = pathlib.Path(os.environ['WORD_CODE']) if os.environ.get('WORD_CODE') else find('train_classifier.py')
if code is None: raise SystemExit('train_classifier.py missing')
sys.path.insert(0, str(code.parent))
import train_classifier as T

fonts_dir = os.environ.get('WORD_FONTS') or find('fonts', 'dir')
texts_dir = os.environ.get('WORD_TEXTS') or find('texts', 'dir')
if texts_dir is None: raise SystemExit('texts missing')
if fonts_dir:
    T.DEV_FONTS = T.localize_fonts(T.DEV_FONTS, str(fonts_dir))
    T.TAM_FONTS = T.localize_fonts(T.TAM_FONTS, str(fonts_dir))
    T.OTHER_FONTS = T.localize_fonts(T.OTHER_FONTS, str(fonts_dir))
words = T.load_words(str(texts_dir))
print('words', len(words), 'fonts', len(T.DEV_FONTS), len(T.TAM_FONTS), len(T.OTHER_FONTS), flush=True)

CLASSES = ['dev', 'tam', 'other']
CLS = {c:i for i,c in enumerate(CLASSES)}
H, W = 64, 256
MARGIN = 8

def crop_ink(im):
    a = np.asarray(im)
    cols = np.where((a < 200).any(axis=0))[0]
    rows = np.where((a < 200).any(axis=1))[0]
    if len(cols) == 0 or len(rows) == 0: return None
    x0 = max(0, int(cols[0]) - MARGIN); x1 = min(a.shape[1], int(cols[-1]) + MARGIN + 1)
    y0 = max(0, int(rows[0]) - MARGIN); y1 = min(a.shape[0], int(rows[-1]) + MARGIN + 1)
    return im.crop((x0, y0, x1, y1))

def fit_word(im, rng, target_h=None):
    if target_h is None: target_h = rng.choice([24, 28, 32, 36, 40, 48])
    w, h = im.size
    nw = max(1, int(round(w * target_h / max(1, h))))
    if nw > W - 8:
        nw = W - 8; target_h = max(1, int(round(h * nw / max(1, w))))
    im = im.resize((nw, target_h), Image.BILINEAR)
    canvas = Image.new('L', (W, H), 255)
    x = rng.randint(0, max(0, W - nw)); y = rng.randint(0, max(0, H - target_h))
    canvas.paste(im, (x, y))
    return canvas

def render_sample(word, prof, seed, label):
    rng = random.Random(seed)
    r = T.render_word((word, prof, seed))
    if r is None: return None
    im = Image.fromarray(r['img'])
    im = crop_ink(im)
    if im is None: return None
    im = fit_word(im, rng)
    return np.asarray(im, dtype=np.uint8), CLS[label]

def render_other_sample(seed):
    rng = random.Random(seed)
    r = T.render_other(seed)
    if r is None: return None
    im = crop_ink(Image.fromarray(r['img']))
    if im is None: return None
    im = fit_word(im, rng)
    return np.asarray(im, dtype=np.uint8), CLS['other']

def pure_label(prof):
    s = {p for p in prof if p != 'other'}
    return next(iter(s)) if len(s) == 1 else None

def make_pool(word_list, n, seed):
    rng = random.Random(seed); xs=[]; ys=[]
    tries=0
    while len(xs) < n and tries < n*8:
        tries += 1
        if rng.random() < 0.18:
            s = render_other_sample(rng.randrange(1<<30))
        else:
            w, prof = rng.choice(word_list)
            lab = pure_label(prof)
            if lab not in ('dev','tam'): continue
            s = render_sample(w, prof, rng.randrange(1<<30), lab)
        if s is None: continue
        xs.append(s[0]); ys.append(s[1])
    x = np.stack(xs); y = np.array(ys, dtype=np.int64)
    print('pool', x.shape, np.bincount(y, minlength=3).tolist(), flush=True)
    return x, y

class DS(Dataset):
    def __init__(self, x, y, train=True): self.x=x; self.y=y; self.train=train
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        a = self.x[i].astype(np.float32)/255.0
        x = torch.from_numpy(1.0-a).unsqueeze(0)
        if self.train: x = augment(x)
        return x, int(self.y[i])

def augment(x):
    H, W = x.shape[-2:]
    if random.random() < 0.8:
        ang = random.uniform(-2.5, 2.5)*math.pi/180
        sc = random.uniform(0.88, 1.12)
        tx = random.uniform(-3,3)*2/W; ty = random.uniform(-2,2)*2/H
        theta = torch.tensor([[sc*math.cos(ang), -sc*math.sin(ang), tx],
                              [sc*math.sin(ang), sc*math.cos(ang), ty]])
        grid = F.affine_grid(theta.unsqueeze(0), (1,1,H,W), align_corners=False)
        x = F.grid_sample(x.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=False)[0]
    if random.random() < 0.5: x = (x*random.uniform(0.7,1.2)+random.uniform(-0.15,0.15)).clamp(0,1)
    if random.random() < 0.4: x = (x + torch.randn_like(x)*random.uniform(0.005,0.04)).clamp(0,1)
    if random.random() < 0.25:
        k = random.choice([3,5]); pad=k//2
        x = F.avg_pool2d(F.pad(x.unsqueeze(0),(pad,pad,pad,pad),mode='replicate'),k,stride=1)[0]
    return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a,b):
            return nn.Sequential(nn.Conv2d(a,b,3,padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                                 nn.Conv2d(b,b,3,padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.f = nn.Sequential(block(1,32), block(32,64), block(64,128), nn.AdaptiveAvgPool2d(1))
        self.o = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128,3))
    def forward(self,x): return self.o(self.f(x))

def evaluate(net, x, y, dev):
    net.eval(); preds=[]
    with torch.no_grad():
        for i in range(0,len(x),512):
            xb=torch.from_numpy(1.0-x[i:i+512].astype(np.float32)/255.0).unsqueeze(1).to(dev)
            preds.append(net(xb).argmax(1).cpu())
    p=torch.cat(preds); yt=torch.from_numpy(y)
    acc=float((p==yt).float().mean())
    per={c:float((p[yt==CLS[c]]==yt[yt==CLS[c]]).float().mean()) if (yt==CLS[c]).any() else 0.0 for c in CLASSES}
    cm=torch.zeros(3,3,dtype=torch.long)
    for t,q in zip(yt,p): cm[t,q]+=1
    net.train(); return acc,per,cm.tolist()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=3000); ap.add_argument('--batch',type=int,default=256)
    ap.add_argument('--lr',type=float,default=2e-3); ap.add_argument('--train-pool',type=int,default=20000)
    ap.add_argument('--val-pool',type=int,default=5000); ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--out',default='/kaggle/working/word_cls.pt')
    args=ap.parse_args()
    rng=random.Random(args.seed); order=list(range(len(words))); rng.shuffle(order)
    nval=max(1000,len(order)//20); vw=[words[i] for i in order[:nval]]; tw=[words[i] for i in order[nval:]]
    xt,yt=make_pool(tw,args.train_pool,args.seed+1); xv,yv=make_pool(vw,args.val_pool,args.seed+2)
    dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net=Net().to(dev); print('device',dev,'params',sum(p.numel() for p in net.parameters())/1e6,flush=True)
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps,eta_min=1e-5)
    loader=DataLoader(DS(xt,yt),batch_size=args.batch,shuffle=True,num_workers=4,drop_last=True,persistent_workers=True)
    def cycle():
        while True:
            for b in loader: yield b
    it=cycle(); best=0.0; out=pathlib.Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); t0=time.time()
    for step in range(1,args.steps+1):
        x,y=next(it); loss=F.cross_entropy(net(x.to(dev)),y.to(dev),label_smoothing=0.02)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step(); sched.step()
        if step%500==0 or step==args.steps:
            acc,per,cm=evaluate(net,xv,yv,dev)
            print(f'step {step} loss {loss.item():.3f} acc {acc:.4f} '+' '.join(f'{c}:{per[c]:.4f}' for c in CLASSES)+f' {time.time()-t0:.0f}s',flush=True)
            if acc>best:
                best=acc
                torch.save({'state_dict':net.state_dict(),'classes':CLASSES,'arch':'word_cnn','img_h':H,'img_w':W,'val_acc':acc,'per_class':per},out)
    acc,per,cm=evaluate(net,xv,yv,dev)
    torch.save({'state_dict':net.state_dict(),'classes':CLASSES,'arch':'word_cnn','img_h':H,'img_w':W,'val_acc':best,'per_class':per},out.with_name('word_cls_last.pt'))
    metrics={'best_val_acc':best,'val_acc':acc,'per_class':per,'confusion':cm,'classes':CLASSES,
             'train_pool':len(yt),'val_pool':len(yv),'steps':args.steps,'params_M':sum(p.numel() for p in net.parameters())/1e6,'seed':args.seed}
    (out.parent/'word_cls_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print('FINAL',json.dumps(metrics),flush=True)

if __name__=='__main__': main()
