import json,re,pathlib
from rapidfuzz.distance import Levenshtein
SUS=re.compile(r"[A-Za-z\\|*#\[\]{}<>]")
BLOCKS=[('sb01_comm_p0042','data/RTS_sb_01/gt/comm/p0042.txt'),('sb01_core_p0043','data/RTS_sb_01/gt/core/p0043.txt'),('sb01_comm_p0043','data/RTS_sb_01/gt/comm/p0043.txt'),('sb01_core_p0044','data/RTS_sb_01/gt/core/p0044.txt'),('sb01_comm_p0044','data/RTS_sb_01/gt/comm/p0044.txt')]
def groups(runs):
    by={}
    for r in runs: by.setdefault(r['line'],[]).append(r)
    out=[]
    for li in sorted(by):
        cur=[]
        for r in by[li]:
            if cur and r['script']!=cur[-1]['script']:
                out.append(cur); cur=[]
            cur.append(r)
        if cur: out.append(cur)
    return out
def gtext(g,key):
    if key=='drive': return g[0].get('drive_sep') or ''
    if key=='dev': return ' '.join((r.get('dev_run') or '').strip() for r in g if (r.get('dev_run') or '').strip())
    if key=='drive_line': return g[0].get('drive_line') or ''
def variant(g,v):
    d=gtext(g,'drive'); t=gtext(g,'dev'); s=g[0]['script']
    if v=='drive_sep': return d
    if v=='dev_run': return t
    if v=='dev_runs_for_dev': return t if s=='dev' else d
    if v=='dev_runs_for_tam': return t if s=='tam' else d
    if v=='empty_fill': return d or t
    if v=='suspect_hybrid': return t if (d and SUS.search(d) and t) else d
res={}; tot={}
for name,gtp in BLOCKS:
    runs=[json.loads(l) for l in open(f'evaluation/sb_eval/runs_{name}.jsonl')]
    gs=groups(runs); gt=''.join(open(gtp,encoding='utf-8').read().split()); res[name]={}
    for v in ['drive_sep','dev_run','dev_runs_for_dev','dev_runs_for_tam','empty_fill','suspect_hybrid']:
        lines=[]
        by={}
        for gi,g in enumerate(gs): by.setdefault(g[0]['line'],[]).append(g)
        for li in sorted(by):
            parts=[]; first=True
            for g in by[li]:
                t=variant(g,v)
                if not t: continue
                if not first and g[0]['pre_space']: parts.append(' ')
                parts.append(t); first=False
            lines.append(''.join(parts))
        q=''.join(('\n'.join(lines)).split()); d=Levenshtein.distance(gt,q); res[name][v]={'cer':d/len(gt),'dist':d,'n':len(gt)}
        a=tot.setdefault(v,[0,0]); a[0]+=d; a[1]+=len(gt)
out={'blocks':res,'micro':{k:{'cer':d/n,'dist':d,'n':n} for k,(d,n) in tot.items()},'method':'group-level; new pure line-layout eval'}
pathlib.Path('evaluation/sb_eval/combine_group.json').write_text(json.dumps(out,indent=1)+'\n')
print(json.dumps({k:round(v['cer'],4) for k,v in out['micro'].items()},indent=1))
