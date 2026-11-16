"""Render reproducible report figures from checked-in evidence; CPU only.
Run: MPLCONFIGDIR=/tmp/report-mpl python3 report-assets/generate_figures.py
Requires matplotlib; SVG text is converted to paths for portable previews.
"""
from pathlib import Path
import csv,json,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
KV=(REPO/'静态KV优化索引.md').exists()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
BLUE='#167594';GREEN='#278368';RED='#bf544c';GRAY='#64748b';INK='#20364b'
def save(fig,name):
 fig.savefig(ROOT/(name+'.svg'),bbox_inches='tight',facecolor='white')
 svg=ROOT/(name+'.svg')
 svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
 fig.savefig(ROOT/(name+'.png'),dpi=150,bbox_inches='tight',facecolor='white')
 plt.close(fig)
def canvas(title,subtitle='',height=6):
 fig,ax=plt.subplots(figsize=(12,height));ax.set(xlim=(0,12),ylim=(0,6));ax.axis('off')
 fig.suptitle(title,x=.06,ha='left',fontsize=19,color=INK,weight='bold')
 if subtitle:fig.text(.06,.89,subtitle,ha='left',fontsize=10,color=GRAY)
 fig.subplots_adjust(top=.85,bottom=.09,left=.03,right=.98)
 return fig,ax
def box(ax,x,y,w,h,text,color='#e8f2f6',size=11):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025',facecolor=color,edgecolor='#9bb3c2',lw=1))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=INK)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':GRAY,'lw':1.5})
def note(fig,text):fig.text(.06,.025,text,fontsize=9,color=GRAY)
if KV:
 fig,ax=canvas('Static KV INT8: one numerical contract','Post-RoPE Q/K permutation; BF16 dot operands, FP32 accumulation')
 box(ax,.3,4.25,3,.9,'Independent calibration\n42 layers / 2 KV heads')
 box(ax,4.1,4.25,3.5,.9,'Fixed 4-channel protection\nStatic per-channel K scales')
 arrow(ax,(3.3,4.7),(4.1,4.7))
 box(ax,.3,2.4,3,.95,'Post-RoPE Q, K, V\nReal physical slot mapping')
 box(ax,4.1,2.4,3.5,.95,'Fused Q transform + KV write\nQ-head parallelism')
 arrow(ax,(3.3,2.9),(4.1,2.9));arrow(ax,(5.85,4.25),(5.85,3.35))
 box(ax,8.4,2.4,3.2,.95,'Page32 cache\nINT8 K/V + BF16 protected K')
 arrow(ax,(7.6,2.9),(8.4,2.9))
 box(ax,4.1,.45,7.5,1,'Restore protected K / convert INT8 to BF16\nAttention + required reduction -> BF16 output','#e4f3ec')
 arrow(ax,(10,2.4),(10,1.45));arrow(ax,(5.8,2.4),(5.8,1.45))
 note(fig,'Q workspace and reduction remain; this is not native INT8 x INT8 Attention.')
 save(fig,'01-architecture')
 fig,ax=canvas('KV storage: 512 -> 276 bytes','Allocated bytes per token per KV head (D=128); not total model memory',5)
 ax.set_ylim(0,6)
 for y,parts,label in [(3.7,[(256,GRAY,'K BF16\n256 B'),(256,'#94a3b8','V BF16\n256 B')],'BF16'),(1.65,[(128,BLUE,'K INT8\n128 B'),(16,'#daa84e',''),(128,GREEN,'V INT8\n128 B'),(4,'#9e72af','')],'Static KV')]:
  x=1.6;ax.text(.1,y+.45,label,weight='bold',va='center')
  for n,c,t in parts:
   w=n/512*9;ax.barh(y+.45,w,left=x,height=.9,color=c,edgecolor='white')
   if t:ax.text(x+w/2,y+.45,t,ha='center',va='center',color='white',fontsize=11)
   x+=w
  ax.text(x+.15,y+.45,f'{sum(v[0] for v in parts)} B',va='center',weight='bold')
 ax.text(1.6,.85,'K metadata: 16 B (includes 4 protected BF16 values)     V scale: 4 B',fontsize=10)
 note(fig,'46.09% less allocated KV storage; freed capacity can still be filled by more tokens.')
 save(fig,'02-cache-layout')
 fig,ax=canvas('Fusing Q processing into real KV writes','Same static cache semantics; additional costs stay in the timed full call',5)
 for y,label,items in [(3.65,'Before',['Q transform','KV write','Attention','Reduce']), (1.45,'Adopted',['Q transform + KV write','Attention','Reduce'])]:
  ax.text(.1,y+1.1,label,weight='bold',color=INK)
  widths=[2.4]*4 if len(items)==4 else [4.1,3,2.8];x=.25
  for j,(w,t) in enumerate(zip(widths,items)):
   box(ax,x,y,w,.8,t,'#e4f3ec' if label=='Adopted' else '#e8f2f6')
   if j:arrow(ax,(x-.5,y+.4),(x,y+.4))
   x+=w+.5
 note(fig,'One writer per KV slot. Negative slots still process Q. BF16 Q workspace is retained.')
 save(fig,'03-fusion')
 # Parse the checked-in report table, preserving negative cases.
 text=(REPO/'report.md').read_text();section=text.split('### 4.1 ')[1].split('### 4.2 ')[0]
 points=[]
 for line in section.splitlines():
  cells=[x.strip() for x in line.strip('|').split('|')]
  if len(cells)==4 and re.match(r'^\d+ /',cells[0]):points.append((cells[0],float(cells[1]),float(cells[2])))
 fig,ax=plt.subplots(figsize=(12,5.8));labels=[x[0] for x in points];vals=[(1-c/b)*100 for _,b,c in points];yp=np.arange(len(vals))
 ax.barh(yp,vals,color=[GREEN if v>=0 else RED for v in vals]);ax.set_yticks(yp,labels);ax.invert_yaxis();ax.axvline(0,color=GRAY,lw=1)
 ax.set_xlabel('Full-call latency reduction vs original V-only fresh (%)');ax.set_title('KV microbenchmark: gains and regressions',loc='left',fontsize=17,pad=18)
 for i,v in enumerate(vals):ax.text(v+(0.5 if v>=0 else -.5),i,f'{v:+.2f}%',va='center',ha='left' if v>=0 else 'right',fontsize=10)
 ax.set_xlim(min(vals)-9,max(vals)+10);fig.tight_layout(rect=(0,.07,1,1));note(fig,'B / KV length / Q length; seed20261216, 9 paired rounds. Not a BF16 model-throughput comparison.');save(fig,'04-microbench')
 fig,ax=canvas('Static K scale folded into Q','Protected channels keep the original Q and BF16 K values',4)
 ax.text(.5,4.25,r'$d\notin S:\quad q_{\mathrm{eff},d}=\mathrm{BF16}(q_d s_d),\quad k_{\mathrm{eff},d}=\mathrm{BF16}(z_d)$',fontsize=20)
 ax.text(.5,2.8,r'$d\in S:\quad q_{\mathrm{eff},d}=q_d,\quad k_{\mathrm{eff},d}=k_d$',fontsize=20)
 ax.text(.5,1.25,r'$\mathrm{score}=\sum_d q_{\mathrm{eff},d}k_{\mathrm{eff},d}/\sqrt{D}$',fontsize=22)
 note(fig,'BF16 rounding after Q scaling differs from K-side dequantization; validate numerical error separately.');save(fig,'05-formulas')
else:
 fig,ax=canvas('W8A16 across devices: shared contract, local policies','Persistent INT8 weights + FP32 per-output-channel scales; BF16 activations')
 box(ax,2.5,4.6,7, .8,'Shared arithmetic + complete-call validation + Algorithm / Config')
 for x,title,body in [(.4,'MetaX C500','Original Split-K partial / reduce\nC500 launch and dispatch policy'),(6.5,'Iluvatar BI-V150','Reuse arithmetic; adapt JIT launch\nRemeasure tiles, split count and boundaries')]:
  box(ax,x,2.65,5.1,1.1,title+'\n'+body);arrow(ax,(x+2.55,4.6),(x+2.55,3.75))
 box(ax,.4,.55,11.2,1.05,'Fused Dense | Fused Split-K | Separate\nLocal device / software / provider hashes; legal fallback retained','#e4f3ec')
 arrow(ax,(2.95,2.65),(2.95,1.6));arrow(ax,(9.05,2.65),(9.05,1.6))
 note(fig,'Two platform adapters exist today; this is not a claim of a single merged automatic backend.');save(fig,'01-cross-platform')
 fig,ax=canvas('Three algorithms, complete costs','All paths restore INT8 weights to BF16 operands; FP32 accumulation',5.6)
 paths=[('Dense',['INT8 load / cast','BF16 dot + scale','BF16 output']),('Split-K',['FP32 workspace + partial','Reduce + scale','BF16 output']),('Separate',['INT8 -> BF16 buffer','BF16 MM + scale','BF16 output'])]
 for y,(label,items) in zip([4.45,2.8,1.15],paths):
  ax.text(.1,y+.38,label,va='center',weight='bold');x=1.8
  for j,(w,t) in enumerate(zip([3.1,3.2,2.2],items)):
   box(ax,x,y,w,.8,t,size=10)
   if j:arrow(ax,(x-.45,y+.4),(x,y+.4))
   x+=w+.45
 note(fig,'Allocation, conversion, partials and reduction count. Generic Separate has distinct BF16-intermediate rounding.');save(fig,'02-algorithms')
 rows=[json.loads(line) for line in (REPO/'docs/optimization/evidence/2026-11-16-down-cache-condition/results.jsonl').read_text().splitlines()]
 fig,axes=plt.subplots(1,2,figsize=(12,5.1),sharey=True)
 for ax,cond,title in zip(axes,['hot','cache_cleared'],['Hot cache: continuous replay','GPU cache cleared each timing iteration']):
  x=np.arange(len(rows));w=.34
  for off,key,label,color in [(-w/2,'current_public_bf16','Public BF16',GRAY),(w/2,'current_policy','W8A16 policy',BLUE)]:
   vals=[row['median_ms'][cond][key]*1000 for row in rows];bars=ax.bar(x+off,vals,width=w,label=label,color=color)
   for bar,v in zip(bars,vals):ax.text(bar.get_x()+w/2,v+1,f'{v:.2f}',ha='center',fontsize=10)
  ax.set_xticks(x,[str(row['seed']) for row in rows]);ax.set_title(title,fontsize=12);ax.set_xlabel('Independent seed');ax.set_ylim(0,110)
  # Shared legend below both axes avoids covering value labels.
 axes[0].set_ylabel('Full-call latency (microseconds; lower is better)')
 fig.suptitle('Down M64: cache conditions reverse the ranking',fontsize=18,x=.06,ha='left');fig.tight_layout(rect=(0,.17,1,.91));fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.065),ncol=2,frameon=False);note(fig,'Same-process paired diagnostic, 9 rounds per seed. Neither condition represents the full 42-layer model.');save(fig,'03-cache-conditions')
 e=REPO/'docs/optimization/evidence/2026-11-16-w8a16';kv=list(csv.DictReader((e/'static-kv-only-raw.csv').open()));combo=list(csv.DictReader((e/'combined-diagnostic-raw.csv').open()))
 fig,axes=plt.subplots(1,2,figsize=(12,5.1))
 for ax,length,base in zip(axes,[4096,16384],[3073.11,2790.28]):
  old=[float(x['Total tok/s']) for x in kv if int(x['Prefill'])==length];new=[float(x['Total tok/s']) for x in combo if int(x['Prefill'])==length];x=np.arange(2);w=.34
  for off,vals,label,color in [(-w/2,old,'Static KV',GREEN),(w/2,new,'Static KV + Down',BLUE)]:
   bars=ax.bar(x+off,vals,w,label=label,color=color)
   for bar,v in zip(bars,vals):ax.text(bar.get_x()+w/2,v+35,f'{v:.2f}',ha='center',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',pad=1))
  ax.axhline(base,color=GRAY,ls='--',label='Historical s2final');ax.set_xticks(x,['Round 1','Round 2']);ax.set_title(f'{length//1024}K scenario');ax.set_ylim(0,3900)
 axes[0].set_ylabel('Total tokens / s');fig.suptitle('Model combination: no consistent incremental gain from Down',fontsize=17,x=.06,ha='left');fig.tight_layout(rect=(0,.17,1,.91));fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False);note(fig,'Two-round diagnostics, not official four-run summaries. Different times/providers; not a strict one-factor A/B.');save(fig,'04-model-combination')
 fig,ax=canvas('Shared W8A16 numerical contract','INT8 storage does not imply native INT8 x INT8 GEMM',4)
 ax.text(.5,4.3,r'$Q_{n,k}=\mathrm{round\_clip}(W_{n,k}/s_n)$',fontsize=23)
 ax.text(.5,2.7,r'$Y=\mathrm{BF16}\left(\mathrm{FP32acc}(X_{\mathrm{BF16}}Q_{\mathrm{BF16}}^{T})\,s\right)$',fontsize=22)
 ax.text(.5,1.1,'s: FP32 per-output-channel scale     Q: persistent INT8 [N, K]',fontsize=14,color=INK)
 note(fig,'Bias / TP ordering remains in the framework. Generic Separate rounding is validated with its own reference.');save(fig,'05-formulas')
print('Generated five PNG and five SVG report figures')
