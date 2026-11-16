"""Generate report figures from archived evidence; no GPU work."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
colors=['#64748b','#0e7490','#15803d']
def box(ax,x,y,w,h,s,color='#e8f2f8'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015',facecolor=color,edgecolor='#486477',linewidth=1.1))
    ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=11)
def arrow(ax,a,b):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#486477','lw':1.5})
def save(fig,name):
    fig.savefig(ROOT/(name+'.svg'),bbox_inches='tight',facecolor='white')
    fig.savefig(ROOT/(name+'.png'),dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)
fig,ax=plt.subplots(figsize=(12,6.5));ax.set(xlim=(0,12),ylim=(0,6.5));ax.axis('off')
box(ax,3.1,5.6,5.8,.6,'MiniCPM5-2B / vLLM serving')
box(ax,.4,4.1,4.6,.85,'42 Down projections\nINT8 weights + FP32 channel scales')
box(ax,7,4.1,4.6,.85,'Gate/Up, QKV, O\nOriginal BF16 weights')
arrow(ax,(5,5.6),(2.7,4.95));arrow(ax,(7,5.6),(9.3,4.95))
box(ax,.4,2.65,4.6,.85,'Opaque custom op\nShape dispatch + capability checks')
box(ax,7,2.65,4.6,.85,'Registered BF16 MM provider\nM2048 / M4096 tile overrides')
arrow(ax,(2.7,4.1),(2.7,3.5));arrow(ax,(9.3,4.1),(9.3,3.5))
box(ax,.4,1.05,2.15,.95,'Small M\nFused Split-K', '#dcfce7')
box(ax,2.85,1.05,3.1,.95,'Large M\nConvert + MM/scale', '#dcfce7')
box(ax,7,1.05,4.6,.95,'BF16 dot / FP32 accumulation\nBF16 output', '#dcfce7')
arrow(ax,(1.5,2.65),(1.5,2));arrow(ax,(4.1,2.65),(4.1,2));arrow(ax,(9.3,2.65),(9.3,2))
ax.text(6,.35,'Dense and legal fallbacks retained | Offline tuning; no large search during serving',ha='center',fontsize=11,color='#475569')
save(fig,'01-architecture')
fig,ax=plt.subplots(figsize=(12,5.5));ax.set(xlim=(0,12),ylim=(0,5.5));ax.axis('off')
ax.text(.2,5.05,'Original Separate path',fontsize=15,weight='bold')
for x,w,s in [(.2,2.2,'INT8 Q\nload + convert'),(3.2,2.1,'BF16 Q buffer'),(6.1,2.1,'BF16 MM\nBF16 temporary'),(9.1,2.6,'Scale kernel\nBF16 output')]:box(ax,x,3.65,w,.9,s)
for a,b in [(2.4,3.2),(5.3,6.1),(8.2,9.1)]:arrow(ax,(a,4.1),(b,4.1))
ax.text(.2,2.75,'Adopted large-M path',fontsize=15,weight='bold')
box(ax,.2,1.3,2.2,1,'INT8 Q\nBLOCK4096 / 4 warps','#dcfce7')
box(ax,3.2,1.3,2.1,1,'BF16 Q buffer','#dcfce7')
box(ax,6.1,1.3,5.6,1,'Jointly tuned BF16 MM\nFP32 accumulator × scale → BF16 output','#dcfce7')
arrow(ax,(2.4,1.8),(3.2,1.8));arrow(ax,(5.3,1.8),(6.1,1.8))
ax.text(6,.4,'Conversion and allocation remain inside full-call timing; no persistent BF16 weight copy',ha='center',fontsize=11,color='#475569')
save(fig,'02-scale-fusion')
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
labels=['Fresh default\nBF16','Historical\ntuned BF16','Final\ncombination']
for ax,title,vals in zip(axes,['4K scenario','16K scenario'],[[5050.28,5127,5668.66],[6981.43,7651.48,8675.61]]):
    bars=ax.bar(labels,vals,color=colors,width=.6)
    ax.set_title(title,weight='bold');ax.set_ylabel('Total tokens / s (higher is better)')
    ax.set_ylim(0,max(vals)*1.17);ax.spines[['top','right']].set_visible(False)
    for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+max(vals)*.025,f'{v:,.2f}',ha='center',fontsize=11)
fig.text(.5,.015,'Fresh default BF16 + historical tuned BF16 references; not a quantization-only ablation.',ha='center',fontsize=10,color='#475569')
fig.tight_layout(rect=(0,.06,1,1));save(fig,'03-throughput')
print('Generated 3 PNG and 3 SVG figures')
