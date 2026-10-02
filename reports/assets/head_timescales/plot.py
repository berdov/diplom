"""Render one paired-delta figure from saved summary, without model imports."""
import argparse
import json
from pathlib import Path


def plot(summary,svg,png=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    pairs=summary['cohorts']['new4_full']['contrasts']['head_tau-shared_tau']['pairs']
    if [r['seed'] for r in pairs]!=[2027,2028,2029,2030]:raise ValueError('Four complete confirmation pairs required')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(8.8,4.8),layout='constrained')
    for i,row in enumerate(pairs):
        delta=row['delta'];color='#176B73' if delta>=0 else '#B64B3F'
        ax.plot([0,delta],[i,i],color=color,linewidth=2)
        ax.scatter([delta],[i],s=80,color=color,marker='o',zorder=3)
        ax.annotate(f'{delta:+.4f}',(delta,i),xytext=(9 if delta>=0 else -9,0),textcoords='offset points',
                    va='center',ha='left' if delta>=0 else 'right',color=color)
    ax.axvline(0,color='#777777',linewidth=1)
    bound=max(.0005,max(abs(row['delta']) for row in pairs))*1.65
    ax.set_xlim(-bound,bound)
    ax.set_yticks(range(4),['2027','2028','2029','2030'])
    ax.invert_yaxis();ax.set_ylim(3.5,-.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:+.4f}'))
    ax.set_xlabel('Paired Δ VALID NDCG@10 · head τ − shared τ')
    ax.set_title('Learned head-specific time scales',loc='left',fontweight='bold',pad=15)
    for name in ('top','right','left'):ax.spines[name].set_visible(False)
    ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.15)
    fig.text(.02,-.01,'Four new paired seeds. Exploratory pilot 2026 excluded; VALID only.',fontsize=9)
    fig.savefig(svg,format='svg',bbox_inches='tight',metadata={'Date':None})
    if png:fig.savefig(png,format='png',dpi=150,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('summary',type=Path);parser.add_argument('svg',type=Path);parser.add_argument('--png',type=Path)
    args=parser.parse_args();plot(json.loads(args.summary.read_text()),args.svg,args.png)
