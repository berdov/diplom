"""One primary paired-delta figure from audited saved results."""
import argparse
import json
from pathlib import Path


def plot(summary,svg,png=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    if summary['status']!='PASS' or summary['scientific_fits_completed']!=8 or summary['complete_pairs']!=4:
        raise ValueError('Only the full audited confirmation may be plotted')
    pairs=summary['primary_new_seeds']['pairs']
    if [r['seed'] for r in pairs]!=[2027,2028,2029,2030]:raise ValueError('Wrong primary cohort')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none',
                         'svg.hashsalt':'gap-trap-centered-confirmation'})
    fig,ax=plt.subplots(figsize=(7.8,4.8),layout='constrained')
    for i,row in enumerate(pairs):
        delta=row['delta'];positive=delta>=0;color='#176B73' if positive else '#B64B3F'
        ax.plot([i,i],[0,delta],color=color,linewidth=2)
        ax.scatter([i],[delta],s=70,color=color,marker='o' if positive else 's',zorder=3)
        ax.annotate(f'{delta:+.4f}',(i,delta),xytext=(0,10 if positive else -17),textcoords='offset points',ha='center',fontsize=10)
    ax.axhline(0,color='#666666',linewidth=1)
    bound=max(.0005,max(abs(r['delta']) for r in pairs))*1.4
    ax.set_ylim(-bound,bound);ax.set_xlim(-.5,3.5)
    ax.set_xticks(range(4),[str(r['seed']) for r in pairs]);ax.set_xlabel('Seed')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:+.4f}'))
    ax.set_ylabel('Paired Δ VALID NDCG@10\ncentered − fixed')
    ax.set_title('Centered Gap-Trap: confirmation',loc='left',fontweight='bold',pad=13)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16)
    fig.text(.02,-.015,'Four new paired seeds; exploratory pilot 2026 excluded. VALID only, TEST = 0.',fontsize=9)
    fig.savefig(svg,format='svg',bbox_inches='tight',metadata={'Date':None})
    Path(svg).write_text('\n'.join(line.rstrip() for line in Path(svg).read_text().splitlines())+'\n')
    if png:fig.savefig(png,format='png',dpi=160,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('summary',type=Path);parser.add_argument('svg',type=Path);parser.add_argument('--png',type=Path);args=parser.parse_args()
    plot(json.loads(args.summary.read_text()),args.svg,args.png)
