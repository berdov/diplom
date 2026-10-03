"""Draw saved best-checkpoint curves; no model or dataset access."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
diag=json.loads((HERE/'diagnostics.json').read_text())['layer_specific']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'layer-temporal-pilot-2026'})
fig,axes=plt.subplots(2,2,figsize=(9,6),sharex=True,sharey=True,layout='constrained')
for row,name in enumerate(('decay','scan')):
    for h in range(2):
        ax=axes[row,h]
        for layer,color in ((0,'#2563eb'),(1,'#d97706')):
            d=diag['layers'][layer][name]
            ax.plot(d['gaps_over_R0'],[x[h] for x in d['scale_by_gap_and_head']],marker='o',ms=4,lw=1.8,label=f'Layer {layer}',color=color)
        ax.set_xscale('symlog',linthresh=.01,linscale=.8)
        ax.set_xticks([0,.01,.1,1,10,100],['0','0.01','0.1','1','10','100'])
        ax.set_ylim(.45,2.05)
        ax.set_title(f'{name} · H{h}')
        ax.axhline(1,color='#9ca3af',lw=.7,ls='--')
        ax.grid(alpha=.18)
        if row==1:ax.set_xlabel('Gap / R0')
        if h==0:ax.set_ylabel('Scale')
axes[0,0].legend(frameon=False)
fig.suptitle('Independent temporal functions by layer\nSeed 2026 · selected epoch 28 · fixed R0 = 838393 ms',fontsize=13)
fig.savefig(HERE/'curves.svg',metadata={'Date':None})
svg=HERE/'curves.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
# Local preview only; the publication uses the vector file.
fig.savefig('/tmp/layer-temporal-curves-preview.png',dpi=140)
plt.close(fig)
