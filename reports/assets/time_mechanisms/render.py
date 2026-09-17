"""Render compact, deterministic SVG figures from extracted existing artifacts.

No plotting dependency or scientific model evaluation is required.
"""
import json
import math
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLORS = ['#0072b2', '#d55e00', '#009e73', '#cc79a7', '#333333']


def text(x, y, value, size=15, anchor='start', color='#222'):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}">{escape(str(value))}</text>'


def svg(width, height):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>', '<g font-family="Arial, sans-serif">']


def line(points, color, dashed=False):
    dash = ' stroke-dasharray="7 4"' if dashed else ''
    return f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="2"{dash}/>'


def learning(data):
    out = svg(1000, 510)
    out += [text(72, 28, 'VALID NDCG@10: existing logs, seed 2026', 20)]
    xmin, xmax, ymin, ymax = 0, 62, 0., .07
    X = lambda x: 75 + 760*(x-xmin)/(xmax-xmin)
    Y = lambda y: 365 - 290*(y-ymin)/(ymax-ymin)
    for y in [0,.01,.02,.03,.04,.05,.06,.07]:
        out += [line([(75,Y(y)),(835,Y(y))], '#ddd'), text(66,Y(y)+5,f'{y:.2f}',13,'end')]
    for x in [0,10,20,26,30,40,50,62]:
        out += [text(X(x),389,x,13,'middle')]
    out += [line([(X(26),75),(X(26),365)], '#777', True), text(X(26)+6,95,'first 27 epochs',13)]
    for i,(mode,row) in enumerate(data['runs'].items()):
        if not row['history']:
            continue
        out += [line([(X(r['epoch']),Y(r['valid_ndcg10'])) for r in row['history']],COLORS[i])]
        best = next(r for r in row['history'] if r['epoch']==row['best_epoch'])
        out += [f'<circle cx="{X(best["epoch"])}" cy="{Y(best["valid_ndcg10"])}" r="4" fill="{COLORS[i]}"/>']
        lx = 75 + (i%3)*280; ly = 438 + (i//3)*25
        out += [line([(lx,ly-5),(lx+22,ly-5)],COLORS[i]),text(lx+29,ly,f'{mode}; best epoch {best["epoch"]}',13)]
    out += [text(455,412,'Epoch (zero-based)',15,'middle'),text(75,496,'Source: extracted.json; dots = selected maxima; no repeated VALID.',13),'</g></svg>']
    (HERE/'learning_curves.svg').write_text('\n'.join(out)+'\n')


def calibrators(data):
    out = svg(1080, 805)
    out += [text(45,30,'Best-checkpoint calibrator functions (CPU, not VALID distributions)',21)]
    modes = ['shared','decay_only','scan_only','separate']
    gaps = data['gap_ms']; pos = [x for x in gaps if x > 0]
    lo, hi = math.log10(min(pos)), math.log10(max(pos))
    for n,mode in enumerate(modes):
        ox = 65+(n%2)*535; oy = 90+(n//2)*340
        X=lambda x:ox+65+350*(math.log10(x)-lo)/(hi-lo)
        Y=lambda y:oy+200-180*(y-.5)/1.5
        out += [text(ox,oy-22,mode,19)]
        for val in [.5,1.,1.5,2.]:
            out += [line([(ox,Y(val)),(ox+415,Y(val))],'#ddd'),text(ox-8,Y(val)+5,val,13,'end')]
        ref=X(data['reference_ms'])
        out += [line([(ref,oy+20),(ref,oy+200)],'#777',True),text(ref+5,oy+16,'reference: 13.97 min',12)]
        for val,label in [(10,'10 ms'),(1000,'1 s'),(60000,'1 min'),(3600000,'1 h'),(86400000,'1 d')]:
            out += [text(X(val),oy+225,label,12,'middle')]
        out += [text(ox+6,oy+225,'0',13,'middle'),text(ox+230,oy+249,'Physical history gap (log axis for >0)',13,'middle')]
        row=data['runs'][mode]
        if not row['checkpoint_exists']:
            out += [text(ox+40,oy+90,'Checkpoint unavailable',18)]
        for j,(name,c) in enumerate(row.get('calibrators',{}).items()):
            path='shared' if mode=='shared' else name.split('.')[-2]
            for head in range(2):
                color=COLORS[j*2+head]
                out += [line([(X(gap),Y(s[head])) for gap,s in zip(gaps,c['active_scales']) if gap>0],color)]
                out += [f'<circle cx="{ox+6}" cy="{Y(c["active_scales"][0][head])}" r="3" fill="{color}"/>']
                lx=ox+(j*2+head)%2*225; ly=oy+276+(j*2+head)//2*22
                out += [line([(lx,ly-5),(lx+20,ly-5)],color),text(lx+27,ly,f'{path}, head {head}',13)]
    out += [text(45,768,'Bounds [0.5, 2]. Active zero-gap dots at left; first event / padding are always 1 (not plotted).',14),
            text(45,793,'Grid: frozen TRAIN min/max + reference. Source/checkpoint SHA and weights: extracted.json.',14),'</g></svg>']
    (HERE/'calibrator_curves.svg').write_text('\n'.join(out)+'\n')


if __name__ == '__main__':
    data=json.loads((HERE/'extracted.json').read_text())
    learning(data)
    calibrators(data)
