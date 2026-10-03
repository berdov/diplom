"""Publish only complete, independently audited saved confirmation artifacts."""
import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PACKAGE=ROOT/'experiments/mamba3_gap_trap/centered/confirmation'
BASE_MAIN='e8f8f1ecce0646c5815b524d0bd9683c423b4775'
MODES=('fixed_replay','centered_gap_trap')
SEEDS=(2027,2028,2029,2030)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def pm(value,signed=False):
    if value['mean'] is None:return '—'
    mean=0. if round(value['mean'],6)==0 else value['mean']
    label=format(mean,'+.6f' if signed else '.6f')
    return label+' ± '+format(value['sample_std'],'.6f') if value['sample_std'] is not None else label+' (n=1)'


def signs(group):
    return '/'.join(str(group['signs'][k]) for k in ('positive','negative','zero'))


def copy_exact(source,target):
    raw=Path(source).read_bytes();target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.read_bytes()!=raw:raise ValueError('Refusing to change existing artifact '+str(target))
    if not target.exists():target.write_bytes(raw)


def load_audited(folder):
    audit=read(folder/'independent_audit.json');files=folder/'files'
    if audit['status']!='PASS' or not audit['terminal'] or audit['scientific_fits_verified']!=8 or audit['complete_pairs_verified']!=4 or audit['test_evaluation_count']!=0:
        raise ValueError('Complete terminal audit required')
    records=[]
    for seed in SEEDS:
        for mode in MODES:
            label='fixed' if mode=='fixed_replay' else 'centered';run_id=f'mamba3_gaptrap_centered_confirm_{label}_seed{seed}_001'
            path=files/'runs'/(run_id+'.json')
            if sha(path)!=audit['result_sha256'][run_id]:raise ValueError('Result changed after audit')
            r=read(path)
            if (r['seed'],r['gap_trap_mode'],r['run_id'])!=(seed,mode,run_id):raise ValueError('Unplanned run identity')
            if r['status']!='PASS' or r['TEST']!='NOT_RUN' or r['test_evaluation_count']!=0:raise ValueError('Run incomplete/TEST')
            records.append(r)
    summaries=[p for p in (files/'slurm_logs').glob('attempt_*/confirmation_summary.json') if sha(p)==audit['summary_sha256']]
    if len(summaries)!=1:raise ValueError('Missing audited final summary')
    summary=read(summaries[0])
    for path in (files/'runs').glob('*.json'):copy_exact(path,PACKAGE/'runs'/path.name)
    copy_exact(summaries[0],PACKAGE/'runs/confirmation_summary.json')
    return audit,records,summary


def assets(folder,audit,records,summary):
    write_json(HERE/'summary.json',summary)
    sources=[];diagnostics=[]
    for r in records:
        path=PACKAGE/'runs'/(r['run_id']+'.json');meta=folder/'files/slurm_logs/fits'/r['run_id']/'checkpoints/best_metadata.json'
        sources.append(dict(run_id=r['run_id'],seed=r['seed'],variant=r['gap_trap_mode'],source_json=str(path.relative_to(ROOT)),
            result_sha256=sha(path),metadata=str(meta.relative_to(ROOT)),metadata_sha256=sha(meta),checkpoint_path=r['checkpoint_path'],
            checkpoint_sha256=r['checkpoint_sha256'],checkpoint_bytes=read(folder/'preservation_manifest.json')['checkpoints']['slurm_logs/fits/'+r['run_id']+'/checkpoints/best_state_dict.pth']['bytes'],execution_commit=r['execution_commit'],source_hash=r['source_hash'],job_id=r['job_id']))
        if r['gap_trap_mode']=='centered_gap_trap':
            trajectory=[dict(epoch=x['epoch'],alpha=x['diagnostics']['gap_trap']['alpha']) for x in r['history']];alpha=[x['alpha'] for x in trajectory]
            diagnostics.append(dict(seed=r['seed'],run_id=r['run_id'],best_epoch=r['best_epoch'],best_alpha=r['best_diagnostics']['gap_trap']['alpha'],
                max_epoch_alpha=max(alpha),final_alpha=alpha[-1],zero_alpha_epochs=alpha.count(0.),alpha_trajectory=trajectory,
                best_checkpoint=r['best_diagnostics']['gap_trap']))
    pilot=[]
    for mode in MODES:
        path=PACKAGE.parent/'runs/attempt_002'/f'mamba3_gaptrap_centered_{mode}_seed2026_002.json';r=read(path)
        pilot.append(dict(run_id=r['run_id'],source_json=str(path.relative_to(ROOT)),sha256=sha(path),cohort='exploratory pilot; excluded from primary'))
    references=[]
    for seed in SEEDS:
        path=ROOT/f'experiments/mamba3_mimo_time/confirmation/runs/mamba3_mimo_confirm_dual_seed{seed}_001.json'
        references.append(dict(seed=seed,source_json=str(path.relative_to(ROOT)),sha256=sha(path),role='replay reference only; not additional observations'))
    write_json(HERE/'sources.json',dict(confirmation=sources,exploratory_pilot=pilot,historical_replay_references=references,
        independent_audit=str((folder/'independent_audit.json').relative_to(ROOT)),audit_sha256=sha(folder/'independent_audit.json')))
    write_json(HERE/'diagnostics.json',dict(scope='Saved epoch boundaries and analytic best-checkpoint grid; no model/data forward',rows=diagnostics))
    group=summary['primary_new_seeds']
    tex=['% VALID only; new seeds 2027--2030; sample std ddof=1; pilot excluded.',
         r'\begin{tabular}{lcc}',r'\hline',r'Model / contrast & VALID NDCG@10 & $+/-/0$ \\',r'\hline']
    for label,key in [('Fixed','fixed'),('Centered','centered'),(r'Centered $-$ fixed','paired_delta')]:
        value=pm(group[key],key=='paired_delta').replace(' ± ',r' \pm ')
        tex.append(label+' & $'+value+'$ & '+(signs(group) if key=='paired_delta' else '--')+r' \\')
    tex.extend([r'\hline',r'\end{tabular}','% Four paired seeds; TEST evaluations = 0.',''])
    (HERE/'valid_table.tex').write_text('\n'.join(tex))


def detailed_report(audit,records,summary,conclusion):
    by={(r['seed'],r['gap_trap_mode']):r for r in records};g=summary['primary_new_seeds'];first=summary['first27_new_seeds']
    lines=['# Centered Gap-Trap: подтверждение','',conclusion.strip(),'','Завершены 8 новых запусков и 4 пары на заранее заданных seeds 2027–2030. Основная метрика — VALID NDCG@10, TEST = 0. Пилот 2026 был известен заранее и исключён из основной статистики.','',
        '| Seed | Fixed | Centered | Δ | HR@10 fixed / centered |','|---|---:|---:|---:|---:|']
    for pair in g['pairs']:
        f,z=(by[pair['seed'],v] for v in MODES)
        lines.append(f"| {pair['seed']} | {pair['fixed']:.4f} | {pair['centered']:.4f} | {pair['delta']:+.4f} | {f['best_valid_metrics']['hit@10']:.4f} / {z['best_valid_metrics']['hit@10']:.4f} |")
    lines += ['', '| Seed | Fixed HR@10 | Centered HR@10 | Δ HR@10 (secondary) |', '|---|---:|---:|---:|']
    for seed in SEEDS:
        f,z=(by[seed,v]['best_valid_metrics']['hit@10'] for v in MODES)
        lines.append(f'| {seed} | {f:.4f} | {z:.4f} | {z-f:+.4f} |')
    lines+=['','Mean ± sample std, ddof=1; relative — разница средних на одном наборе seeds.','',
        '| Набор | Fixed | Centered | Paired Δ | + / − / 0 | Relative |','|---|---:|---:|---:|---:|---:|']
    for label,group in [('Новые 4, primary',g),('Все 5, including exploratory pilot',summary['all5_including_exploratory_pilot']['full_run'])]:
        lines.append(f"| {label} | {pm(group['fixed'])} | {pm(group['centered'])} | {pm(group['paired_delta'],True)} | {signs(group)} | {group['relative_means_percent']:+.3f}% |")
    lines+=['','## Первые 27 эпох и длина обучения','','First27 — максимум на реальных эпохах 0–26; неполное окно не дополняется. Это те же histories, не независимые runs и не равный GPU-бюджет.','',
        '| Seed | Fixed first27 | Centered first27 | Δ | Best epochs fixed / centered | Actual epochs fixed / centered |',
        '|---|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        f,z=(by[seed,v] for v in MODES);x,y=f['first27_best_ndcg10'],z['first27_best_ndcg10']
        fmt=lambda v:'—' if v is None else f'{v:.4f}'
        delta='—' if x is None or y is None else f'{y-x:+.4f}'
        lines.append(f"| {seed} | {fmt(x)} | {fmt(y)} | {delta} | {f['best_epoch']} / {z['best_epoch']} | {f['actual_epochs']} / {z['actual_epochs']} |")
    lines += ['',f"First27 новых seeds: fixed **{pm(first['fixed'])}**, centered **{pm(first['centered'])}**; paired Δ **{pm(first['paired_delta'],True)}**, знаки **{signs(first)}**, доступных полных пар {len(first['pairs'])}/4.",
        'Эпохи нумеруются с нуля. Разная длина обучения при одинаковом early stopping не доказывает причинного эффекта extra epochs.','',
        '| Seed | Variant | TRAIN, с | VALID, с | Peak allocated / reserved, bytes |','|---|---|---:|---:|---:|']
    for r in records:lines.append(f"| {r['seed']} | {r['gap_trap_mode']} | {r['train_seconds']:.3f} | {r['valid_seconds']:.3f} | {r['peak_gpu_allocated_bytes']} / {r['peak_gpu_reserved_bytes']} |")
    lines += ['', '## Полные VALID-метрики', '', 'В каждой ячейке значения при K = 5 / 10 / 20 / 50. HR и Recall совпадают при одном релевантном target.', '',
              '| Seed | Variant | HR@K | Recall@K | NDCG@K |', '|---|---|---|---|---|']
    for r in records:
        values=[' / '.join(f"{r['best_valid_metrics'][metric+'@'+str(k)]:.4f}" for k in (5,10,20,50)) for metric in ('hit','recall','ndcg')]
        lines.append(f"| {r['seed']} | {r['gap_trap_mode']} | "+' | '.join(values)+' |')
    lines += ['', '| Seed | Variant | Checkpoint SHA256 |', '|---|---|---|']
    for r in records:lines.append(f"| {r['seed']} | {r['gap_trap_mode']} | `{r['checkpoint_sha256']}` |")
    lines+=['','Время включает first-use JIT/cache и не является сравнением warm-kernel latency.','',
        '## Alpha и BF16','','| Seed | Best alpha | Max по границам эпох | Final alpha | Нулевых эпох | BF16: исчезло / eligible | Fraction |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        if r['gap_trap_mode']!='centered_gap_trap':continue
        a=[x['diagnostics']['gap_trap']['alpha'] for x in r['history']];d=r['best_diagnostics']['gap_trap'];b=d['bf16'];fraction='N/A' if b['rounded_to_zero_fraction'] is None else f"{100*b['rounded_to_zero_fraction']:.2f}%"
        lines.append(f"| {r['seed']} | {d['alpha']:.9f} | {max(a):.9f} | {a[-1]:.9f} | {a.count(0.)}/{len(a)} | {b['rounded_to_zero_cases']}/{b['nonzero_shift_grid_cases']} | {fraction} |")
    lines += ['', 'Alpha — общий scalar силы centered Trap bias, не пользовательская шкала времени или доказанная важность механизма. BF16 fraction относится к аналитической сетке, не к событиям датасета. Precision не менялась.',
        '[Все траектории и q/shift/odds/sigmoid/BF16 grid](../../../../reports/assets/gap_trap_centered_confirmation/diagnostics.json). Новых dataset forwards для диагностики не было.','',
        '## Проверки и происхождение','',f"Сверены {sum(r['actual_epochs'] for r in records)} эпох и {audit['metric_cells']} значений метрик с RecBole log и process stderr. Exact historical fixed replay 4/4, checkpoint streaming SHA/metadata, initial/RNG/first batch и pair evidence согласованы. Weights при аудите не загружались.",
        'CPU 38/38 и no-Git 38/38 PASS; centered GPU 6/158 и smoke унаследованы по exact SHA, как и one-sided 4/133, head 9/228, MIMO 45/2342. Новых GPU gate/smoke не запускалось.',
        f"Execution `{audit['execution_commit']}`; source hash `{audit['source_hash']}`.",
        '[Аудит](evidence/job4372023/independent_audit.json), [preservation](evidence/job4372023/preservation_manifest.json), [raw summary](runs/confirmation_summary.json), [источники](../../../../reports/assets/gap_trap_centered_confirmation/sources.json).',
        'Все 12 метрик HR/Recall/NDCG@5/10/20/50 — в raw result JSON. P-values, SOTA/TEST claims и causal claims не делаются. Статья/Overleaf и пункт 4 не менялись.','']
    return '\n'.join(lines)


def registry(records):
    path=ROOT/'experiments/results.csv';before=path.read_bytes();frozen=subprocess.check_output(['git','show',BASE_MAIN+':experiments/results.csv'],cwd=ROOT)
    if before!=frozen:raise ValueError('Registry changed; inspect actual prefix before append')
    reader=csv.DictReader(io.StringIO(before.decode()));fields=reader.fieldnames;old=list(reader)
    if len(old)!=112 or len({r['run_id'] for r in old})!=112:raise ValueError('Unexpected registry count')
    rows=[]
    if {r['run_id'] for r in old}&{r['run_id'] for r in records}:raise ValueError('Duplicate registry run')
    for r in records:
        row={k:'' for k in fields};row.update(record_type='experiment',source='ours',run_id=r['run_id'],model='GapTrapMamba3Rec',
            model_variant='MIMO_dual_'+r['gap_trap_mode'],dataset='KuaiRand',protocol='B',split='validation',evaluation='full_7111_items',
            status='completed',seed=str(r['seed']),train_candidates='full_softmax',item_universe='7111',best_epoch=str(r['best_epoch']),
            actual_epochs=str(r['actual_epochs']),validation_ndcg10=str(r['best_valid_score']),test_evaluation_count='0',git_commit=r['execution_commit'],
            notes_path='reports/MAMBA3_TIME_MECHANISMS_RESULTS.md#gap-trap-centered-confirmation',test_used='no',
            source_json=str((PACKAGE/'runs'/(r['run_id']+'.json')).relative_to(ROOT)))
        for label,metric in [('HR','hit'),('Recall','recall'),('NDCG','ndcg')]:
            for k in (5,10,20,50):row[f'{label}@{k}']=str(r['best_valid_metrics'][f'{metric}@{k}'])
        rows.append(row)
    output=io.StringIO(newline='');csv.DictWriter(output,fieldnames=fields,lineterminator='\n').writerows(rows)
    path.write_bytes(before+output.getvalue().encode())
    if not path.read_bytes().startswith(before):raise ValueError('Registry prefix changed')
    return dict(before=112,after=120,added=8,previous_bytes=len(before),previous_sha256=hashlib.sha256(before).hexdigest(),
        registry_sha256=sha(path),old_prefix_unchanged=True,added_run_ids=[r['run_id'] for r in records])


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--conclusion',type=Path,required=True);p.add_argument('--append-registry',action='store_true');args=p.parse_args()
    audit,records,summary=load_audited(args.folder.resolve());assets(args.folder.resolve(),audit,records,summary)
    (PACKAGE/'RESULTS.md').write_text(detailed_report(audit,records,summary,args.conclusion.read_text()))
    if args.append_registry:
        if 'id="gap-trap-centered-confirmation"' not in (ROOT/'reports/MAMBA3_TIME_MECHANISMS_RESULTS.md').read_text():raise ValueError('Report anchor required first')
        write_json(PACKAGE/'runtime/publication_audit.json',dict(status='PASS',registry=registry(records),execution_commit=audit['execution_commit'],
            preservation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),independent_audit_sha256=sha(args.folder/'independent_audit.json'),
            scheduler_terminal_sha256=sha(args.folder/'scheduler_terminal.json'),preservation_manifest_sha256=sha(args.folder/'preservation_manifest.json')))


if __name__=='__main__':main()
