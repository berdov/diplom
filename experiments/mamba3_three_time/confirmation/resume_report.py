"""Resolve only prospectively selected attempts; never rewrite parent summaries."""
from . import config as c, resume_config as r, resume_provenance as q
from .provenance import create_record, sha, now
from .report import build, markdown, atomic_text
from .state import paired


def collect(base, check_bytes=True):
    records = {}
    for entry in r.plan()['source_index']:
        mode, seed = entry['mode'], entry['seed']
        row = q.resolve(mode, seed, base, check_bytes=check_bytes)
        records[(seed, mode)] = dict(row, source_json=str(c.ROOT/entry['path']), source_sha256=sha(c.ROOT/entry['path']))
    for seed in c.SEEDS:
        if all(records[(seed, mode)]['status'] == 'PASS' for mode in c.MODES):
            paired(records[(seed,'dual')], records[(seed,'triple')], first_batch=True)
    summaries = build(records)
    for horizons in summaries.values():
        for result in horizons.values():
            for pair in result['rows']:
                for mode in c.MODES:
                    original = records[(pair['seed'], mode)]
                    pair[mode].update(job_id=original['job_id'], execution_commit=original['execution_commit'],
                                      source_hash=original['source_hash'])
    new = [records[(seed,mode)] for mode,seed in r.ORDER]
    return dict(**base, generated_at=now(), summaries=summaries, exploratory_seed=2026,
                status='INCOMPLETE' if summaries['new_four_pairs']['full']['incomplete'] else 'PASS',
                scientific_fits_this_attempt=sum(x.get('scientific_fit_started',False) for x in new),
                previous_completed_confirmation_fits=1,
                total_completed_confirmation_fits=1+sum(x['status']=='PASS' for x in new),
                new_diagnostic_batches=0, new_diagnostic_optimizer_steps=0)


def main():
    base = q.identity()
    value = collect(base)
    create_record(r.SUMMARY, value)
    text = markdown(value) + '\n## Source index\n\n| seed | mode | job | commit | source | SHA256 |\n|---|---|---|---|---|---|\n'
    for pair in value['summaries']['all_five_pairs']['full']['rows']:
        for mode in c.MODES:
            row = pair[mode]
            text += f"| {pair['seed']} | {mode} | {row['job_id']} | {row['execution_commit']} | {row['source_json']} | {row['source_sha256']} |\n"
    atomic_text(r.SUMMARY.with_suffix('.md'), text)


if __name__ == '__main__':
    main()
