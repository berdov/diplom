"""Unchanged validation/summary plus mandatory historical control reproduction."""
from .. import report as parent
from .. import config as old
from . import config as c
from .reuse import bind
from experiments.mamba3_mimo_time.records import read


def replay_check(record):
    reference = read(old.paths('fixed_replay')['result'])
    keys = ('checkpoint_sha256','best_epoch','best_valid_metrics','actual_epochs',
            'first27_best_ndcg10','first_train_batch_sha256','initial_backbone_sha256',
            'initial_common_calibrator_hashes','rng_components','optimizer_settings','precision',
            'protocol','manifest_sha256','train_time_stats_sha256','verified_history_stats')
    differences = [k for k in keys if record.get(k)!=reference.get(k)]
    for i,(actual,expected) in enumerate(zip(record.get('history',[]),reference['history'])):
        for key in ('epoch','valid_ndcg10','valid_metrics','train_loss'):
            if actual.get(key)!=expected.get(key):differences.append(f'history[{i}].{key}')
        # Gap diagnostics have intentionally changed; common calibrators must not.
        a={k:v for k,v in actual['diagnostics'].items() if k!='gap_trap'}
        b={k:v for k,v in expected['diagnostics'].items() if k!='gap_trap'}
        if a!=b:differences.append(f'history[{i}].common_diagnostics')
    if differences:raise ValueError('Fresh fixed replay differs from predeclared exact contract: '+repr(differences))
    return dict(status='PASS',checkpoint_sha256=record['checkpoint_sha256'],scientific_history_exact=True,
                historical_job_id='4370162',timing_memory_excluded=True)


def validate_record(record,variant):
    _base_validate(record,variant)
    if variant=='fixed_replay' and record.get('status')=='PASS':replay_check(record)


def markdown(record):
    return parent.markdown(record).replace('# Gap Trap:', '# Centered Gap Trap:')


_engine=bind(parent,{'c':c},__package__)
_base_validate=_engine['validate_record']
_engine.update(validate_record=validate_record,markdown=markdown)
validate_checkpoint,summarize,write=(_engine[k] for k in ('validate_checkpoint','summarize','write'))
