import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.mamba3_input_time import config, provenance, report, submit
from experiments.mamba3_input_time.adapter import InputAdapter, COUNTS
from experiments.mamba3_input_time.gpu_checks import require_gpu_evidence


def test_config_and_source_guard():
    actual = provenance.verify()
    assert len(actual['source_hash'])==64
    reference = json.loads((config.ROOT/config.plan()['historical_reference']['source_json']).read_text())
    for task in config.plan()['tasks']:
        assert config.scientific_settings(config.settings(task))==config.scientific_settings(reference['config'])
    assert config.plan()['parameter_counts']==COUNTS


def test_unique_paths_and_overwrite_guards(tmp_path):
    tasks = config.plan()['tasks']
    for key in config.paths(tasks[0]):
        assert len({config.paths(t)[key] for t in tasks})==4
    p = {key:tmp_path/key for key in ('lock','result','checkpoint','metadata')}
    provenance.acquire(p,dict(run_id='synthetic'))
    with pytest.raises(FileExistsError):
        provenance.acquire(p,dict(run_id='duplicate'))
    for key in ('result','checkpoint','metadata'):
        other = {k:tmp_path/(key+'_'+k) for k in p}
        other[key].touch()
        with pytest.raises(FileExistsError):
            provenance.acquire(other,{})


def test_common_attention_weights_and_isolated_rng():
    before = provenance.rng_hash()
    states = []
    for mode in config.MODES:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2026)
            model = InputAdapter(mode)
        assert provenance.rng_hash()==before
        if mode.startswith('attention_'):
            states.append(provenance.tensor_hash(model.common_state()))
    assert states[0]==states[1]


@pytest.mark.parametrize('answer', ['success','failed','timeout','ambiguous'])
def test_submit_is_durable_and_exactly_once(monkeypatch,tmp_path,answer):
    commit='a'*40
    monkeypatch.setenv('RUN_COMMIT',commit)
    monkeypatch.setenv('EXPECTED_CORE_HASH',config.CORE)
    monkeypatch.setenv('EXPECTED_STUDY_HASH','study')
    monkeypatch.setattr(submit,'verify',lambda:dict(source_hash='study'))
    monkeypatch.setattr(submit,'RUNTIME',tmp_path)
    monkeypatch.setattr(submit,'GPU_EVIDENCE',tmp_path/'gpu.json')
    monkeypatch.setattr(submit,'SUMMARY',tmp_path/'summary.json')
    monkeypatch.setattr(submit,'paths',lambda t:{k:tmp_path/(t['mode']+k) for k in ('result','lock','checkpoint','metadata')})
    def git(args,**kwargs):
        if args[1]=='rev-parse': return commit+'\n'
        if args[1]=='branch': return 'exp/mamba3-input-time\n'
        return ''
    monkeypatch.setattr(submit.subprocess,'check_output',git)
    calls=[]
    def execute(command,**kwargs):
        calls.append(command)
        record=json.loads((tmp_path/'submission_001.json').read_text())
        assert record['status']=='PREPARED'
        assert command==['sbatch','--parsable','slurm/mamba3_input_time.sh']
        if answer=='timeout': raise subprocess.TimeoutExpired(command,60)
        return SimpleNamespace(returncode=1 if answer=='failed' else 0,
                               stdout='123;cluster\n' if answer=='success' else 'unparseable',stderr='')
    if answer=='success':
        submit.submit_once(execute)
    else:
        with pytest.raises((RuntimeError,subprocess.TimeoutExpired)):
            submit.submit_once(execute)
    with pytest.raises(FileExistsError):
        submit.submit_once(execute)
    assert len(calls)==1


def synthetic_success(task):
    reference=json.loads((config.ROOT/config.plan()['historical_reference']['source_json']).read_text())
    row=copy.deepcopy(reference)
    row.update(**task,study_id=config.STUDY_ID,source_hash='synthetic',core_hash=config.CORE,
               config=config.settings(task),parameter_count=COUNTS[task['mode']],test_status='NOT_RUN',
               runtime_seconds=10,peak_cuda_allocated_bytes=100,peak_cuda_reserved_bytes=200)
    return row


def test_report_ties_and_no_provenance_warning():
    records={t['mode']:synthetic_success(t) for t in config.plan()['tasks']}
    result=report.summarize(records,'synthetic')
    assert result['status']=='COMPLETE'
    assert result['comparisons']['attention_time_minus_attention_content']==0
    row=records['attention_time']
    row['source_hash']='wrong'
    with pytest.raises(ValueError): report.summarize(records,'synthetic')
    row['source_hash']='synthetic'
    row['test_evaluation_count']=1
    with pytest.raises(ValueError): report.summarize(records,'synthetic')


def test_partial_report_and_svg(tmp_path):
    result=report.summarize({},'synthetic')
    assert result['status']=='INCOMPLETE' and all(r['status']=='NOT_RUN' for r in result['rows'])
    result['historical']=dict(ndcg10=.0633)
    report.render(result,tmp_path/'summary.svg')
    assert (tmp_path/'summary.svg').is_file()
    with pytest.raises(FileExistsError): report.render(result,tmp_path/'summary.svg')


def test_trainer_consumes_batch_and_preserves_last_equal_tie(tmp_path):
    from experiments.mamba3_input_time.trainer import trainer_class
    loader=object()
    model=torch.nn.Linear(2,1)
    model.diagnostic_collector=None
    batch=SimpleNamespace(interaction={'ids':torch.tensor([1,2])})
    calls=[]
    model.calculate_loss=lambda actual: calls.append(actual) or torch.tensor(1.)
    class Base:
        def _train_epoch(self,data,epoch,loss_func,show_progress):
            return loss_func(data)
    record=dict(mode='separate_replay',seed=2026,run_id='mock',execution_commit='a'*40,
                config_sha256='cfg',source_hash='source',core_hash=config.CORE,history=[])
    p={k:tmp_path/k for k in ('result','checkpoint','metadata')}
    cls=trainer_class(Base,loader,record,p,provenance.tensor_hash(batch.interaction))
    trainer=cls()
    trainer.model=model
    trainer._train_epoch(batch,0)
    assert calls==[batch] and record['first_train_batch_sha256']==provenance.tensor_hash(batch.interaction)
    for epoch in (0,1):
        row=dict(epoch=epoch,valid_ndcg10=.06,valid_metrics={'ndcg@10':.06},diagnostics={'epoch':epoch})
        record['history'].append(row)
        trainer.best_valid_score=.06
        trainer._save_checkpoint(epoch)
    assert record['best_epoch']==1 and record['best_diagnostics']=={'epoch':1}
    with pytest.raises(ValueError): trainer.evaluate(object())
    restored=torch.load(p['checkpoint'],weights_only=True)
    assert provenance.tensor_hash(restored)==provenance.tensor_hash(model.state_dict())


def test_incomplete_gpu_evidence_rejected(monkeypatch):
    monkeypatch.setenv('RUN_COMMIT','a'*40)
    monkeypatch.setenv('SLURM_JOB_ID','synthetic')
    for evidence in ({},{'status':'PASS'}):
        with pytest.raises(ValueError): require_gpu_evidence(evidence,dict(source_hash='expected'))


def test_frozen_scientific_files_unchanged():
    base=config.plan()['base_main']
    directories=['experiments/mamba3_baseline','experiments/mamba3_timeaware',
                 'experiments/mamba3_time_mechanisms','experiments/mamba3_time_confirmation',
                 'experiments/mamba3_context_time','experiments/results.csv','reports']
    changed=subprocess.check_output(['git','diff','--name-only',base,'--',*directories],cwd=config.ROOT,text=True)
    assert changed==''


def test_source_errors_not_swallowed_as_plot_warning(monkeypatch):
    def fail(): raise ValueError('source drift')
    monkeypatch.setattr(report,'verify',fail)
    with pytest.raises(ValueError,match='source drift'): report.main()


def test_plot_failure_preserves_numeric_and_markdown(monkeypatch,tmp_path):
    (tmp_path/'runs').mkdir()
    monkeypatch.setattr(report,'HERE',tmp_path)
    monkeypatch.setattr(report,'SUMMARY',tmp_path/'runs/pilot_summary.json')
    monkeypatch.setattr(report,'paths',lambda t:{'result':tmp_path/(t['mode']+'.json')})
    monkeypatch.setattr(report,'verify',lambda:dict(source_hash='synthetic',files={}))
    def fail(*args): raise ImportError('synthetic optional renderer error')
    monkeypatch.setattr(report,'render',fail)
    report.main()
    result=json.loads((tmp_path/'runs/pilot_summary.json').read_text())
    assert result['plot_status']=='FAILED' and result['status']=='INCOMPLETE'
    assert len(result['rows'])==4 and (tmp_path/'runs/pilot_summary.md').is_file()


@pytest.mark.parametrize('failure', [None,'gpu_checks','run','budget'])
def test_pipeline_finite_order_and_stop(monkeypatch,tmp_path,failure):
    from experiments.mamba3_input_time import pipeline
    monkeypatch.setenv('SLURM_JOB_ID','synthetic')
    monkeypatch.setenv('RUN_COMMIT','a'*40)
    monkeypatch.setattr(pipeline,'RUNTIME',tmp_path/'runtime')
    monkeypatch.setattr(pipeline,'HERE',tmp_path)
    monkeypatch.setattr(pipeline,'GPU_EVIDENCE',tmp_path/'gpu.json')
    monkeypatch.setattr(pipeline,'SUMMARY',tmp_path/'runs/pilot_summary.json')
    monkeypatch.setattr(pipeline,'require_submission',lambda:dict(source_hash='synthetic'))
    monkeypatch.setattr(pipeline.signal,'signal',lambda *args:None)
    p=lambda t:{k:tmp_path/t['mode']/k for k in ('result','lock','checkpoint','metadata','runtime')}
    monkeypatch.setattr(pipeline,'paths',p)
    study=copy.deepcopy(config.plan())
    if failure=='budget': study['budget']['minimum_start_remaining_seconds']=1000000
    monkeypatch.setattr(pipeline,'plan',lambda:study)
    calls=[]
    def child(module,args,directory,deadline):
        kind=module.rsplit('.',1)[-1]
        calls.append((kind,list(args)))
        if kind==failure: raise RuntimeError('synthetic '+kind+' failure')
        if kind=='run':
            task=next(t for t in study['tasks'] if t['mode']==args[-1])
            provenance.atomic_json(p(task)['result'],dict(**task,status='PASS',test_evaluation_count=0))
    monkeypatch.setattr(pipeline,'run_child',child)
    if failure:
        with pytest.raises(SystemExit,match='1'): pipeline.main()
    else:
        pipeline.main()
    assert calls[:2]==[('preflight',[]),('gpu_checks',[])]
    assert calls[-1]==('report',[])
    fits=[args[-1] for kind,args in calls if kind=='run']
    assert fits==(list(config.MODES) if failure is None else ['separate_replay'] if failure=='run' else [])
    statuses=[json.loads(p(t)['result'].read_text())['status'] for t in study['tasks']]
    expected=['PASS']*4 if failure is None else ['FAIL']+['NOT_RUN']*3 if failure=='run' else ['NOT_RUN']*4
    assert statuses==expected
