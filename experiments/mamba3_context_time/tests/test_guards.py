import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.mamba3_context_time.config import HERE, ROOT, CORE, paths, plan, settings, scientific_settings
from experiments.mamba3_context_time.provenance import acquire, atomic_json, rng_hash, save_state_dict, verify
from experiments.mamba3_context_time.gpu_checks import compare
from experiments.mamba3_context_time.report import summarize, METRICS


def test_frozen_hashes_config_and_unique_paths():
    assert verify()['source_hash']
    historical=json.loads((ROOT/plan()['historical_reference']['source_json']).read_text())
    for task in plan()['tasks']:
        assert scientific_settings(settings(task))==scientific_settings(historical['config'])
    for key in paths(plan()['tasks'][0]):
        assert len({paths(t)[key] for t in plan()['tasks']})==5
    with pytest.raises(ValueError): paths(dict(mode='routed',seed=42,run_id='../unsafe'))


def test_locks_and_safe_checkpoint(tmp_path):
    p=dict(result=tmp_path/'result.json',checkpoint=tmp_path/'best.pth',metadata=tmp_path/'metadata.json',lock=tmp_path/'run.lock')
    acquire(p,{'run_id':'fixture'})
    with pytest.raises(FileExistsError): acquire(p,{})
    model=torch.nn.Linear(3,2)
    save_state_dict(model,p['checkpoint'])
    state=torch.load(p['checkpoint'],weights_only=True,map_location='cpu')
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    atomic_json(p['result'],{'status':'FAIL'})
    with pytest.raises(FileExistsError): acquire(p,{})
    with pytest.raises(ValueError): atomic_json(tmp_path/'bad.json',{'x':float('nan')})


def test_rng_isolation_and_real_error_reporting():
    from experiments.mamba3_context_time.temporal import ContextTime
    from experiments.mamba3_time_mechanisms.time_mechanisms import TimeMechanisms
    torch.manual_seed(2026); _=TimeMechanisms('separate'); expected=rng_hash()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026); _=ContextTime('routed')
    assert rng_hash()==expected
    result=compare(torch.ones(3),torch.ones(3)+1e-7)
    assert result['passed'] and result['max_abs_error']>0
    assert not compare(torch.ones(3),torch.full((3,),float('nan')))['passed']


def test_actual_wrapper_causality_and_two_layer_scale_reuse():
    from experiments.mamba3_context_time.temporal import ContextTime
    from experiments.mamba3_timeaware.time_inputs import history_gaps
    source=ast.parse((HERE/'model.py').read_text())
    cls=next(n for n in source.body if isinstance(n,ast.ClassDef))
    forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    calls=[]
    def scan(mixer,x,d,s):
        calls.append((d,s)); return x*(d.mean(-1,keepdim=True)+s.mean(-1,keepdim=True))
    ns=dict(torch=torch,history_gaps=history_gaps,mechanism_mamba3_forward=scan)
    exec(compile(ast.Module(body=[forward],type_ignores=[]),'CPU wiring double, not GPU evidence','exec'),ns)
    fake=SimpleNamespace(context_mode='routed',mechanisms=ContextTime('routed'),diagnostic_collector=None,
        item_embedding=torch.nn.Embedding(20,64),input_norm=torch.nn.Identity(),input_dropout=torch.nn.Identity(),
        output_norm=torch.nn.Identity(),gather_indexes=lambda x,i:x[torch.arange(len(i)),i],
        layers=[SimpleNamespace(norm1=torch.nn.Identity(),mixer=None,mixer_dtype=torch.float32,
                                dropout1=torch.nn.Identity(),use_ffn=False) for _ in range(2)])
    items=torch.tensor([[1,2,0]]); lengths=torch.tensor([2]); times=torch.tensor([[1e12,1e12,float('nan')]],dtype=torch.float64)
    result=ns['forward'](fake,items,lengths,times)
    assert result.shape==(1,64) and calls[0][0] is calls[1][0] and calls[0][1] is calls[1][1]
    with pytest.raises(ValueError): ns['forward'](fake,torch.tensor([[1,0,2]]),lengths,times)
    # Execute the inherited encoder entry point: changing targets cannot enter forward.
    tree=ast.parse((ROOT/'experiments/mamba3_timeaware/model.py').read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_encode')
    namespace={};exec(compile(ast.Module(body=[method],type_ignores=[]),'inherited encoder','exec'),namespace)
    fake.ITEM_SEQ='items'; fake.ITEM_SEQ_LEN='lengths'; fake.time_sequence_field='times'
    fake.forward=lambda i,l,t:ns['forward'](fake,i,l,t)
    interaction=dict(items=items,lengths=lengths,times=times,target_item=1,target_time=10)
    a=namespace['_encode'](fake,interaction)
    interaction.update(target_item=999,target_time=1e20)
    assert torch.equal(a,namespace['_encode'](fake,interaction))


def test_no_test_loader_no_unsafe_deserialization_or_hidden_submission():
    source=(HERE/'run.py').read_text(); tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='FullSortEvalDataLoader']
    assert len(calls)==1 and ast.unparse(calls[0].args[1])=='valid_ds'
    assert 'del reserved' in source
    for p in HERE.glob('*.py'):
        t=ast.parse(p.read_text())
        for n in ast.walk(t):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load' and ast.unparse(n.func.value)=='torch':
                assert any(k.arg=='weights_only' and isinstance(k.value,ast.Constant) and k.value.value is True for k in n.keywords)
        assert 'weights_only=False' not in p.read_text() and 'pickle.load' not in p.read_text()
        if p.name!='submit.py': assert "'sbatch'" not in p.read_text()
    launch=(ROOT/'slurm/mamba3_context_time.sh').read_text()
    assert '#SBATCH --time=08:00:00' in launch and '#SBATCH --no-requeue' in launch
    assert '--array' not in launch and 'git ' not in launch and 'sbatch ' not in launch


@pytest.mark.parametrize('answer', ('ok','failed','timeout'))
def test_one_durable_submit_no_retry(tmp_path,monkeypatch,answer):
    from experiments.mamba3_context_time import submit
    monkeypatch.setattr(submit,'RUNTIME',tmp_path)
    monkeypatch.setattr(submit,'verify',lambda:dict(source_hash='fixture'))
    monkeypatch.setattr(submit.subprocess,'check_output',lambda args,**kw:
        'a'*40+'\n' if args[1]=='rev-parse' else ('exp/mamba3-context-time\n' if args[1]=='branch' else ''))
    monkeypatch.setenv('RUN_COMMIT','a'*40);monkeypatch.setenv('EXPECTED_CORE_HASH',CORE);monkeypatch.setenv('EXPECTED_STUDY_HASH','fixture')
    calls=[]
    def execute(*args,**kw):
        assert json.loads((tmp_path/'submission_001.json').read_text())['status']=='PREPARED'
        calls.append(args)
        if answer=='timeout': raise TimeoutError('synthetic lost response')
        return SimpleNamespace(returncode=0 if answer=='ok' else 1,stdout='12345\n',stderr='')
    if answer=='ok': submit.submit_once(execute)
    else:
        with pytest.raises((RuntimeError,TimeoutError)):submit.submit_once(execute)
    with pytest.raises(FileExistsError):submit.submit_once(execute)
    assert len(calls)==1


def records():
    counts=[610572,611214,611284,610968,611232]
    rows={}
    for task,count,value in zip(plan()['tasks'],counts,[.06,.061,.062,.059,.0605]):
        metrics={k:value for k in METRICS}
        history=[dict(epoch=i,valid_ndcg10=value,valid_metrics=metrics,diagnostics={},train_loss=1.,train_seconds=1.,valid_seconds=2.) for i in range(2)]
        rows[task['mode']]=dict(**task,study_id='mamba3_context_time_001',status='PASS',core_hash=CORE,source_hash='fixture',
            selection_split='VALID',evaluation_mode='full-ranking',test_evaluation_count=0,config=settings(task),
            parameter_count=count,history=history,actual_epochs=2,best_epoch=1,best_valid_score=value,best_valid_metrics=metrics,best_diagnostics={})
        from experiments.mamba3_time_confirmation.config import MANIFEST_SHA,STATS_SHA
        reference=json.loads((ROOT/plan()['historical_reference']['source_json']).read_text())
        rows[task['mode']].update(manifest_sha256=MANIFEST_SHA,train_time_stats_sha256=STATS_SHA,protocol=reference['protocol'])
    return rows


def test_summary_ties_negative_results_and_partial_no_winner():
    r=records(); result=summarize(r,'fixture')
    assert result['status']=='COMPLETE' and result['comparisons']['routed_minus_dense12']<0
    r['dense12']['status']='FAIL'
    partial=summarize(r,'fixture')
    assert partial['status']=='INCOMPLETE' and partial['comparisons']=={}
    r=records(); r['dense11']['best_epoch']=0
    with pytest.raises(ValueError): summarize(r,'fixture')
    r=records(); r['uniform']['test_evaluation_count']=1
    with pytest.raises(ValueError): summarize(r,'fixture')


def test_checkpoint_callback_keeps_last_equal_maximum_and_evaluation_guard(tmp_path):
    from experiments.mamba3_context_time.trainer import trainer_class
    class Base:
        def __init__(self):
            self.model=torch.nn.Linear(3,2)
            self.best_valid_score=.06
        def evaluate(self,*args,**kwargs):
            return {'ndcg@10':.06}
    record=dict(mode='separate_replay',seed=2026,run_id='fixture',execution_commit='a'*40,
                config_sha256='config',source_hash='source',core_hash=CORE,history=[])
    p=dict(result=tmp_path/'result.json',checkpoint=tmp_path/'checkpoint.pth',metadata=tmp_path/'metadata.json')
    loader=object();trainer=trainer_class(Base,loader,record,p)()
    for epoch in (0,1):
        record['history'].append(dict(epoch=epoch,valid_ndcg10=.06,valid_metrics={'ndcg@10':.06},diagnostics={'epoch':epoch}))
        trainer._save_checkpoint(epoch)
    assert record['best_epoch']==1 and record['best_diagnostics']=={'epoch':1}
    assert json.loads(p['metadata'].read_text())['epoch']==1
    assert trainer.evaluate(loader)=={'ndcg@10':.06}
    with pytest.raises(ValueError):trainer.evaluate(object())
    with pytest.raises(ValueError):trainer.evaluate(loader,load_best_model=True)
    with pytest.raises(ValueError):trainer.evaluate(loader,model_file='elsewhere')


@pytest.mark.parametrize('failed_stage', ('gpu_checks','separate_replay','dense11'))
def test_pipeline_stops_after_technical_error_and_reports_partial(tmp_path,monkeypatch,failed_stage):
    from experiments.mamba3_context_time import pipeline
    monkeypatch.setattr(pipeline,'RUNTIME',tmp_path/'runtime')
    monkeypatch.setattr(pipeline,'HERE',tmp_path)
    monkeypatch.setattr(pipeline,'GPU_EVIDENCE',tmp_path/'runs/gpu.json')
    monkeypatch.setattr(pipeline,'SUMMARY',tmp_path/'runs/pilot_summary.json')
    monkeypatch.setattr(pipeline,'require_submission',lambda:dict(source_hash='fixture'))
    monkeypatch.setenv('SLURM_JOB_ID','fixture');monkeypatch.setenv('RUN_COMMIT','a'*40)
    tasks=plan()['tasks']
    def local_paths(task):
        runtime=tmp_path/task['mode']
        return dict(runtime=runtime,result=tmp_path/'runs'/(task['mode']+'.json'),lock=runtime/'run.lock',
                    checkpoint=runtime/'best.pth',metadata=runtime/'best.json')
    monkeypatch.setattr(pipeline,'paths',local_paths)
    calls=[]
    def child(module,args,stage_dir,deadline):
        stage=args[-1] if args else module.rsplit('.',1)[-1]
        calls.append(stage)
        if stage==failed_stage:raise RuntimeError('injected technical failure')
        if stage in [t['mode'] for t in tasks]:
            task=next(t for t in tasks if t['mode']==stage)
            atomic_json(local_paths(task)['result'],dict(**task,status='PASS'))
    monkeypatch.setattr(pipeline,'run_child',child)
    # Do not install signal handlers in the pytest process.
    monkeypatch.setattr(pipeline.signal,'signal',lambda *args:None)
    with pytest.raises(SystemExit):pipeline.main()
    assert calls[-1]=='report'
    ordered=['preflight','gpu_checks']+[t['mode'] for t in tasks]
    assert calls[:-1]==ordered[:ordered.index(failed_stage)+1]
    future=[t for t in tasks if ordered.index(t['mode'])>ordered.index(failed_stage)]
    assert all(json.loads(local_paths(t)['result'].read_text())['status']=='NOT_RUN' for t in future)
    assert json.loads((tmp_path/'runtime/pipeline_status.json').read_text())['status']=='FAIL'
