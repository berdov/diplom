import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.mamba3_time_confirmation.aggregate import METRICS, aggregate, validate
from experiments.mamba3_time_confirmation.config import (CORE, COUNTS, HERE, ROOT, paths,
                                                        plan, scientific_settings, settings)
from experiments.mamba3_time_confirmation.inputs import constant_timestamps
from experiments.mamba3_time_confirmation.provenance import acquire, atomic_json, verify
from experiments.mamba3_time_mechanisms.time_mechanisms import TimeMechanisms
from experiments.mamba3_timeaware.config import load_config
from experiments.mamba3_timeaware.time_inputs import TimeCalibrator, history_gaps


@pytest.mark.parametrize('task', plan()['tasks'])
def test_config_parity_and_unique_paths(task):
    config=settings(task)
    assert scientific_settings(config)==scientific_settings(load_config())
    assert config['seed']==task['seed']
    assert config.get('time_mechanism_mode') == (None if task['mode']=='shared' else 'separate')
    for key in paths(task):
        assert len({str(paths(t)[key]) for t in plan()['tasks']}) == 9
    assert config['epochs']==300 and config['stopping_step']==10


def test_reused_2026_and_source_guards():
    assert verify()['source_hash']
    assert [(t['mode'],t['seed']) for t in plan()['reuse']]==[('shared',2026),('separate',2026)]
    assert plan()['tasks'][8]['mode']=='separate_constant_gap'


@pytest.mark.parametrize('seed',range(2026,2031))
def test_temporal_initialization_counts_and_rng_parity(seed):
    torch.manual_seed(seed)
    shared=TimeCalibrator(2,838393)
    shared_rng=torch.random.get_rng_state()
    torch.manual_seed(seed)
    separate=TimeMechanisms('separate')
    assert torch.equal(shared_rng,torch.random.get_rng_state())
    for branch in separate.calibrators.values():
        for key,value in shared.state_dict().items():
            assert torch.equal(value,branch.state_dict()[key])
    frozen=json.loads((ROOT/'experiments/mamba3_baseline/runs/mamba3_validation_001.json').read_text())
    base=frozen['model']['total_parameters']
    assert base+sum(p.numel() for p in shared.parameters())==COUNTS['shared']
    assert base+sum(p.numel() for p in separate.parameters())==COUNTS['separate_constant_gap']
    # Both production constructors initialize the exact same backbone before temporal modules.
    source=(ROOT/'experiments/mamba3_time_mechanisms/model.py').read_text()
    assert source.index('Mamba3Rec.__init__(self, config, dataset)') < source.index('self.mechanisms =')
    source=(ROOT/'experiments/mamba3_timeaware/model.py').read_text()
    assert source.index('super().__init__(config, dataset)') < source.index('self.time_calibrator =')


def control_wrapper():
    # Execute the actual thin input wrapper with an explicit CPU receiver, not a fake GPU PASS.
    class Receiver(nn.Module):
        def __init__(self):
            super().__init__()
            self.mechanisms=TimeMechanisms('separate')
        def forward(self,items,lengths,times):
            self.received=times
            return self.mechanisms(*history_gaps(times,items!=0))
    tree=ast.parse((HERE/'model.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    namespace=dict(MechanismMamba3Rec=Receiver,constant_timestamps=constant_timestamps)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'actual_control_wrapper','exec'),namespace)
    return namespace['ConstantGapMamba3Rec']()


def test_constant_invariance_neutrality_zero_gap_and_learnability():
    model=control_wrapper()
    items=torch.tensor([[1,2,3,0],[1,0,0,0]])
    lengths=torch.tensor([3,1])
    first=torch.tensor([[1e12,1e12,1e12+1e8,0],[1e12,0,0,0]],dtype=torch.float64)
    other=torch.tensor([[5.,7.,99.,0.],[4.,0.,0.,0.]],dtype=torch.float64)
    initial=model(items,lengths,first)
    assert all(torch.equal(v,torch.ones_like(v)) for v in initial)
    gaps,active=history_gaps(model.received,items!=0)
    assert gaps[active].tolist()==[838393.,838393.]
    assert active.tolist()==[[False,True,True,False],[False,False,False,False]]
    before=[c.last.weight.detach().clone() for c in model.mechanisms.calibrators.values()]
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    (initial[0][active].square().sum()+2*initial[1][active].square().sum()).backward()
    for c in model.mechanisms.calibrators.values():
        assert torch.isfinite(c.last.weight.grad).all() and c.last.weight.grad.abs().sum()>0
        assert torch.equal(c.first.weight.grad,torch.zeros_like(c.first.weight.grad))
    opt.step()
    for old,c in zip(before,model.mechanisms.calibrators.values()):
        assert not torch.equal(old,c.last.weight)
    # Make the two paths intentionally distinct, without any time-derived information.
    with torch.no_grad():model.mechanisms.calibrators['scan'].last.bias.add_(.2)
    x=model(items,lengths,first);y=model(items,lengths,other)
    assert all(torch.equal(a,b) for a,b in zip(x,y))
    assert not torch.equal(x[0],x[1])
    assert all(torch.equal(v[~active],torch.ones_like(v[~active])) for v in x)


def test_bad_padding_rejected():
    with pytest.raises(ValueError):constant_timestamps(torch.tensor([[1,0,2]]),torch.tensor([2]))


def test_exclusive_lock_atomic_results_and_no_retry(tmp_path):
    p=dict(result=tmp_path/'runs/result.json',runtime=tmp_path/'runtime',lock=tmp_path/'runtime/run.lock')
    acquire(p,{'run_id':'one'})
    with pytest.raises(FileExistsError):acquire(p,{'run_id':'two'})
    atomic_json(p['result'],dict(status='FAIL'))
    assert json.loads(p['result'].read_text())['status']=='FAIL'
    with pytest.raises(FileExistsError):acquire(p,{'run_id':'retry'})
    assert not p['result'].with_name('result.json.tmp').exists()
    with pytest.raises(ValueError):atomic_json(tmp_path/'nan.json',dict(metric=float('nan')))


def records(seeds=(2026,2027)):
    rows=[];histories={}
    for seed in seeds:
        for mode in ('shared','separate'):
            value=.06 + (.001 if seed==2026 else -.002)*(mode=='separate')
            run_id=f'{mode}_{seed}'
            rows.append(dict(run_id=run_id,seed=seed,mode=mode,status='PASS',selection_split='VALID',
                test_evaluation_count=0,actual_epochs=2,best_valid_metrics={k:value for k in METRICS}))
            histories[run_id]=[dict(epoch=0,valid_ndcg10=value-.01),dict(epoch=1,valid_ndcg10=value)]
    return rows,histories


def test_matched_aggregation_keeps_negative_results_and_incomplete_flag():
    rows,history=records();r=aggregate(rows,history)
    assert (r['n_available'],r['n_expected'],r['incomplete_study'])==(2,5,True)
    ndcg=r['full_horizon']['ndcg@10']
    assert ndcg['positive']==1 and ndcg['negative']==1
    assert ndcg['paired_delta']['mean']==pytest.approx(-.0005)
    assert ndcg['paired_delta']['sample_std']==pytest.approx(.003/2**.5)
    assert r['first27']['available']
    history.pop('shared_2026');assert not aggregate(rows,history)['first27']['available']
    rows.pop();assert aggregate(rows,history)['n_available']==1


@pytest.mark.parametrize('mutation', ['test','duplicate','unexpected_seed','infinite'])
def test_aggregation_rejects_invalid(mutation):
    rows,history=records()
    if mutation=='test':rows[0]['selection_split']='TEST'
    if mutation=='duplicate':rows.append(copy.deepcopy(rows[0]))
    if mutation=='unexpected_seed':rows[0]['seed']=42
    if mutation=='infinite':rows[0]['best_valid_metrics']['ndcg@10']=float('inf')
    with pytest.raises(ValueError):aggregate(rows,history)


def test_complete_five_pairs_and_separate_exploratory_control():
    rows,history=records(range(2026,2031))
    control=copy.deepcopy(rows[0]);control.update(mode='separate_constant_gap',run_id='control')
    r=aggregate(rows+[control],history)
    assert r['n_available']==5 and not r['incomplete_study']
    assert r['exploratory_control']['seed']==2026


def test_no_test_loader_and_smoke_reset_order():
    source=(HERE/'run.py').read_text();tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
           and n.func.id=='FullSortEvalDataLoader']
    assert len(calls)==1 and ast.unparse(calls[0].args[1])=='valid_ds'
    assert 'del reserved' in source and "if eval_data is not valid_data:" in source
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    ordered=ast.unparse(train)
    assert ordered.index('synthetic_smoke(task') < ordered.index('init_seed(config')
    assert ordered.index('init_seed(config') < ordered.index('dataset = PreciseHistoryDataset')
    assert ordered.index('train_data = TrainDataLoader') < ordered.index('model = cls(config, train_ds)')
    assert "torch.load(trainer.saved_model_file" in source  # After fit, metadata only.
    assert ordered.index('trainer.fit(') < ordered.index('torch.load(trainer.saved_model_file')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='resume_checkpoint'
                   for n in ast.walk(tree))
    launcher=(ROOT/'slurm/mamba3_time_confirmation.sh').read_text()
    assert '#SBATCH --array=0-8%1' in launcher and '#SBATCH --no-requeue' in launcher
    assert 'git ' not in launcher and 'sbatch ' not in launcher


@pytest.mark.parametrize('success',[True,False])
def test_single_submit_durable_record_and_no_automatic_retry(tmp_path,monkeypatch,success):
    from experiments.mamba3_time_confirmation import submit
    commit='a'*40
    monkeypatch.setattr(submit,'HERE',tmp_path)
    monkeypatch.setattr(submit,'verify',lambda:dict(source_hash='hash'))
    monkeypatch.setattr(submit,'paths',lambda task:dict(result=tmp_path/(task['run_id']+'.json'),
        lock=tmp_path/(task['run_id']+'.lock'),smoke_checkpoint=tmp_path/(task['run_id']+'.pth'),
        checkpoints=tmp_path/task['run_id']))
    monkeypatch.setattr(submit.subprocess,'check_output',lambda args,**kw:commit+'\n' if args[1]=='rev-parse' else '')
    monkeypatch.setenv('RUN_COMMIT',commit);monkeypatch.setenv('EXPECTED_CORE_HASH',CORE)
    monkeypatch.setenv('EXPECTED_STUDY_HASH','hash')
    calls=[]
    def fake(command,**kwargs):
        record=tmp_path/'slurm_logs/submission_001.json'
        assert json.loads(record.read_text())['status']=='PREPARED'
        calls.append(command)
        return SimpleNamespace(returncode=0 if success else 1,stdout='12345\n' if success else '',stderr='')
    if success:submit.submit_once(fake)
    else:
        with pytest.raises(SystemExit):submit.submit_once(fake)
    assert len(calls)==1
    record=json.loads((tmp_path/'slurm_logs/submission_001.json').read_text())
    assert record['status']==('SUBMITTED' if success else 'FAILED_OR_AMBIGUOUS_DO_NOT_RETRY')
    with pytest.raises(FileExistsError):submit.submit_once(fake)
    assert len(calls)==1
