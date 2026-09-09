import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import math
import itertools
from collections import defaultdict
import numpy as np
import pytest
import torch
from torch import nn
from experiments.target_combination_analysis import common as c
from experiments.target_combination_analysis.summarize import build,factorial,markdown
from experiments.target_combination_analysis.safety import DataAccessGuard


def stage_functions():
    names={'rank_loss','active_auxiliary_losses','multitask_loss','head_prefix','optimizer_for_model','shared_backbone_parameters','flatten_gradients','rng_snapshot','restore_rng','cosine','gradient_diagnostic'}
    tree=ast.parse((c.ROOT/'experiments/stage3_auxiliary_analysis/run.py').read_text())
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    namespace={'torch':torch,'nn':nn,'CURRENT_AUX_TARGETS':c.TARGETS,'math':math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'unchanged_stage3_functions','exec'),namespace)
    return SimpleNamespace(**namespace)

class Model(nn.Module):
    POS_ITEM_ID='item_id'
    def __init__(self):
        super().__init__();self.backbone=nn.Sequential(nn.Linear(3,4),nn.Dropout(.2));self.rank=nn.Linear(4,5);self.loss_fct=nn.CrossEntropyLoss()
        for name in ['click_head','long_view_head','like_head','profile_enter_head']:setattr(self,name,nn.Linear(4,1))
    def shared_representation(self,x):return self.backbone(x['history'])
    def ranking_logits_from_representation(self,h):return self.rank(h)
    def auxiliary_logits_from_representation(self,h):return {t:getattr(self,stage_functions().head_prefix(t)[:-1])(h).squeeze(-1) for t in c.TARGETS}


def test_manifest_and_frozen_sources():
    assert c.frozen_checks()['historical_files_checked']>50
    bad=deepcopy(c.combinations());bad[15]=bad[14]
    with pytest.raises(AssertionError):c.check_manifest(bad)

@pytest.mark.parametrize('cell',c.combinations(),ids=lambda x:x['combination_id'])
def test_loss_and_active_optimizer(cell):
    f=stage_functions();torch.manual_seed(2026);model=Model();model.eval()
    x={'history':torch.randn(8,3),'item_id':torch.arange(8)%5,**{t:torch.arange(8)%2 for t in c.TARGETS}}
    active=cell['active_targets'];w=c.weights(active);pos={t:1.5 for t in active}
    loss,rank,aux=f.multitask_loss(model,x,active,.13,w,pos)
    expected=rank+.13*torch.stack(list(aux.values())).mean() if active else rank
    assert torch.allclose(loss,expected)
    assert all(v==1/len(active) for v in w.values())
    opt=f.optimizer_for_model(model,active,.01,.02,.001)
    before={n:p.clone() for n,p in model.named_parameters()};loss.backward();opt.step()
    for t in c.TARGETS:
        assert any(not torch.equal(p,before[n]) for n,p in model.named_parameters() if n.startswith(f.head_prefix(t)))==(t in active)
    assert not torch.equal(model.backbone[0].weight,before['backbone.0.weight'])


def test_diagnostic_restores_rng_and_grads():
    f=stage_functions();torch.manual_seed(2026);m=Model();m.train();x={'history':torch.randn(8,3),'item_id':torch.arange(8)%5,**{t:torch.arange(8)%2 for t in c.TARGETS}}
    state=torch.get_rng_state().clone();r=f.gradient_diagnostic(m,x,c.TARGETS,{t:1. for t in c.TARGETS})
    assert torch.equal(state,torch.get_rng_state()) and all(p.grad is None for p in m.parameters())
    assert len(r['pairwise_auxiliary'])==6
    assert all(math.isfinite(v['norm']) for v in r['tasks'].values())


def test_known_factorial_fixture():
    values={frozenset(cell['active_targets']):.1+.01*('is_click' in cell['active_targets'])+.02*('long_view' in cell['active_targets'])+.04*({'is_click','long_view'}<=set(cell['active_targets'])) for cell in c.combinations()}
    marginal,pairs=factorial(values)
    assert [r['count'] for r in marginal]==[8]*4
    assert marginal[0]['mean']==pytest.approx(.03) and marginal[1]['mean']==pytest.approx(.04)
    assert pairs[0]['mean']==pytest.approx(.04) and all(r['count']==4 for r in pairs)
    assert all(abs(r['mean'])<1e-12 for r in pairs[1:])


def test_summary_16_partial_and_mixed_sha():
    cfg=c.config();cells=c.combinations();hashes={'source_digest':'s','config':'c','combinations':'m'};runs={}
    for cell in cells:
        runs[cell['run_id']]={**cell,'status':'completed','scientific_run':True,'git_commit':'a','fingerprint':cfg['fingerprint'],'evaluation_split':'validation','test_usage':'forbidden','test_evaluation_count':0,'source_digest':'s','config_sha256':'c','combinations_sha256':'m','loss_weight_mode':'uniform_normalized_aux','auxiliary_loss_weights':c.weights(cell['active_targets']),'lambda_aux':cfg['optimization']['lambda_aux'],'seed':2026,'gates':{'passed':True},'ranking_metrics':{m:.1+cell['array_index']*.001 for m in c.METRICS},'best_epoch':2,'runtime_sec':10}
    s=build(cells,runs,'a',cfg,hashes);assert s['completeness']=='16/16';assert s['ranking'][0]['array_index']==15
    assert len(s['canonical_order'])==16 and len(s['best_by_aux_count'])==5
    json.dumps(s,allow_nan=False);assert '16/16' in markdown(s)
    runs[cells[0]['run_id']]['git_commit']='bad';runs.pop(cells[1]['run_id']);runs[cells[2]['run_id']]['status']='failed'
    s=build(cells,runs,'a',cfg,hashes);assert s['completeness']=='13/16'
    assert s['canonical_order'][0]['status']=='invalid_provenance_or_metrics'
    assert all(r['delta_NDCG10_vs_primary'] is None for r in s['ranking'])


def test_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'HERE',tmp_path);c.reserve('run')
    with pytest.raises(FileExistsError):c.reserve('run')
    (tmp_path/'runs').mkdir();(tmp_path/'runs/another.json').write_text('{}')
    with pytest.raises(FileExistsError):c.reserve('another')


def test_test_and_shared_write_guard(tmp_path):
    g=DataAccessGuard([tmp_path/'shared']);(tmp_path/'shared').mkdir()
    with pytest.raises(RuntimeError):g.check_path(tmp_path/'test.parquet')
    link=tmp_path/'alias.parquet';link.symlink_to(tmp_path/'test.parquet')
    with pytest.raises(RuntimeError):g.check_path(link)
    with pytest.raises(RuntimeError):g.check_path(tmp_path/'shared/train.inter',True)
    with pytest.raises(RuntimeError):g.loader_factory(lambda *a:None)({},'test')
    assert g.loader_factory(lambda *a:'ok')({},'valid')=='ok'


def test_no_label_input_and_sequence_prefix():
    tree=ast.parse((c.ROOT/'experiments/multitask_tim4rec/model.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef))
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='shared_representation')
    text=ast.unparse(method)
    assert all(t not in text for t in c.TARGETS)
    from experiments.multitask_tim4rec_optuna.prepare_validation_only import sequential_rows
    rows=[dict(user_id=1,item_id=i,timestamp=i,source_row_id=i) for i in range(65)]
    train,valid=sequential_rows(rows,dict(user_id=1,item_id=65,timestamp=65,source_row_id=65))
    assert valid['item_id_list']==list(range(15,65))
    assert all(r['item_id'] not in r['item_id_list'] for r in train)
