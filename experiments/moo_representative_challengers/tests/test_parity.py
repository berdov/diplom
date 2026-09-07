"""Numerical comparisons execute pinned official cores, not another local port."""
import ast
import importlib.util
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
from experiments.moo_representative_challengers.references import reference_path, HERE
from experiments.moo_representative_challengers.methods import ferero, most, phn_hvi

TOL = yaml.safe_load((HERE / 'config.yaml').read_text())['parity']


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('tasks', [2, 5])
@pytest.mark.parametrize('seed', [2026, 17, 88])
def test_ferero_official_coefficients_direction_constraints(tasks, seed):
    official = module_at('ferero_reference', reference_path('ferero', 'toy_experiments/solvers/min_norm_solvers_numpy.py'))
    rng = np.random.default_rng(seed)
    grads = rng.normal(size=(tasks, 11)) * .04
    losses = rng.uniform(.2, 1., tasks)
    r = np.full(tasks, .4 / (tasks - 1)); r[0] = .6
    b = ferero.ray_matrix(r)
    v = 1 / r; v /= np.linalg.norm(v)
    np.testing.assert_allclose(b @ v, 0, atol=1e-16)
    h = b @ losses
    ref_c, ref_d = official.MinNormSolver.find_min_norm_element_PGD_H(grads, np.eye(tasks), b, h)
    ours, state = ferero.solve(losses, grads @ grads.T, r)
    np.testing.assert_allclose(ours, ref_c, atol=TOL['ferero_atol'], rtol=TOL['ferero_rtol'])
    np.testing.assert_allclose(ours @ grads, ref_d, atol=TOL['ferero_atol'], rtol=TOL['ferero_rtol'])
    np.testing.assert_allclose(ours @ losses, ref_c @ losses, atol=TOL['ferero_atol'])
    np.testing.assert_allclose(state['constraint_state'], h)
    assert state['effective_step_size'] == .1
    assert not np.allclose(ours, np.ones(tasks) / tasks)


def test_ferero_spectral_cap_large_gradient_no_fallback():
    grads = np.eye(5) * 100
    c, state = ferero.solve(np.arange(1, 6), grads @ grads.T, [.6,.1,.1,.1,.1])
    assert np.isfinite(c).all() and state['effective_step_size'] < .1
    assert abs(sum(state['dual_f']) - 1) < 1e-10


def official_most(monkeypatch):
    for p in ['copsolver/copsolver.py','copsolver/frank_wolfe_solver.py',
              'commondescentvector/common_descent_vector.py','commondescentvector/multi_objective_cdv.py','ipot.py']:
        reference_path('most', p)
    root = reference_path('most', 'update.py').parent
    monkeypatch.syspath_prepend(str(root))
    from copsolver.frank_wolfe_solver import FrankWolfeSolver
    from commondescentvector.multi_objective_cdv import MultiObjectiveCDV
    # Extract unmodified AST class nodes to avoid importing application/data code.
    source = ast.parse((root / 'update.py').read_text())
    tree = ast.Module(body=[n for n in source.body if isinstance(n, ast.ClassDef) and n.name in {'GlobalUpdate','Moving_Average'}], type_ignores=[])
    env = dict(np=np, torch=torch, Module=torch.nn.Module, F=torch.nn.functional,
               FrankWolfeSolver=FrankWolfeSolver, MultiObjectiveCDV=MultiObjectiveCDV,
               ipot=module_at('ipot_reference', root / 'ipot.py'), time=__import__('time'))
    exec(compile(tree, str(root / 'update.py'), 'exec'), env)
    return env


@pytest.mark.parametrize('mgda', [False, True])
def test_most_official_complete_update(monkeypatch, mgda):
    ref = official_most(monkeypatch)
    rng = np.random.default_rng(2026)
    gradients = rng.normal(0, .02, (3, 5, 13)).astype(np.float32)
    # Dense assignments also exercise MGDA on CPU (upstream single-active path hardcodes CUDA).
    losses = np.tile([.91, 1.02, 1.13], (5, 1)).astype(np.float32)
    if not mgda:
        losses += rng.uniform(0, .1, losses.shape).astype(np.float32)
    args = SimpleNamespace(gpu=None,device='cpu',num_users=5,num_model=3,
        frank_wolfe_max_iter=20,normalize_gradients=True,diversity_reg=0.,
        MGDA_fast_mode=False,topk_model_ratio=.1,adjust_ab=1,warmup_epochs=0,
        ot_skip=1,ot_algo_version='default',ot_ma=0,set_objective_treshold=1.,mask_threshold=1e-8)
    average = 100. if mgda else 0.
    reference = ref['GlobalUpdate'](args).update_weights(
        [[torch.tensor(g) for g in model] for model in gradients], torch.tensor(losses),
        average, 1., iters=20, obj_weights_average=ref['Moving_Average'](args), epoch=1)
    core = most.MosT(); core.average_objective = average
    coeff, state = core.solve(losses, gradients @ gradients.transpose(0,2,1))
    atol, rtol = TOL['most_atol'], TOL['most_rtol']
    np.testing.assert_allclose(state['transport'], reference[3].numpy(), atol=atol, rtol=rtol)
    for i in range(3):
        np.testing.assert_allclose(coeff[:, i] @ gradients[i], reference[0][i].numpy(), atol=atol, rtol=rtol)
        if mgda:
            np.testing.assert_allclose(coeff[:, i], reference[1][i], atol=atol, rtol=rtol)
    np.testing.assert_allclose(state['set_objective'], reference[2].item(), atol=atol, rtol=rtol)
    assert state['mgda_active'] is mgda
    other = most.MosT(); other.average_objective = average
    np.testing.assert_array_equal(coeff, other.solve(losses, gradients @ gradients.transpose(0,2,1))[0])


def official_hvi(monkeypatch):
    for p in ['Jura/functions_hv_grad_3d.py','Jura/functions_hv_python3.py','Jura/functions_evaluation.py']:
        reference_path('phn_hvi', p)
    path = reference_path('phn_hvi', 'Jura/phn/solvers.py')
    monkeypatch.syspath_prepend(str(path.parents[1]))
    # Compatibility alias only; reference math is not edited.
    monkeypatch.setattr(np, 'bool', np.bool_, raising=False)
    return module_at('hvi_reference', path)


@pytest.mark.parametrize('case', ['random','duplicates','dominated','ties'])
def test_hvi_five_dimensions_official_weights_objective_backward(monkeypatch, case):
    official = official_hvi(monkeypatch)
    rng = np.random.default_rng(2026)
    points = rng.uniform(.2, 1.6, (8, 5)).astype(np.float32)
    if case == 'duplicates': points[3] = points[0]
    if case == 'dominated': points[1] = points[0] + .2
    if case == 'ties': points[0,0] = points[2,0]
    base = [1.5]*5
    weights, state = phn_hvi.hv_weights(points, base)
    ref = official.HvMaximization(8, 5, base).compute_weights(points.T).numpy().T
    np.testing.assert_allclose(weights, ref, atol=TOL['phn_hvi_atol'], rtol=TOL['phn_hvi_rtol'])
    preferences = phn_hvi.preference_samples(rng)
    a = torch.tensor(points, requires_grad=True); b = a.detach().clone().requires_grad_()
    ours = phn_hvi.surrogate(a, preferences, weights)
    penalty = list(torch.nn.functional.cosine_similarity(b, torch.tensor(preferences), dim=-1))
    reference = official.MultiHead(8, 5, base).get_weighted_loss(
        points.T[None], 'cpu', [v[:,None] for v in b], 8, penalty, .001)
    np.testing.assert_allclose(ours.detach(), reference.detach(), atol=3e-6, rtol=2e-5)
    ours.backward(); reference.backward()
    np.testing.assert_allclose(a.grad, b.grad, atol=2e-6, rtol=2e-5)
    from functions_hv_python3 import HyperVolume
    np.testing.assert_allclose(state['hypervolume'], HyperVolume(tuple(state['dynamic_reference'])).compute(points.tolist()), atol=1e-6)
    assert torch.isfinite(a.grad).all() and a.grad.norm() > 0


def test_hvi_cross_sample_interaction_and_dirichlet():
    rng = np.random.default_rng(2026)
    samples = phn_hvi.preference_samples(rng)
    expected = np.random.default_rng(2026).dirichlet([.2]*5, size=8).astype(np.float32)
    np.testing.assert_array_equal(samples, expected)
    points = np.array([[.2,.8,.3,.7,.4],[.6,.4,.7,.3,.8]])
    joint = phn_hvi.hv_weights(points, [1.5]*5)[0][0]
    solo = phn_hvi.hv_weights(points[:1], [1.5]*5)[0][0]
    assert not np.allclose(joint, solo)
