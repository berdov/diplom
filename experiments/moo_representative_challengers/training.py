"""Small challenger loop; dataset/loss/evaluation code stays in the benchmark."""
import hashlib
import time
import numpy as np
import torch
from .methods import ferero, phn_hvi
from .safety import TASK_ORDER


def gradient_vectors(vector, parameters, *, keep_graph):
    rows = []
    for i, loss in enumerate(vector):
        grads = torch.autograd.grad(loss, parameters, retain_graph=keep_graph or i < len(vector)-1, allow_unused=True)
        rows.append(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(parameters,grads)]).detach())
    matrix = torch.stack(rows)
    if not torch.isfinite(matrix).all():
        raise FloatingPointError('Non-finite task gradients')
    return matrix


def assign_gradient(parameters, flat):
    offset = 0
    for p in parameters:
        p.grad = flat[offset:offset+p.numel()].view_as(p).clone()
        offset += p.numel()
    if offset != flat.numel():
        raise RuntimeError('Gradient/parameter length mismatch')


def parameter_hash(model):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def parameter_snapshot(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters()}


def parameter_group(name):
    if 'preference_hypernetwork' in name: return 'hypernetwork'
    if any(s in name for s in ['click_head','long_view_head','like_head','profile_enter_head']):
        return name.split('.')[0]
    return 'backbone'


def update_deltas(model, before):
    out = {}
    for n,p in model.named_parameters():
        group = parameter_group(n)
        out[group] = out.get(group,0.) + float((p.detach().cpu()-before[n]).double().square().sum())
    return {k:v**.5 for k,v in out.items()}


def checked_step(model, optimizer):
    sums = {}
    for name, p in model.named_parameters():
        if p.grad is None: continue
        if not torch.isfinite(p.grad).all():
            raise FloatingPointError(f'Non-finite gradient: {name}')
        group = parameter_group(name)
        sums[group] = sums.get(group,0) + p.grad.detach().float().square().sum()
    norms = {g:float(v.sqrt().cpu()) for g,v in sums.items()}
    if norms.get('backbone',0) <= 0:
        raise RuntimeError('Zero backbone gradient')
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('Non-finite parameters after update')
    return norms


def train_epoch(method, models, optimizers, train_data, sampled, pos_weights, loss_scales,
                method_config, preferences, state, rng, helpers, max_batches=None):
    start = time.monotonic()
    for model in models: model.train()
    sums = np.zeros(len(TASK_ORDER)); raw_sums = np.zeros(len(TASK_ORDER))
    scalar_sum = 0.; batches = 0; rows = 0; first_state = None; last_state = None
    gradient_max = [dict() for _ in models]

    def losses(model, interaction):
        return helpers.task_losses(model, interaction, sampled, pos_weights, loss_scales=loss_scales)['normalized_task_vector']

    def step(index):
        norms = checked_step(models[index], optimizers[index])
        for k,v in norms.items(): gradient_max[index][k] = max(gradient_max[index].get(k,0),v)

    for batch in train_data:
        interaction = helpers.interaction_from_batch(batch).to(next(models[0].parameters()).device)
        n = len(interaction)
        for optimizer in optimizers: optimizer.zero_grad(set_to_none=True)
        if method == 'ferero':
            points, diagnostics, scalars = [], [], []
            for i,(model,pref) in enumerate(zip(models,preferences)):
                vector = losses(model,interaction)
                params = [e.parameter for e in helpers.shared_parameter_entries(model,'all_backbone')]
                gradients = gradient_vectors(vector,params,keep_graph=True)
                gram = (gradients @ gradients.T).cpu().numpy(); del gradients
                coeff, info = ferero.solve(vector.detach().cpu().numpy(),gram,pref['weights'],**method_config['solver'])
                scalar = torch.dot(torch.as_tensor(coeff,device=vector.device,dtype=vector.dtype),vector)
                if not torch.isfinite(scalar): raise FloatingPointError('Non-finite FERERO scalar')
                scalar.backward(); step(i)
                points.append(vector.detach().cpu().numpy()); scalars.append(float(scalar.detach()))
                diagnostics.append(info)
            matrix = np.asarray(points); scalar = sum(scalars)
            last_state = {'solutions':diagnostics}
        elif method == 'most':
            points, gradients, params_list, grams = [], [], [], []
            for model in models:
                params = [p for p in model.parameters() if p.requires_grad]
                vector = losses(model,interaction)
                g = gradient_vectors(vector,params,keep_graph=False)
                points.append(vector.detach().cpu().numpy()); gradients.append(g); params_list.append(params)
                grams.append((g @ g.T).cpu().numpy())
            matrix = np.asarray(points)
            coeff, last_state = state.solve(matrix.T,np.asarray(grams))
            scalar = float((coeff.T * matrix).sum())
            for i,g in enumerate(gradients):
                assign_gradient(params_list[i],torch.as_tensor(coeff[:,i],dtype=g.dtype,device=g.device) @ g)
                step(i)
            del gradients, g
        else:
            rays = phn_hvi.preference_samples(rng,method_config['simultaneous_preferences'],method_config['dirichlet_alpha'])
            scalar, matrix, last_state = phn_hvi.replay_backward(
                models[0],rays,lambda:losses(models[0],interaction),method_config['base_reference'],method_config['cosine_weight'],
                inverse_priority=method_config['inverse_priority_cosine'])
            step(0)
        if not np.isfinite(matrix).all() or not np.isfinite(scalar):
            raise FloatingPointError('Non-finite training objectives')
        if first_state is None: first_state = last_state
        sums += matrix.mean(0)*n; raw_sums += matrix.mean(0)*np.asarray(loss_scales)*n
        scalar_sum += scalar*n; rows += n; batches += 1
        if max_batches is not None and batches >= max_batches: break
    if not batches: raise RuntimeError('Empty train loader')
    return {'batches':batches,'examples':rows,'normalized_task_means':(sums/rows).tolist(),
            'raw_task_means':(raw_sums/rows).tolist(),'moo_scalar':scalar_sum/rows,
            'gradient_norm_max_by_model':gradient_max,'first_method_state':first_state,
            'last_method_state':last_state,'runtime_seconds':time.monotonic()-start}


def evaluate(models, trainers, preferences, method, train_data, valid_data, config, helpers):
    from .common import select_operating_point
    from experiments.moo_8families.evaluation.pareto import pareto_summary
    records = []
    pairs = [(models[0],trainers[0],p,i) for i,p in enumerate(preferences)] if method=='phn_hvi' else [
        (m,t,preferences[i] if preferences else None,i) for i,(m,t) in enumerate(zip(models,trainers))]
    for model,trainer,pref,index in pairs:
        ranking = helpers.evaluate_validation_ranking(trainer=trainer,model=model,valid_data=valid_data,
            train_data=train_data,topk=config['objectives']['validation_topk'],
            preference=pref['weights'] if method=='phn_hvi' else None)
        auxiliary = helpers.evaluate_auxiliary(model,valid_data,next(model.parameters()).device)
        records.append({'solution_index':index,'preference_id':pref['id'] if pref else None,
                        'preference':pref['weights'] if pref else None,'metrics':ranking['metrics'],
                        'checks':ranking['checks'],'auxiliary_validation':auxiliary})
    point = select_operating_point(records,method,config)
    return {'ranking_operating_point':point,'points':records,
            'pareto':pareto_summary([r['objective_vector'] for r in records],config['selection']['validation_coordinate_scales']),
            'selection_is_validation_oracle':False,'operating_point_selection':config['selection']}
