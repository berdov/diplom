"""Unchanged VALID selection with detached bounded Trap diagnostics."""
import math
import torch
from . import config as c
from experiments.mamba3_mimo_time.trainer import trainer_class as mimo_trainer
from experiments.mamba3_mimo_time.records import read, update


def diagnostics(model):
    alpha = float(model.gap_trap.alpha.detach()) if model.gap_trap_mode == 'gap_trap' else 0.
    grid = c.plan()['diagnostic_grid']
    shifts = [alpha * g / (1 + g) for g in grid]
    logits = c.plan()['diagnostic_content_logits']
    return dict(alpha=alpha,alpha_bounds=[0.,1.],at_lower_bound=alpha==0.,at_upper_bound=alpha==1.,
                shared_across_heads_and_layers=True,gaps_over_R0=grid,logit_shift=shifts,
                odds_multiplier=[math.exp(x) for x in shifts],content_logits=logits,
                current_fraction=[[1/(1+math.exp(-t-s)) for s in shifts] for t in logits],
                interpretation='Conditional on fixed content logit; not a measured distribution of learned event gates')


def trainer_class(base, valid_loader, record, paths, expected_first_batch=None):
    parent = mimo_trainer(base,valid_loader,record,paths,expected_first_batch)
    class GapTrapTrainer(parent):
        def _valid_epoch(self, valid_data, show_progress=False):
            score,metrics = super()._valid_epoch(valid_data,show_progress)
            record['history'][-1]['diagnostics']['gap_trap'] = diagnostics(self.model)
            update(paths['result'],record)
            return score,metrics

        def _save_checkpoint(self,epoch,verbose=True,**kwargs):
            super()._save_checkpoint(epoch,verbose,**kwargs)
            meta=read(paths['metadata'])
            meta.update(gap_trap_mode=record['gap_trap_mode'],gap_trap=record['best_diagnostics']['gap_trap'])
            update(paths['metadata'],meta)
    return GapTrapTrainer
