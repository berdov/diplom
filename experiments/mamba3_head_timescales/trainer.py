"""Add detached reference diagnostics before the unchanged checkpoint callback."""
import torch
from . import config as c
from experiments.mamba3_mimo_time.trainer import trainer_class as mimo_trainer
from experiments.mamba3_mimo_time.records import read, update


def diagnostics(model):
    result={}
    with torch.no_grad():
        for name,cal in model.times.calibrators.items():
            q=cal.log_reference_ratio() if hasattr(cal,'alpha') else torch.zeros(1,dtype=torch.float64,device=cal.reference.device)
            ratios=q.exp()
            grid=torch.tensor(c.plan()['diagnostic_grid'],device=q.device,dtype=torch.float64).reshape(1,-1)
            curve=cal(grid*cal.reference,torch.ones_like(grid,dtype=torch.bool))
            result[name]=dict(alpha=cal.alpha.detach().cpu().tolist() if hasattr(cal,'alpha') else None,
                             reference_ratio=ratios.cpu().tolist(),reference_ms=(cal.reference*ratios).cpu().tolist(),
                             displayed_ratio_per_head=ratios.expand(2).cpu().tolist(),
                             displayed_ms_per_head=(cal.reference*ratios.expand(2)).cpu().tolist(),
                             near_reference_bound=((ratios<.2525)|(ratios>3.96)).cpu().tolist(),
                             independent_parameters=0 if not hasattr(cal,'alpha') else cal.alpha.numel(),
                             gaps_over_R0=c.plan()['diagnostic_grid'],scale_by_gap_and_head=curve[0].cpu().tolist())
    return dict(time_scale_mode=model.time_scale_mode,global_model_parameters=True,mechanisms=result)


def trainer_class(base, valid_loader, record, paths, expected_first_batch=None):
    parent=mimo_trainer(base,valid_loader,record,paths,expected_first_batch)
    class HeadTimeTrainer(parent):
        def _valid_epoch(self, valid_data, show_progress=False):
            score,metrics=super()._valid_epoch(valid_data,show_progress)
            record['history'][-1]['diagnostics']['head_timescales']=diagnostics(self.model)
            update(paths['result'],record)
            return score,metrics

        def _save_checkpoint(self,epoch,verbose=True,**kwargs):
            super()._save_checkpoint(epoch,verbose,**kwargs)
            meta=read(paths['metadata'])
            meta.update(time_scale_mode=record['time_scale_mode'],head_timescales=record['best_diagnostics']['head_timescales'])
            update(paths['metadata'],meta)
    return HeadTimeTrainer
