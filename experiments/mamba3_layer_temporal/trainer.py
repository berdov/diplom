"""Reuse unchanged fit/selection and layer0 VALID diagnostics; record layer1 too."""
from .diagnostics import diagnostics
from experiments.mamba3_mimo_time.trainer import trainer_class as mimo_trainer
from experiments.mamba3_three_time.validation_pilot.diagnostics import Diagnostics
from experiments.mamba3_mimo_time.records import read,update


def trainer_class(base,valid_loader,record,paths,expected_first_batch=None):
    parent=mimo_trainer(base,valid_loader,record,paths,expected_first_batch)
    class LayerTemporalTrainer(parent):
        def _valid_epoch(self,valid_data,show_progress=False):
            collector=Diagnostics('dual')
            hook=self.model.layer1_times.register_forward_hook(collector.hook) if self.model.temporal_sharing=='layer_specific' else None
            try:score,metrics=super()._valid_epoch(valid_data,show_progress)
            finally:
                if hook is not None:hook.remove()
            row=record['history'][-1]['diagnostics']
            if hook is not None:
                row['shared_across_layers']=False
                row['calibrator_count']=4
                row['layer1_valid_scales']=dict(collector.result(),shared_across_layers=False)
            row['layer_temporal']=diagnostics(self.model)
            update(paths['result'],record)
            return score,metrics

        def _save_checkpoint(self,epoch,verbose=True,**kwargs):
            super()._save_checkpoint(epoch,verbose,**kwargs)
            meta=read(paths['metadata']);meta.update(temporal_sharing=record['temporal_sharing'],layer_temporal=record['best_diagnostics']['layer_temporal'])
            update(paths['metadata'],meta)
    return LayerTemporalTrainer
