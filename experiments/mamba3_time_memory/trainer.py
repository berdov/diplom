"""Reuse VALID-only fit/selection; observe memory in the existing VALID forwards."""
from .diagnostics import MemoryDiagnostics
from .progress import write as write_progress
from experiments.mamba3_mimo_time.trainer import trainer_class as mimo_trainer
from experiments.mamba3_mimo_time.records import read,update


def trainer_class(base,valid_loader,record,paths,expected_first_batch=None):
    parent=mimo_trainer(base,valid_loader,record,paths,expected_first_batch)
    class MemoryTrainer(parent):
        def _valid_epoch(self,valid_data,show_progress=False):
            collector=MemoryDiagnostics(record['memory_mode']) if record['memory_mode']!='no_memory' else None
            if getattr(self.model,'memory_observer',None) is not None:raise ValueError('Observer lifecycle collision')
            if collector is not None:self.model.memory_observer=collector.observe
            try:score,metrics=super()._valid_epoch(valid_data,show_progress)
            finally:self.model.memory_observer=None
            if collector is not None:record['history'][-1]['diagnostics']['memory']=collector.result()
            update(paths['result'],record);write_progress(record,paths)
            return score,metrics

        def _save_checkpoint(self,epoch,verbose=True,**kwargs):
            super()._save_checkpoint(epoch,verbose,**kwargs)
            meta=read(paths['metadata']);meta.update(memory_mode=record['memory_mode'],memory=record['best_diagnostics'].get('memory'))
            update(paths['metadata'],meta)
    return MemoryTrainer
