"""Persistent exclusive guard and durable publication for one-shot evaluation."""

import json
import os
from pathlib import Path


class OneShotResult:
    def __init__(self, result, lock):
        self.result = Path(result)
        self.lock = Path(lock)
        if self.result.exists():
            raise FileExistsError(f'Final result already exists: {self.result}')
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        with self.lock.open('x') as handle:
            json.dump({'pid': os.getpid(), 'job_id': os.environ.get('SLURM_JOB_ID')}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Never remove the lock automatically, including after technical failure.
        if self.result.exists():
            raise FileExistsError('Final result appeared while acquiring guard')

    def publish(self, payload):
        if self.result.exists():
            raise FileExistsError('Refusing to overwrite final result')
        temporary = self.result.with_suffix('.json.tmp')
        with temporary.open('x') as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        # All publishers must hold the exclusive persistent lock above.
        if self.result.exists():
            raise FileExistsError('Refusing to overwrite final result')
        os.rename(temporary, self.result)
        directory = os.open(self.result.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
