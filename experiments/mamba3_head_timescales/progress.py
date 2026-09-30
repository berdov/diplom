"""Persist intermediate observations without treating progress as a verdict."""
import json
from experiments.mamba3_mimo_time.records import case


def snapshot(value):
    # Strict JSON also rejects tensors/graphs and nonfinite metadata.
    return json.loads(json.dumps(value,allow_nan=False))


def progress_callback(save,latest):
    history=[]
    observed={}
    def persist(checks,phase):
        entry=snapshot(dict(checks=checks,phase=phase))
        if any(k not in entry['checks'] or entry['checks'][k]!=v for k,v in observed.items()):
            raise ValueError('Progress lost or changed an observed numerical check')
        observed.update(entry['checks'])
        history.append(entry)
        latest.clear();latest.update(snapshot(entry))
        payload=case(entry['checks']) if entry['checks'] else dict(checks={},required_keys=[])
        payload.update(hook_phase=entry['phase'],progress_history=history)
        # Each save owns its metadata; later callbacks cannot mutate it.
        save(snapshot(payload))
    return persist
