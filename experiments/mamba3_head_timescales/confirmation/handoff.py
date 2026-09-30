"""One mutable service state, deliberately outside the frozen manifest inputs."""
from . import config as c
from experiments.mamba3_mimo_time.records import read,create,update,now


def save(phase,**values):
    c.RUNTIME.mkdir(parents=True,exist_ok=True)
    p=c.RUNTIME/'state.json'
    state=read(p) if p.exists() else {}
    phases=['pilot published','confirmation prepared','submitted','pipeline_finished','terminal','audited','published']
    previous=state.get('phase')
    same_attempt=values.get('execution_attempt',state.get('execution_attempt'))==state.get('execution_attempt')
    if same_attempt and previous in phases and phase in phases and phases.index(previous)>phases.index(phase):
        phase=previous
        values.pop('next_safe_step',None)
    if 'job_ids' in values:values['job_ids']=list(dict.fromkeys(state.get('job_ids',[])+values['job_ids']))
    state.update(phase=phase,updated_at=now(),**values)
    if any(k.lower() in ('token','reservation_token','environment','password','secret') for k in state):
        raise ValueError('Handoff must not contain credentials or full environment')
    (update if p.exists() else create)(p,state)
    lines=['# Head-timescales confirmation','',f'Phase: {phase}','']
    lines.extend(f'- {k}: {v}' for k,v in state.items() if k not in ('phase','task_attachment'))
    lines+=['','Never submit from a missing SSH response. Read reservations and scheduler first. '
            'Maximum12 scientific starts,2 submissions,12h requested GPU time; no2026 refit, TEST0.','']
    (c.RUNTIME/'handoff.md').write_text('\n'.join(lines))
    return state
