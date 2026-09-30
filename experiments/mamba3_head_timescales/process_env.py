"""Mask CPU subprocess before import; GPU children retain allocation visibility."""
def child_environment(stage, parent):
    if stage not in ('preflight','gate','smoke','fixed','shared_tau','head_tau'):
        raise ValueError('Unplanned stage')
    child=dict(parent,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    if stage=='preflight':
        child['CUDA_VISIBLE_DEVICES']=''
    return child
