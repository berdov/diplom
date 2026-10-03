"""Set child visibility before imports; preserve scheduler GPU allocation."""
from . import config as c

def child_environment(stage,parent):
    if stage not in ('coverage','preflight','gate','smoke',*c.MODES):raise ValueError('Unplanned stage')
    result=dict(parent,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    if stage in ('coverage','preflight'):result['CUDA_VISIBLE_DEVICES']=''
    return result
