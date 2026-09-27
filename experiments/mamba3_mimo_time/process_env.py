"""Per-child CPU isolation; never changes the allocation's environment."""

CUDA_MASK = 'CUDA_VISIBLE_DEVICES'


def visibility(environment):
    return dict(present=CUDA_MASK in environment, value=environment.get(CUDA_MASK))


def child_environment(stage, parent):
    if stage not in ('preflight', 'admission', 'smoke', 'base', 'dual', 'triple'):
        raise ValueError('Unexpected pipeline stage: ' + stage)
    child = dict(parent)
    if stage == 'preflight':
        child[CUDA_MASK] = ''
    return child
