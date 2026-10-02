"""Same 2 x 3 synthetic updates, no recommendation evaluation."""
from .. import smoke as parent
from . import config as c
from .reuse import bind
from .provenance import identity,require_stage,runtime
from .gate import fresh
from .modulation import attach_projection
main=bind(parent,dict(c=c,identity=identity,require_stage=require_stage,runtime=runtime,
                     fresh=fresh,attach_projection=attach_projection),__package__)['main']
if __name__=='__main__':raise SystemExit(main())
