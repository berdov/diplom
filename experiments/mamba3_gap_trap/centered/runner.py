"""Immutable fit body with explicit centered dependencies; fresh process per fit."""
from .. import runner as parent
from . import config as c
from .reuse import bind
from .provenance import identity,require_stage,runtime,imported_sources
from .state import effective_check,initial,paired

_engine=bind(parent,dict(c=c,identity=identity,require_stage=require_stage,runtime=runtime,
    imported_sources=imported_sources,effective_check=effective_check,initial=initial,paired=paired),__package__)
train,main=(_engine[k] for k in ('train','main'))
if __name__=='__main__':main()
