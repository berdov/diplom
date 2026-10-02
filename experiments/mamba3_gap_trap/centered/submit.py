"""Same durable one-call submission and two required CPU preflights."""
from .. import submit as parent
from . import config as c
from .reuse import bind
from .provenance import login_verify,verify,bindings
_engine=bind(parent,dict(c=c,login_verify=login_verify,verify=verify,bindings=bindings),__package__)
reserve_and_submit,main=(_engine[k] for k in ('reserve_and_submit','main'))
if __name__=='__main__':main()
