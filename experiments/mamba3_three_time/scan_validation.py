"""Verify the two vendored functions differ only in the frozen scan candidates."""

import ast
from pathlib import Path

ANGLE_OLD = """chunk_sum = tl.sum(grad_out_vals, axis=0)
fwd_cumsum = tl.cumsum(grad_out_vals, axis=0)
rev_cumsum = chunk_sum[None, :] - fwd_cumsum + grad_out_vals"""
ANGLE_NEW = """chunk_sum = tl.sum(grad_out_vals, axis=0)
rev_cumsum = tl.cumsum(grad_out_vals, axis=0, reverse=True)"""
ADT_OLD = """dM_rev_vector += (tl.sum(dM_rev_vector) + dM_scalar) + tl.cumsum(dM_vector - dM_rev_vector) - dM_vector"""
ADT_NEW = """indices = tl.arange(0, CHUNK_SIZE)
prefix_v = tl.cumsum(dM_vector, axis=0)
exclusive_v = tl.where(indices > 0, tl.gather(prefix_v, tl.maximum(indices - 1, 0), axis=0), 0.0)
dM_rev_vector = tl.cumsum(dM_rev_vector, axis=0, reverse=True) + exclusive_v + dM_scalar"""


def validate_one(original, candidate, old, new, names):
    class Replace(ast.NodeTransformer):
        count = 0
        def generic_visit(self,node):
            for field,value in ast.iter_fields(node):
                if isinstance(value,list) and value and isinstance(value[0],ast.stmt):
                    pattern = ast.parse(old).body
                    for i in range(len(value)-len(pattern)+1):
                        if [ast.dump(v) for v in value[i:i+len(pattern)]] == [ast.dump(v) for v in pattern]:
                            value[i:i+len(pattern)] = ast.parse(new).body
                            self.count += 1
                            break
            return super().generic_visit(node)
    official = ast.parse(original)
    transform = Replace()
    transform.visit(official)
    if transform.count != 1:
        raise ValueError("Expected exactly one arithmetic replacement")
    def extract(tree):
        return [ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(extract(official)) != 2 or extract(official) != extract(ast.parse(candidate)):
        raise ValueError("Copied kernel/wrapper contains unrelated changes")
    return True


def validate_installed(package_parent, here):
    result = {}
    for filename, local, old, new, names in (
        ("angle_dt.py","stable_angle.py",ANGLE_OLD,ANGLE_NEW,("angle_dt_bwd_kernel","angle_dt_bwd")),
        ("mamba3_siso_bwd.py","stable_adt.py",ADT_OLD,ADT_NEW,("mamba3_siso_bwd_kernel_dqkv","compute_dqkv"))):
        original = Path(package_parent)/"mamba_ssm/ops/triton/mamba3"/filename
        result[local] = validate_one(original.read_text(),(here/local).read_text(),old,new,names)
    return result
