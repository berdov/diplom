"""Frozen attempt002 identity and paths, disjoint from attempt001."""

from .provenance import HERE

ATTEMPT = "002"
PARENT = "1bff0863e13c90161b6828c1905b2f5a0ab62a14"
PLAN = HERE / "test_plan_002.json"
LOGS = HERE / "slurm_logs" / "attempt_002"
RUNS = HERE / "runs"
SUMMARY = RUNS / "technical_summary_002.json"
SUBMISSION = LOGS / "submission_002.json"
LOCK = LOGS / "execution_002.lock"
PREFLIGHT = LOGS / "login_preflight.json"
PIPELINE = LOGS / "pipeline_status.json"


def evidence(arch):
    if arch not in ("SISO", "MIMO"):
        raise ValueError(arch)
    return RUNS / f"{arch.lower()}_correctness_002.json"


def require_unused():
    paths = [LOCK, PIPELINE, SUMMARY, evidence("SISO"), evidence("MIMO")]
    paths += [LOGS / f"{a}_{s}.log" for a in ("siso", "mimo") for s in ("stdout", "stderr")]
    occupied = [str(path) for path in paths if path.exists() or path.is_symlink()]
    if occupied:
        raise FileExistsError("Attempt002 already reserved/executed: " + repr(occupied))
