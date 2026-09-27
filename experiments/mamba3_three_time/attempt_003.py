"""Immutable attempt003 paths. Old attempts are never reset or reused."""

from .provenance import HERE

ATTEMPT = "003"
PARENT = "d0c8a2356b235981436715e2d340803e1a0951a0"
PLAN = HERE / "test_plan_003.json"
SOURCE_MANIFEST = HERE / "source_manifest_003.json"
LOGS = HERE / "slurm_logs" / "attempt_003"
RUNS = HERE / "runs"
SUMMARY = RUNS / "technical_summary_003.json"
SUBMISSION = LOGS / "submission_003.json"
LOCK = LOGS / "execution_003.lock"
PREFLIGHT = LOGS / "login_preflight.json"
PIPELINE = LOGS / "pipeline_status.json"


def evidence(arch):
    if arch not in ("SISO", "MIMO"):
        raise ValueError(arch)
    return RUNS / f"{arch.lower()}_correctness_003.json"


def require_unused():
    paths = [LOCK, PIPELINE, SUMMARY, evidence("SISO"), evidence("MIMO")]
    paths += [LOGS / f"{a}_{s}.log" for a in ("siso", "mimo") for s in ("stdout", "stderr")]
    occupied = [str(path) for path in paths if path.exists() or path.is_symlink()]
    if occupied:
        raise FileExistsError("Attempt003 already reserved/executed: " + repr(occupied))
