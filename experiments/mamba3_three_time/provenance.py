"""Content hashes on compute nodes; git is used only by explicit local freeze."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from experiments.mamba3_time_mechanisms.provenance import fingerprint, PIN

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE = "460bd3e687d26a458cafc21960391f83905da962ca5c992d6062c7b048f0a73f"
BASE = "ccc4849e6e1ea460a316ab9648f36c54bd832fd4"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest():
    files = [*HERE.glob("*.py"),*(HERE/"tests").glob("*.py"),
             *(HERE/name for name in ("future_plan.json","test_plan.json","test_plan_002.json","test_plan_003.json","upstream_manifest.json",
                                     "frozen_snapshot.json","README.md","LICENSE.upstream")),
             HERE/"evidence/upstream_audit.md",HERE/"evidence/stable_scan_algebra.md",
             HERE/"evidence/attempt003_contract.md",
             ROOT/"slurm/mamba3_three_time_correctness.sh",ROOT/"slurm/mamba3_three_time_correctness_002.sh",
             ROOT/"slurm/mamba3_three_time_correctness_003.sh"]
    hashes = {str(path.relative_to(ROOT)):sha(path) for path in sorted(files)}
    digest = hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return dict(schema_version=1,source_hash=digest,core_hash=CORE,files=hashes)


def verify(attempt_id="003"):
    if fingerprint() != CORE:
        raise ValueError("Frozen temporal core changed")
    frozen = json.loads((HERE/"frozen_snapshot.json").read_text())
    bad = [path for path,expected in frozen["files"].items() if sha(ROOT/path) != expected]
    if bad:
        raise ValueError("Frozen sources/results changed: "+repr(bad))
    actual = source_manifest()
    manifest_path = HERE / ("source_manifest_003.json" if attempt_id == "003" else "source_manifest.json")
    if actual != json.loads(manifest_path.read_text()):
        raise ValueError("Three-time source manifest mismatch")
    return actual


def upstream():
    manifest = json.loads((HERE/"upstream_manifest.json").read_text())
    distribution = importlib.metadata.distribution("mamba-ssm")
    direct = json.loads(distribution.read_text("direct_url.json"))
    if direct.get("vcs_info",{}).get("commit_id") != PIN:
        raise ValueError("Installed Mamba commit mismatch")
    spec = importlib.util.find_spec("mamba_ssm")
    package_parent = Path(spec.origin).resolve().parent.parent
    actual = {}
    for name,expected in manifest["files"].items():
        path = HERE/"LICENSE.upstream" if name == "LICENSE" else package_parent/name
        actual[name] = sha(path)
        if actual[name] != expected:
            raise ValueError(f"Upstream file changed: {path}, {actual[name]} != {expected}")
    return dict(pinned_commit=PIN,manifest_sha256=sha(HERE/"upstream_manifest.json"),
                installed_path=str(package_parent),files=actual)


def require_submission(attempt_id="003"):
    manifest = verify(attempt_id)
    commit = os.environ.get("RUN_COMMIT","")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Login-verified exact execution commit required")
    if os.environ.get("EXPECTED_STUDY_HASH") != manifest["source_hash"] or os.environ.get("EXPECTED_CORE_HASH") != CORE:
        raise ValueError("Submission hashes mismatch")
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Explicit Slurm allocation required")
    return manifest


def freeze():
    # This is a metadata-generation command, never called by Slurm.
    names = subprocess.check_output(["git","ls-tree","-r","--name-only",BASE],cwd=ROOT,text=True).splitlines()
    names = [n for n in names if n.startswith("experiments/mamba3_") or
             n in ("experiments/results.csv","reports/RESULTS.md","outputs/data/protocol_b_manifest.json")]
    hashes = {}
    for name in names:
        expected = subprocess.check_output(["git","show",f"{BASE}:{name}"],cwd=ROOT)
        if sha(ROOT/name) != hashlib.sha256(expected).hexdigest():
            raise ValueError("Base snapshot differs: "+name)
        hashes[name] = sha(ROOT/name)
    if json.loads((HERE/"frozen_snapshot.json").read_text()) != dict(base_commit=BASE,files=hashes):
        raise ValueError("Do not rewrite the frozen snapshot")
    (HERE/"source_manifest_003.json").write_text(json.dumps(source_manifest(),indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze",action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        print(json.dumps(verify(),indent=2))
