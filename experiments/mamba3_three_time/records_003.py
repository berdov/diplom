"""Explicit registries, nested required checks, durable case state transitions."""

import traceback
from .evidence import update_record

FINAL = {"PASS", "FAIL", "INCONCLUSIVE", "NOT_RUN", "INTERRUPTED", "ADVISORY"}


def required_pass(value):
    if isinstance(value, dict):
        if value.get("required") is False:
            return True
        if "passed" in value and value["passed"] is not True:
            return False
        if value.get("status") in ("FAIL", "INCONCLUSIVE", "NOT_RUN", "INTERRUPTED", "RUNNING"):
            return False
        return all(required_pass(v) for v in value.values())
    if isinstance(value, list):
        return all(required_pass(v) for v in value)
    return True


def identity(arch, suite, *, backend="candidate", length="all", prefix="none", multiplier=1):
    return f"{suite}/{arch}/{backend}/L{length}/P{prefix}/x{multiplier}"


class Registry:
    def __init__(self, result, save, expected):
        if len(expected) != len(set(expected)):
            raise ValueError("Duplicate expected case IDs")
        self.result, self.save = result, save
        result.update(required_registry=expected, cases=[], progress={})
        self.save()

    def start(self, case_id):
        if case_id not in self.result["required_registry"]:
            raise ValueError("Unexpected case ID: " + case_id)
        if case_id in self.result["progress"]:
            raise ValueError("Duplicate case ID: " + case_id)
        self.result["progress"][case_id] = {"status": "RUNNING"}
        self.save()

    def finish(self, case_id, row):
        if self.result["progress"].get(case_id, {}).get("status") != "RUNNING":
            raise ValueError("Case not owned/running: " + case_id)
        row = dict(row, case_id=case_id)
        row["passed"] = bool(row.get("passed")) and required_pass(row)
        row["status"] = "PASS" if row["passed"] else (row.get("status") if row.get("status") in
            ("INCONCLUSIVE", "NOT_RUN", "INTERRUPTED") else "FAIL")
        if row["status"] not in FINAL:
            row["status"] = "FAIL"
        self.result["cases"].append(row)
        self.result["progress"][case_id] = {"status": row["status"]}
        self.save()
        return row

    def run(self, case_id, factory, *, gate=False):
        self.start(case_id)
        try:
            row = factory()
        except Exception:
            row = dict(name=case_id, passed=False, status="FAIL", traceback=traceback.format_exc())
            traceback.print_exc()
        row = self.finish(case_id, row)
        if gate and not row["passed"]:
            raise RuntimeError("Required gate failed; evidence persisted: " + case_id)
        return row

    def close(self):
        for state in self.result["progress"].values():
            if state["status"] == "RUNNING":
                state["status"] = "INTERRUPTED"
        actual = [r["case_id"] for r in self.result["cases"]]
        expected = self.result["required_registry"]
        self.result["coverage"] = dict(missing=sorted(set(expected)-set(actual)),
            unexpected=sorted(set(actual)-set(expected)), duplicates=len(actual) != len(set(actual)))
        good = sorted(actual) == sorted(expected) and all(r["passed"] and required_pass(r) for r in self.result["cases"])
        self.save()
        return good


def interrupt_file(path):
    """Parent subprocess owner finalizes evidence after signals/abrupt child exit."""
    import json
    if not path.exists():
        return
    value = json.loads(path.read_text())
    changed = False
    for state in value.get("progress", {}).values():
        if state["status"] == "RUNNING":
            state["status"] = "INTERRUPTED"
            changed = True
    if value.get("status") == "RUNNING":
        value["status"] = "INTERRUPTED"
        changed = True
    if changed:
        update_record(path, value)
