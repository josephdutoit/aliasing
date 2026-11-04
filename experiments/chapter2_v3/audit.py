#!/usr/bin/env python3
"""Run the repository test suite and record the v3 preflight audit on the cluster."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def write_record(run_dir: Path, record: dict) -> None:
    path = run_dir / "AUDIT_PASS.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def classify_tests(xml_path: Path, passed: bool) -> dict[str, str]:
    checks = {
        "new_v3_math": "NOT_FOUND", "definition_specific_counterexamples": "NOT_FOUND",
        "legacy_compatibility": "NOT_FOUND", "gauge_controls": "NOT_FOUND",
        "signed_readout_identity": "NOT_FOUND", "gaussian_wishart": "NOT_FOUND",
        "sparse_moments": "NOT_FOUND", "numerical_identity_maxima": "NOT_RUN",
        "smoke_resume_lock_bundle": "NOT_RUN",
        "legacy_pytest": "PASSED" if passed else "FAILED",
    }
    if not xml_path.is_file():
        return checks
    root = ET.parse(xml_path).getroot()
    testcases = root.findall(".//testcase")
    def has(fragment: str) -> bool:
        return any(fragment in ((case.get("classname", "") + "." + case.get("name", "")).lower())
                   for case in testcases)
    all_new = any("test_intrinsic_gad" in case.get("classname", "") for case in testcases)
    checks["new_v3_math"] = "PASSED" if all_new and passed else "FAILED" if all_new else "NOT_FOUND"
    checks["definition_specific_counterexamples"] = (
        "PASSED" if (has("counterexample") and passed) else "FAILED" if has("counterexample") else "NOT_FOUND")
    checks["legacy_compatibility"] = "PASSED" if (has("test_linear") and passed) else "FAILED" if has("test_linear") else "NOT_FOUND"
    checks["gauge_controls"] = "PASSED" if (has("gauge") and passed) else "FAILED" if has("gauge") else "NOT_FOUND"
    checks["signed_readout_identity"] = "PASSED" if (has("signed") and passed) else "FAILED" if has("signed") else "NOT_FOUND"
    checks["gaussian_wishart"] = "PASSED" if (has("wishart") and passed) else "FAILED" if has("wishart") else "NOT_FOUND"
    checks["sparse_moments"] = "PASSED" if (has("moment") and passed) else "FAILED" if has("moment") else "NOT_FOUND"
    return checks


def numerical_maxima() -> dict[str, float]:
    """Record deterministic residuals for the core population-metric identities."""
    import numpy as np
    from aliasing.intrinsic_gad import gauge_transform, interpolating_population_gad

    rng = np.random.default_rng(81021)
    factor = rng.normal(size=(5, 5))
    Sigma = factor @ factor.T + np.eye(5)
    H = rng.normal(size=(3, 5))
    theta_star = rng.normal(size=5)
    y = rng.normal(size=3)
    original = interpolating_population_gad(H, y, Sigma, theta_star, 0.23)
    G = rng.normal(size=(5, 5)) + 2.0 * np.eye(5)
    features = rng.normal(size=(7, 5))
    transformed = gauge_transform(G, features, H, Sigma, theta_star, A=original["A"])
    gauged = interpolating_population_gad(
        transformed["H"], y, transformed["Sigma"], transformed["theta_star"], 0.23
    )
    projector = np.linalg.solve(Sigma, H.T) @ np.linalg.solve(original["K"], H)
    projected_oracle = (np.eye(5) - projector) @ theta_star
    expected_D0 = 0.5 * float(projected_oracle @ Sigma @ projected_oracle)
    return {
        "interpolation_residual": float(np.linalg.norm(H @ original["theta_circle"] - y)),
        "risk_decomposition_relative_error": float(original["rel_decomposition_error"]),
        "signed_readout_identity_error": float(original["signed_identity_error"]),
        "D0_projection_identity_error": float(abs(original["D0"] - expected_D0)),
        "gauge_kernel_max_abs_error": float(np.max(np.abs(original["K"] - gauged["K"]))),
        "gauge_A_abs_error": float(abs(original["A"] - gauged["A"])),
        "gauge_D0_abs_error": float(abs(original["D0"] - gauged["D0"])),
    }


def run_tests(run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    junit = run_dir / "audit_junit.xml"
    log = run_dir / "audit_pytest.log"
    command = [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"]
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    before = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - before
    output = completed.stdout + completed.stderr
    log.write_text(output, encoding="utf-8")
    checks = classify_tests(junit, completed.returncode == 0)
    try:
        maxima: dict[str, object] = numerical_maxima()
        numerical_pass = all(float(value) <= 1e-10 for key, value in maxima.items()
                             if key != "risk_decomposition_relative_error")
        numerical_pass = numerical_pass and float(maxima["risk_decomposition_relative_error"]) <= 1e-8
    except Exception as exc:
        maxima = {"error": f"{type(exc).__name__}: {exc}"}
        numerical_pass = False
    checks["numerical_identity_maxima"] = "PASSED" if numerical_pass else "FAILED"
    all_required = all(checks[key] == "PASSED" for key in (
        "new_v3_math", "definition_specific_counterexamples", "legacy_compatibility",
        "gauge_controls", "signed_readout_identity", "gaussian_wishart", "sparse_moments",
        "numerical_identity_maxima",
    ))
    status = "UNIT_TESTS_PASSED" if completed.returncode == 0 and all_required else "FAILED"
    if completed.returncode == 0 and not all_required:
        status = "FAILED"
    tests, failures = 0, 0
    if junit.is_file():
        root = ET.parse(junit).getroot()
        tests = int(root.get("tests", "0"))
        failures = int(root.get("failures", "0")) + int(root.get("errors", "0"))
    return {
        "schema": "chapter2_v3_audit_1", "status": status,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command, "returncode": completed.returncode, "elapsed_seconds": elapsed,
        "tests": tests, "failures": failures, "checks": checks,
        "tolerances": {"well_conditioned_absolute": 1e-10, "well_conditioned_relative": 1e-8},
        "numerical_maxima": maxima,
        "log": log.name, "junit": junit.name,
    }


def check_smoke(run_dir: Path, record: dict) -> dict:
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    if plan.get("profile") == "smoke":
        if record.get("status") not in ("UNIT_TESTS_PASSED", "PASSED"):
            raise ValueError("unit-test audit must pass before checking smoke artifacts")
    else:
        smoke_dir = ROOT / "outputs/chapter2_v3/smoke"
        smoke_plan_path = smoke_dir / "plan.json"
        smoke_audit_path = smoke_dir / "AUDIT_PASS.json"
        if not smoke_plan_path.is_file() or not smoke_audit_path.is_file():
            raise FileNotFoundError("completed smoke plan and audit are required before validating this profile")
        smoke_plan = json.loads(smoke_plan_path.read_text(encoding="utf-8"))
        smoke_audit = json.loads(smoke_audit_path.read_text(encoding="utf-8"))
        if smoke_audit.get("status") != "PASSED":
            raise ValueError("the smoke unit-test and bundle audit must say PASSED")
        if smoke_plan.get("code_sha256") != plan.get("code_sha256"):
            raise ValueError("profile code hash differs from the audited smoke implementation")
        record["checks"] = dict(smoke_audit.get("checks", {}))
        record["checks"]["smoke_resume_lock_bundle"] = "NOT_RUN"
        record["source_audit"] = {"run_id": smoke_plan.get("profile"),
                                   "code_sha256": smoke_plan.get("code_sha256"),
                                   "status": smoke_audit.get("status")}
    manifest_path = run_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("smoke MANIFEST.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise ValueError("smoke run must be COMPLETE")
    job_ids = [job["job_id"] for job in plan["jobs"]]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("duplicate immutable world-shard ids in plan")
    expected_files = [
        run_dir / "summary_matrix.csv", run_dir / "MORNING_README.md",
        run_dir / "DECISION_MEMO.md", run_dir / "NEW_CHAT_HANDOFF.md",
        run_dir / "FIGURE_GALLERY.md", run_dir / "failure_rates.csv", run_dir / "results_bundle.zip",
    ]
    missing = [str(path.relative_to(run_dir)) for path in expected_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"smoke artifacts missing: {missing}")
    valid_by_lane: dict[str, int] = {}
    learned_rows_by_lane: dict[str, int] = {}
    valid_learned_by_lane: dict[str, int] = {}
    nonconverged_learned_by_lane: dict[str, int] = {}
    convergence_flags_present: dict[str, bool] = {}
    for lane in ("A", "B", "C", "D1", "D2"):
        raw_path = run_dir / "lanes" / lane / "raw.csv"
        with raw_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        valid_by_lane[lane] = sum(row.get("trial_valid") == "True" for row in rows)
        learned_names = {"C": {"learned"}, "D1": {"learned"},
                         "D2": {"independently_learned", "nested_learned"}}.get(lane, set())
        learned_rows = [row for row in rows if row.get("control_name") in learned_names]
        learned_rows_by_lane[lane] = len(learned_rows)
        convergence_flags_present[lane] = all(
            row.get("representation_converged") in ("True", "False") for row in learned_rows
        )
        valid_learned_by_lane[lane] = sum(
            row.get("trial_valid") == "True" and row.get("control_name") in learned_names
            for row in rows
        )
        nonconverged_learned_by_lane[lane] = sum(
            row.get("representation_converged") == "False" for row in learned_rows
        )
    if any(valid_by_lane[lane] == 0 for lane in ("A", "B", "C", "D1", "D2")):
        raise ValueError(f"every mandatory lane needs a valid trial row: {valid_by_lane}")
    if any(learned_rows_by_lane[lane] == 0 for lane in ("C", "D1", "D2")):
        raise ValueError(f"smoke must produce learned representation rows in C/D1/D2: {learned_rows_by_lane}")
    if any(not convergence_flags_present[lane] for lane in ("C", "D1", "D2")):
        raise ValueError("smoke learned rows must record an explicit convergence flag")
    with zipfile.ZipFile(run_dir / "results_bundle.zip") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"results bundle failed CRC validation at {bad_member}")
        required = {
            "MANIFEST.json", "DECISION_MEMO.md", "NEW_CHAT_HANDOFF.md", "summary_matrix.csv",
            "failure_rates.csv", "MATH_SPEC_REVIEW.md",
            "implementation/src/aliasing/intrinsic_gad.py",
            "implementation/experiments/chapter2_v3/run.py",
            "implementation/scripts/chapter2_v3/launch_overnight.sh",
        }
        absent = sorted(required - set(archive.namelist()))
        if absent:
            raise ValueError(f"bundle is missing required files: {absent}")
    record["status"] = "PASSED"
    record["checks"]["smoke_resume_lock_bundle"] = "PASSED"
    record["smoke"] = {
        "status": manifest["status"], "completed_jobs": manifest.get("completed_jobs"),
        "expected_jobs": manifest.get("expected_jobs"), "unique_world_shards": len(job_ids),
        "bundle_crc": "PASSED", "required_bundle_files": "PASSED",
        "valid_trial_rows_by_lane": valid_by_lane,
        "learned_trial_rows_by_lane": learned_rows_by_lane,
        "valid_learned_rows_by_lane": valid_learned_by_lane,
        "nonconverged_learned_rows_by_lane": nonconverged_learned_by_lane,
    }
    record["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--check-smoke", action="store_true",
                        help="after the smoke collector completes, check its bundle without rerunning tests")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    audit_path = run_dir / "AUDIT_PASS.json"
    try:
        if args.check_smoke:
            if not audit_path.is_file():
                raise FileNotFoundError("AUDIT_PASS.json is missing")
            record = check_smoke(run_dir, json.loads(audit_path.read_text(encoding="utf-8")))
        else:
            record = run_tests(run_dir)
        write_record(run_dir, record)
    except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        record = {"schema": "chapter2_v3_audit_1", "status": "FAILED",
                  "reason": f"{type(exc).__name__}: {exc}", "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        write_record(run_dir, record)
    print(json.dumps(record, sort_keys=True, indent=2))
    return 0 if record.get("status") in ("UNIT_TESTS_PASSED", "PASSED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
