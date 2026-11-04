#!/usr/bin/env python3
"""Build an immutable, deterministic Chapter 2 v3 Slurm job plan."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "chapter2_v3"
RUN_SCHEMA = "chapter2_v3_plan_1"
LANE_ORDER = {name: index for index, name in enumerate(("A", "B", "C", "D1", "D2", "P"))}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def git_metadata() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    # Generated run folders live under outputs/ and must not invalidate a run
    # plan on resume. Relevant untracked Python source/config content is hashed
    # below and in the semantic config block.
    status = git("status", "--porcelain=v1", "--untracked-files=no")
    return {
        "git_sha": git("rev-parse", "HEAD") or "unknown",
        "git_branch": git("branch", "--show-current") or "unknown",
        "working_tree_clean": status == "",
        "working_tree_status": status,
    }


def code_fingerprint() -> tuple[str, dict[str, str]]:
    paths = [ROOT / "src/aliasing/intrinsic_gad.py", ROOT / "src/aliasing/chapter2_v3.py"]
    paths.extend(sorted((ROOT / "experiments/chapter2_v3").glob("*.py")))
    paths.extend(sorted((ROOT / "scripts/chapter2_v3").glob("*")))
    paths.extend(sorted((ROOT / "tests").glob("test_*.py")))
    hashes: dict[str, str] = {}
    for path in paths:
        if path.is_file() and path.name != "plan.py" or path == Path(__file__):
            relative = path.relative_to(ROOT).as_posix()
            hashes[relative] = sha256_bytes(path.read_bytes())
    return sha256_bytes(canonical_json(hashes).encode("utf-8")), hashes


def _job_id(lane: str, world: dict[str, Any], seed: int | None, block: int) -> str:
    payload = {"lane": lane, "world": world, "representation_seed": seed, "probe_seed_block": block}
    return lane.lower() + "_" + sha256_bytes(canonical_json(payload).encode("utf-8"))[:16]


def _add(jobs: list[dict[str, Any]], lane: str, world: dict[str, Any], seed: int | None,
         block: int, config: dict[str, Any], source_job_id: str | None = None) -> str:
    identifier = _job_id(lane, world, seed, block)
    job = {
        "job_id": identifier,
        "lane": lane,
        "world": world,
        "representation_seed": seed,
        "probe_seed_block": block,
        "config": config,
    }
    if source_job_id:
        job["source_job_id"] = source_job_id
    jobs.append(job)
    return identifier


def validate_profile(config: dict[str, Any]) -> None:
    if config.get("profile") not in ("smoke", "overnight_discovery", "confirmation"):
        raise ValueError("profile must be smoke, overnight_discovery, or confirmation")
    if int(config.get("seed", -1)) < 0:
        raise ValueError("seed must be nonnegative")
    for lane_name in ("lane_A", "lane_B", "lane_C", "lane_D1", "lane_D2", "lane_P"):
        if lane_name not in config or not isinstance(config[lane_name], dict):
            raise ValueError(f"missing {lane_name} configuration")
    n = int(config["lane_A"]["input_dimension"])
    for width in config["lane_A"]["widths"] + config["lane_B"]["widths"]:
        if not 1 <= int(width) <= n:
            raise ValueError("Lane A/B widths must fit their input dimension")
    n_c = int(config["lane_C"]["input_dimension"])
    if any(not 1 <= int(width) < n_c for width in config["lane_C"]["widths"]):
        raise ValueError("Lane C widths must be between one and input_dimension-1")
    n_d2 = int(config["lane_D2"]["input_dimension"])
    if any(not 1 <= int(width) <= n_d2 for width in config["lane_D2"]["widths"]):
        raise ValueError("Lane D2 widths must fit their input dimension")
    if "in_span" in config["lane_D2"].get("target_kinds", []):
        raise ValueError(
            "Lane D2 requires a raw teacher fixed across widths; use dense/sparse targets, "
            "since a representation-relative in_span teacher changes with independently trained width"
        )
    seeds = config.get("representation_seeds", [])
    synthetic_seeds = config.get("synthetic_representation_seeds", seeds)
    blocks = config.get("probe_seed_blocks", [])
    if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0:
        raise ValueError("representation_seeds must be unique nonnegative integers")
    if not synthetic_seeds or len(synthetic_seeds) != len(set(synthetic_seeds)) or min(synthetic_seeds) < 0:
        raise ValueError("synthetic_representation_seeds must be unique nonnegative integers")
    if not blocks or len(blocks) != len(set(blocks)) or min(blocks) < 0:
        raise ValueError("probe_seed_blocks must be unique nonnegative integers")
    if int(config.get("probe_draws_per_block", 0)) < 1:
        raise ValueError("probe_draws_per_block must be positive")
    if any(not 1 <= int(width) <= n for width in config["lane_D1"].get(
            "gaussian_baseline_widths", config["lane_D1"]["source_widths"])):
        raise ValueError("Lane D1 Gaussian baseline widths must fit Lane A's input dimension")


def build_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_profile(config)
    jobs: list[dict[str, Any]] = []
    seeds = [int(value) for value in config["representation_seeds"]]
    synthetic_seeds = [int(value) for value in config.get("synthetic_representation_seeds", seeds)]
    blocks = [int(value) for value in config["probe_seed_blocks"]]
    master_seed = int(config["seed"])

    a = config["lane_A"]
    for width in a["widths"]:
        for family in a["families"]:
            for seed in synthetic_seeds:
                for block in blocks:
                    world = {"input_dimension": a["input_dimension"], "M": int(width), "family": family}
                    lane_config = {**a, "master_seed": master_seed, "profile": config["profile"],
                                   "n_test": config["n_test"], "n_oracle": config["n_oracle"],
                                   "probe_draws": config["probe_draws_per_block"],
                                   "bootstrap_replicates": config["bootstrap_replicates"]}
                    _add(jobs, "A", world, seed, block, lane_config)

    b = config["lane_B"]
    for width in b["widths"]:
        for family in b["families"]:
            for seed in synthetic_seeds:
                for block in blocks:
                    world = {"input_dimension": b["input_dimension"], "M": int(width), "family": family}
                    lane_config = {**b, "master_seed": master_seed, "profile": config["profile"],
                                   "n_test": config["n_test"], "n_oracle": config["n_oracle"],
                                   "probe_draws": config["probe_draws_per_block"],
                                   "bootstrap_replicates": config["bootstrap_replicates"]}
                    _add(jobs, "B", world, seed, block, lane_config)

    c = config["lane_C"]
    c_jobs: dict[tuple[int, float, str, int], str] = {}
    for width in c["widths"]:
        for probability in c["probabilities"]:
            for importance in c["importance_profiles"]:
                for seed in seeds:
                    world = {"input_dimension": c["input_dimension"], "M": int(width),
                             "p": float(probability), "importance_profile": importance}
                    lane_config = {**c, "master_seed": master_seed, "profile": config["profile"],
                                   "n_test": config["n_test"], "n_oracle": config["n_oracle"],
                                   "probe_draws": config["probe_draws_per_block"],
                                   "probe_seed_blocks": blocks,
                                   "bootstrap_replicates": config["bootstrap_replicates"]}
                    identifier = _add(jobs, "C", world, seed, 0, lane_config)
                    c_jobs[(int(width), float(probability), importance, seed)] = identifier

    d1 = config["lane_D1"]
    for width in d1["source_widths"]:
        for probability in d1["source_probabilities"]:
            for importance in d1["source_importance_profiles"]:
                for seed in seeds:
                    source = c_jobs.get((int(width), float(probability), importance, seed))
                    if source is None:
                        continue
                    for block in blocks:
                        world = {"source_job_id": source, "M": int(width), "p": float(probability),
                                 "importance_profile": importance}
                        lane_config = {**d1, "input_dimension": c["input_dimension"],
                                       "master_seed": master_seed, "profile": config["profile"],
                                       "n_test": config["n_test"], "n_oracle": config["n_oracle"],
                                       "probe_draws": config["probe_draws_per_block"],
                                       "bootstrap_replicates": config["bootstrap_replicates"]}
                        _add(jobs, "D1", world, seed, block, lane_config, source)

    gaussian_widths = d1.get("gaussian_baseline_widths", d1["source_widths"])
    for width in gaussian_widths:
        for seed in synthetic_seeds:
            for block in blocks:
                world = {"design": "gaussian_baseline",
                         "input_dimension": config["lane_A"]["input_dimension"],
                         "M": int(width), "family": "gaussian"}
                lane_config = {**d1, "input_dimension": config["lane_A"]["input_dimension"],
                               "master_seed": master_seed, "profile": config["profile"],
                               "n_test": config["n_test"], "n_oracle": config["n_oracle"],
                               "probe_draws": config["probe_draws_per_block"],
                               "bootstrap_replicates": config["bootstrap_replicates"]}
                _add(jobs, "D1", world, seed, block, lane_config)

    d2 = config["lane_D2"]
    d2_lane_config = {**d2, "master_seed": master_seed, "profile": config["profile"],
                      "n_test": config["n_test"], "n_oracle": config["n_oracle"],
                      "probe_draws": config["probe_draws_per_block"],
                      "probe_seed_blocks": blocks,
                      "bootstrap_replicates": config["bootstrap_replicates"]}
    for width in d2["widths"]:
        for seed in seeds:
            world = {"design": "independent_learned_width", "input_dimension": d2["input_dimension"],
                     "M": int(width)}
            _add(jobs, "D2", world, seed, 0, d2_lane_config)
    for family in d2["nested_families"]:
        for seed in seeds:
            world = {"design": "nested_fixed_bank", "input_dimension": d2["input_dimension"],
                     "family": family}
            _add(jobs, "D2", world, seed, 0, d2_lane_config | {"nested_family": family})
    if d2.get("width_matched_random", True):
        for width in d2["widths"]:
            for seed in seeds:
                world = {"design": "width_matched_random", "input_dimension": d2["input_dimension"],
                         "M": int(width)}
                _add(jobs, "D2", world, seed, 0, d2_lane_config)

    p = config["lane_P"]
    world = {"activation_cache": p.get("activation_cache"), "optional": True}
    _add(jobs, "P", world, None, 0, {**p, "master_seed": master_seed,
                                      "profile": config["profile"], "n_test": config["n_test"],
                                      "bootstrap_replicates": config["bootstrap_replicates"]})

    jobs.sort(key=lambda item: (LANE_ORDER[item["lane"]], item["job_id"]))
    for index, job in enumerate(jobs):
        job["job_index"] = index
    return jobs


def _copy_provenance_file(source: Path, destination: Path) -> None:
    if source.is_file() and not destination.exists():
        destination.write_bytes(source.read_bytes())


def create_plan(profile: str, output_dir: Path, config_path: Path | None = None) -> dict[str, Any]:
    source = config_path or (CONFIG_DIR / f"{profile}.json")
    config = json.loads(source.read_text(encoding="utf-8"))
    if config.get("profile") != profile:
        raise ValueError("selected profile does not match the profile recorded in the config")
    jobs = build_jobs(config)
    code_hash, file_hashes = code_fingerprint()
    config_hash = sha256_bytes(canonical_json(config).encode("utf-8"))
    git = git_metadata()
    semantic = {
        "schema": RUN_SCHEMA,
        "profile": profile,
        "config": config,
        "config_sha256": config_hash,
        "jobs": jobs,
        "code_sha256": code_hash,
        "source_sha256": file_hashes,
        "git": git,
    }
    plan_hash = sha256_bytes(canonical_json(semantic).encode("utf-8"))
    plan = semantic | {
        "plan_hash": plan_hash,
        "planned_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_root": str(ROOT),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("plan_hash") != plan_hash:
            raise ValueError("run directory already contains a different immutable plan")
        plan = existing
    else:
        descriptor, name = tempfile.mkstemp(prefix="plan.", suffix=".tmp", dir=output_dir)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(plan, handle, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, plan_path)
        except FileExistsError:
            existing = json.loads(plan_path.read_text(encoding="utf-8"))
            if existing.get("plan_hash") != plan_hash:
                raise ValueError("concurrent planner installed a different plan")
            plan = existing
        finally:
            temporary.unlink(missing_ok=True)

    manifest_path = output_dir / "MANIFEST.json"
    if not manifest_path.exists():
        atomic_json(manifest_path, {
            "schema": RUN_SCHEMA,
            "run_id": output_dir.name,
            "plan_hash": plan_hash,
            "git": git,
            "code_sha256": code_hash,
            "config_sha256": config_hash,
            "status": "PLANNED",
            "expected_jobs": len(jobs),
            "job_status_counts": {"PLANNED": len(jobs)},
            "jobs": [{"job_id": job["job_id"], "job_index": job["job_index"],
                     "lane": job["lane"], "world": job["world"],
                     "representation_seed": job.get("representation_seed"),
                     "probe_seed_block": job.get("probe_seed_block"), "status": "PLANNED"}
                    for job in jobs],
            "resources": {"partition": "NOT_QUERIED", "gpu": "NOT_ASSUMED"},
        })
    audit_path = output_dir / "AUDIT_PASS.json"
    if not audit_path.exists():
        atomic_json(audit_path, {
            "status": "NOT_RUN",
            "reason": "Implementation intentionally does not execute tests or experiments.",
            "checks": {name: "NOT_RUN" for name in (
                "new_v3_math", "definition_specific_counterexamples", "legacy_compatibility",
                "gauge_controls", "signed_readout_identity", "gaussian_wishart",
                "sparse_moments", "numerical_identity_maxima", "smoke_resume_lock_bundle", "legacy_pytest",
            )},
            "tolerances": {"well_conditioned_absolute": 1e-10, "well_conditioned_relative": 1e-8},
        })
    audit_source = ROOT / "outputs/chapter2_v3/implementation_e2eb7a2/REPO_AUDIT.md"
    _copy_provenance_file(audit_source, output_dir / "REPO_AUDIT.md")
    _copy_provenance_file(ROOT / "REFACTOR_CHANGELOG.md", output_dir / "REFACTOR_CHANGELOG.md")
    _copy_provenance_file(ROOT / "MATH_SPEC_REVIEW.md", output_dir / "MATH_SPEC_REVIEW.md")
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True,
                        choices=("smoke", "overnight_discovery", "confirmation"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="optional immutable JSON profile file")
    args = parser.parse_args(argv)
    try:
        plan = create_plan(args.profile, args.output_dir.resolve(), args.config.resolve() if args.config else None)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    counts: dict[str, int] = {}
    for job in plan["jobs"]:
        counts[job["lane"]] = counts.get(job["lane"], 0) + 1
    print(json.dumps({"run_dir": str(args.output_dir.resolve()), "plan_hash": plan["plan_hash"],
                      "git_sha": plan["git"]["git_sha"], "code_sha256": plan["code_sha256"],
                      "jobs": counts, "total_jobs": len(plan["jobs"])}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
