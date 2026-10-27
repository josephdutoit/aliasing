"""Run the merged repository's experiments with a bounded one-node budget.

Run from the repository root via overnight.sbatch. Each experiment gets a log,
command, status and its own output directory. Re-running with --resume uses the
same suite directory, skips completed steps and keeps failed attempts intact.
"""

from __future__ import annotations

import argparse
from collections import deque
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile


REPO = Path(__file__).resolve().parent.parent
KNOWN = {
    "02": ("02_random_feature_aliasing.py", ["--sweep", "samplewise", "--num-seeds", "6", "--num-workers", "12", "--n-test", "1024"]),
    "02-featurewise": ("02_random_feature_aliasing.py", ["--sweep", "featurewise", "--num-seeds", "6", "--num-workers", "12", "--n-test", "1024"]),
    "04": ("04_superposition_aliasing.py", []),
    "05": ("05_static_dictionary_geometry.py", ["--num-seeds", "6", "--sample-sizes", "8,16,32"]),
    "06": ("06_trained_sparse_feature_geometry.py", ["--feature-counts", "8,10", "--num-seeds", "4", "--epochs", "250", "--n-test", "512"]),
    "07": ("07_static_geometry_factorial.py", ["--feature-counts", "6,8", "--sample-sizes", "8,32,96", "--joint-probabilities", "0,0.1,0.2", "--tasks", "sum,pair_contrast", "--num-seeds", "6"]),
    "08": ("08_trained_superposition_aliasing.py", ["--feature-counts", "8,10", "--bottlenecks", "2,4", "--sample-sizes", "16,64,128", "--joint-probabilities", "0,0.1,0.2", "--tasks", "sum,pair_contrast", "--encoder-samples", "1024", "--num-seeds", "4", "--num-readout-draws", "2", "--epochs", "250"]),
    "09": ("09_double_descent_decomposition.py", ["--widths", "8,16,24,32,40", "--sample-sizes", "16,24,32,40,64", "--covariances", "isotropic,spiked", "--num-seeds", "8", "--n-test", "1024"]),
    "10": ("10_near_collision_noise.py", ["--separations", "0,0.03,0.3,1", "--noise-levels", "0.03,0.1,0.3", "--sample-sizes", "8,32,128", "--priors", "0.2,0.5", "--num-seeds", "8", "--n-test", "1024", "--quadrature-order", "40"]),
    "11": ("11_collision_mass_law.py", ["--collision-masses", "0.1,0.5,0.9", "--conditional-masses", "0.1,0.5,0.9", "--separations", "0,1e-3,0.1", "--sample-sizes", "8,32,128", "--num-seeds", "8"]),
    "12": ("12_matched_spectrum_tasks.py", ["--num-seeds", "8", "--num-draws", "2"]),
    "13": ("13_frozen_opt_gad.py", ["--layers", "0,5,11", "--widths", "8,16,32", "--sample-sizes", "16,32,64,128", "--num-readout-draws", "3", "--max-tokens-per-split", "1024"]),
}
GENERALIZED = (
    "glm-samplewise", "glm-featurewise", "frozen-superposition",
    "learned-superposition", "gaussian-superposition", "no-bias-superposition",
    "embedded-complexity", "embedded-mechanics", "embedded-regularization",
    "embedded-heatmap",
)
STOP = False


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def stop(_signal: int, _frame: object) -> None:
    global STOP
    STOP = True


def slug(path: Path) -> str:
    return path.stem.split("_", 1)[0] if path.stem[:2].isdigit() else path.stem


def steps(repo: Path) -> list[tuple[str, Path, list[str] | None]]:
    numbered = {slug(p): p for p in (repo / "experiments").glob("[0-9][0-9]_*.py")}
    ordered = []
    for key, (filename, overrides) in KNOWN.items():
        script = repo / "experiments" / filename
        if script.exists():
            ordered.append((key, script, overrides))
        elif key in numbered:
            ordered.append((key, numbered[key], None))
    for key, script in sorted(numbered.items()):
        if key not in KNOWN:
            ordered.append((key, script, None))
    for key in GENERALIZED:
        ordered.append((key, repo / "generalized_gad" / "experiment_cli.py", None))
    for script in sorted((repo / "generalized_gad" / "ggad_old").glob("*.py")):
        ordered.append(("legacy-" + script.stem, script, None))
    # OPT uses the remaining budget after the standalone and synthetic runs.
    ordered.sort(key=lambda r: r[0] == "13")
    return ordered


def optional_opt(repo: Path) -> tuple[list[str], str]:
    saved = os.getenv("GAD_FEATURES_NPZ")
    text = os.getenv("GAD_TEXT_FILE")
    if saved:
        path = (repo / saved).resolve()
        return (["--from-npz", str(path)], "" if path.is_file() else f"NPZ missing: {path}")
    if text:
        path = (repo / text).resolve()
        extra = ["--text-file", str(path), "--model-id", os.getenv("GAD_MODEL_ID", "facebook/opt-125m"),
                 "--device", os.getenv("GAD_DEVICE", "cpu")]
        return (extra, "" if path.is_file() else f"text corpus missing: {path}")
    return [], "requires GAD_FEATURES_NPZ or GAD_TEXT_FILE (plus a cached model)"


def command(repo: Path, python: str, key: str, script: Path, overrides: list[str] | None,
            target: Path, workers: int) -> tuple[list[str], str]:
    if not script.is_file():
        return [], f"script missing: {script}"
    if key in GENERALIZED:
        cmd = [python, str(script), "run", key, "--output-root", str(target)]
        if key.startswith("glm-"):
            cmd += ["--", "--num-seeds", "6", "--num-workers", str(workers)]
        return cmd, ""
    if key.startswith("legacy-"):
        return [python, str(script)], ""
    args = list(overrides or [])
    if "--num-workers" in args:
        args[args.index("--num-workers") + 1] = str(workers)
    if key == "13":
        inputs, reason = optional_opt(repo)
        if reason:
            return [], reason
        args += inputs
    if overrides is None:
        # Unknown numbered scripts are tried in isolation, with an output path
        # only if the script declares that argument. This avoids guessing APIs.
        source = script.read_text(encoding="utf-8")
        if "--num-seeds" in source:
            args += ["--num-seeds", "6"]
        if "--num-workers" in source:
            args += ["--num-workers", str(workers)]
        if "--epochs" in source:
            args += ["--epochs", "250"]
        if "--output-dir" in source:
            args += ["--output-dir", str(target)]
    else:
        args += ["--output-dir", str(target)]
    return [python, str(script), *args], ""


def dump(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def artifacts(path: Path, root: Path) -> list[str]:
    return [p.relative_to(root).as_posix() for p in sorted(path.rglob("*"))
            if p.is_file() and p.suffix.lower() in {".csv", ".png", ".npz", ".json"}]


def markdown(state: dict, root: Path) -> None:
    lines = ["# Overnight experiment results", "",
             f"Started (UTC): {state['started_utc']}  ",
             f"Updated (UTC): {state['updated_utc']}  ",
             f"Git revision: `{state['git_revision']}`  ",
             "Profile: bounded overnight survey. Reduced trials, epochs and grid sizes; see `results.json`.", "",
             "| Experiment | Status | Minutes | Results | Log |", "| --- | --- | ---: | --- | --- |"]
    for key, row in state["experiments"].items():
        rel = row.get("directory", "")
        directory = f"[{rel}]({rel}/)" if rel else "—"
        log = f"[log]({row['log']})" if row.get("log") else "—"
        lines.append(f"| {key} | {row['status']} | {row.get('minutes', 0):.1f} | {directory} | {log} |")
    lines += ["", "## Notes", ""]
    for key, row in state["experiments"].items():
        if row.get("reason"):
            lines.append(f"- **{key}:** {row['reason'].replace(chr(10), ' ')}")
    lines += ["", "[Open the figure gallery](GALLERY.md). Source scripts record their own mathematical caveats.", ""]
    (root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    gallery = ["# Figures from this suite", "", "See [SUMMARY.md](SUMMARY.md) for status and logs.", ""]
    for key, row in state["experiments"].items():
        pngs = [p for p in row.get("artifacts", []) if p.endswith(".png")]
        if pngs:
            gallery.extend([f"## {key}", ""])
            for png in pngs:
                gallery.extend([f"![{png}]({png})", ""])
    (root / "GALLERY.md").write_text("\n".join(gallery), encoding="utf-8")


def compact_bundle(root: Path) -> None:
    """Share plots and aggregate tables without the large per-trial raw files."""
    bundle = root / "results_bundle.zip"
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if path == bundle or not path.is_file():
                continue
            if path.suffix.lower() in {".png", ".json", ".md"} or (
                path.suffix.lower() == ".csv" and path.name not in {"raw.csv", "replicates.csv"}
            ):
                archive.write(path, arcname=path.relative_to(root))


def terminate(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument("--output-root", type=Path, default=Path("outputs/overnight"))
    ap.add_argument("--run-dir", type=Path, help="Exact suite path; required for --resume")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", help="Comma-separated experiment names for a targeted run")
    ap.add_argument("--max-hours", type=float, default=11.5)
    ap.add_argument("--step-minutes", type=float, default=25)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    repo = a.repo_root.resolve()
    python = sys.executable
    allocated = max(1, int(os.getenv("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 1))))
    workers = max(1, min(allocated, 28, int(os.getenv("GAD_MAX_WORKERS", "12"))))
    chosen = steps(repo)
    if a.only:
        wanted = set(a.only.split(","))
        chosen = [s for s in chosen if s[0] in wanted]
        if wanted - {s[0] for s in chosen}:
            ap.error(f"unknown experiment(s): {sorted(wanted - {s[0] for s in chosen})}")
    if not chosen or a.max_hours <= 0 or a.step_minutes <= 0:
        ap.error("select experiments and positive time limits")
    root = (a.run_dir if a.run_dir else a.output_root / f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{os.getenv('SLURM_JOB_ID', os.getpid())}")
    if not root.is_absolute():
        root = repo / root
    root = root.resolve()
    if a.resume and not a.run_dir:
        ap.error("--resume requires --run-dir")
    if a.dry_run:
        for key, script, overrides in chosen:
            target = root / key
            cmd, reason = command(repo, python, key, script, overrides, target, workers)
            print(key, "SKIPPED: " + reason if reason else shlex.join(cmd))
        return 0
    if root.exists() and not a.resume:
        ap.error(f"run directory already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "results.json"
    if a.resume and manifest.exists():
        state = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                  text=True, check=False)
        state = {"started_utc": utc(), "git_revision": revision.stdout.strip() if revision.returncode == 0 else "unknown",
                 "python": python, "parameters": {"workers": workers, "step_minutes": a.step_minutes,
                 "max_hours": a.max_hours, "GAD_EPOCHS": os.getenv("GAD_EPOCHS", "200"),
                 "GAD_TRIALS": os.getenv("GAD_TRIALS", "3"),
                 "GAD_GRID_STRIDE": os.getenv("GAD_GRID_STRIDE", "4"),
                 "GAD_GLOBAL_SAMPLES": os.getenv("GAD_GLOBAL_SAMPLES", "2000")},
                 "experiments": {}}
    for key, _, _ in chosen:
        state["experiments"].setdefault(key, {"status": "not_run"})
    deadline = time.monotonic() + a.max_hours * 3600
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                "GAD_MAX_WORKERS": str(workers),
                "GAD_EPOCHS": os.getenv("GAD_EPOCHS", "200"),
                "GAD_TRIALS": os.getenv("GAD_TRIALS", "3"),
                "GAD_GRID_STRIDE": os.getenv("GAD_GRID_STRIDE", "4"),
                "GAD_GLOBAL_SAMPLES": os.getenv("GAD_GLOBAL_SAMPLES", "2000"),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"Suite: {root}", flush=True)
    for key, script, overrides in chosen:
        previous = state["experiments"][key]
        if a.resume and previous["status"] == "completed":
            continue
        row = {"status": "not_run"}
        state["experiments"][key] = row
        attempts = [root / key] + [root / f"{key}-retry{i}" for i in range(2, 100)]
        target = next((p for p in attempts if not p.exists()), None)
        if target is None:
            row.update(status="failed", reason="too many previous attempts")
            continue
        cmd, reason = command(repo, python, key, script, overrides, target, workers)
        remaining = deadline - time.monotonic() - 300
        if STOP or remaining <= 0:
            row.update(status="not_run", reason="suite stopped or time budget exhausted")
        elif reason:
            row.update(status="skipped", reason=reason)
        else:
            target.mkdir(parents=True, exist_ok=False)
            log = target / "stdout.log"
            limit = min(a.step_minutes * 60, remaining)
            row.update(status="running", command=cmd, directory=target.relative_to(root).as_posix(),
                       log=log.relative_to(root).as_posix(), script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                       started_utc=utc(), time_limit_seconds=round(limit))
            state["updated_utc"] = utc(); dump(manifest, state); markdown(state, root)
            print(f"[{key}] {shlex.join(cmd)}", flush=True)
            start = time.monotonic()
            with log.open("w", encoding="utf-8") as stream:
                child_cwd = target if key.startswith("legacy-") or (overrides is None and
                    key not in GENERALIZED and "--output-dir" not in cmd) else repo
                child_env = env.copy()
                if key == "13" and os.getenv("GAD_DEVICE", "cpu") == "cpu" and not os.getenv("GAD_FEATURES_NPZ"):
                    child_env["OMP_NUM_THREADS"] = child_env["MKL_NUM_THREADS"] = str(workers)
                process = subprocess.Popen(cmd, cwd=child_cwd, env=child_env, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while process.poll() is None and not STOP and time.monotonic()-start < limit:
                        time.sleep(1)
                finally:
                    if process.poll() is None:
                        terminate(process)
            row["minutes"] = round((time.monotonic()-start)/60, 2)
            row["exit_code"] = process.returncode
            row["artifacts"] = artifacts(target, root)
            row["finished_utc"] = utc()
            if STOP:
                row.update(status="interrupted", reason="received scheduler signal")
            elif time.monotonic()-start >= limit and process.returncode != 0:
                row.update(status="timed_out", reason=f"exceeded {limit/60:.1f} minutes")
            elif process.returncode:
                with log.open(encoding="utf-8", errors="replace") as stream:
                    tail = list(deque(stream, maxlen=8))
                row.update(status="failed", reason=" | ".join(tail)[-800:])
            elif not row["artifacts"]:
                row.update(status="completed_no_artifact", reason="script returned 0 without a PNG/CSV/NPZ/JSON")
            else:
                row["status"] = "completed"
            print(f"[{key}] {row['status']} ({row['minutes']} min)", flush=True)
        state["updated_utc"] = utc(); dump(manifest, state); markdown(state, root)
        if STOP:
            break
    state["updated_utc"] = utc(); dump(manifest, state); markdown(state, root)
    compact_bundle(root)
    print(f"Final report: {root/'SUMMARY.md'}", flush=True)
    print(f"Shareable figures and aggregate tables: {root/'results_bundle.zip'}", flush=True)
    failed = {"failed", "timed_out", "interrupted", "completed_no_artifact", "not_run"}
    return 1 if any(v["status"] in failed for v in state["experiments"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
