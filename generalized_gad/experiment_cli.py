"""Run legacy scripts in separate directories and index their PNGs.

Examples:
    python generalized_gad/experiment_cli.py list
    python generalized_gad/experiment_cli.py run embedded-complexity --dry-run
    python generalized_gad/experiment_cli.py run glm-samplewise -- --num-seeds 4 --num-workers 2
    python generalized_gad/experiment_cli.py index
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote

from experiment_registry import EXPERIMENTS


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GLM_NAMES = {"glm-samplewise", "glm-featurewise"}


def index(output_root: Path, destination: Path) -> None:
    """Create a portable Markdown gallery from historical and isolated runs."""
    images = sorted(HERE.glob("*.png"))
    images += sorted(output_root.glob("*/*/*.png"))
    lines = ["# Generalized GAD results index", "",
             "The first section contains the original uploaded figures. New runs are below.",
             "Plots should be interpreted using [README.md](README.md); images alone do not establish the stated mechanism.", ""]
    for label, files in (("Historical figures", [p for p in images if p.parent == HERE]),
                         ("Isolated runs", [p for p in images if p.parent != HERE])):
        lines.extend([f"## {label}", ""])
        if not files:
            lines.extend(["No figures found.", ""])
        for file in files:
            rel = os.path.relpath(file, destination.parent).replace(os.sep, "/")
            link = quote(rel, safe="/._-")
            lines.extend([f"### {file.stem.replace('_', ' ')}", "",
                          f"![{file.stem}]({link})", ""])
            manifest = file.parent / "manifest.json"
            if manifest.exists():
                manifest_rel = os.path.relpath(manifest, destination.parent).replace(os.sep, "/")
                lines.extend([f"[Run manifest]({quote(manifest_rel, safe='/._-')})", ""])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"Indexed {len(images)} figures: {destination}")


def run(args: argparse.Namespace) -> int:
    script_name, _ = EXPERIMENTS[args.name]
    script = HERE / script_name
    if not script.is_file():
        raise FileNotFoundError(script)
    extra = list(args.script_args)
    if extra and extra[0] == "--":
        extra.pop(0)
    if extra and args.name not in GLM_NAMES:
        raise ValueError("Only GLM scripts currently accept additional arguments")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / args.name / f"{stamp}-{os.getpid()}"
    command = [sys.executable, str(script), *extra]
    print("Run:", " ".join(command))
    print("Results:", run_dir)
    if args.dry_run:
        return 0
    run_dir.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    manifest = {
        "experiment": args.name,
        "script": str(script.relative_to(ROOT)),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "command": command,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "running",
        "notes": "Legacy script defaults and random seeds are defined in the script; see README.md.",
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as log:
            status = subprocess.run(command, cwd=run_dir, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    except BaseException as exc:
        manifest["status"] = "interrupted"
        manifest["error"] = repr(exc)
        raise
    else:
        manifest["returncode"] = status
        manifest["status"] = "completed" if status == 0 else "failed"
        return status
    finally:
        manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest["figures"] = sorted(p.name for p in run_dir.glob("*.png"))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("Log:", run_dir / "stdout.log")
        print("Manifest:", manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("list", help="List named experiments")
    run_parser = subparsers.add_parser("run", help="Run an existing experiment in its own output directory")
    run_parser.add_argument("name", choices=sorted(EXPERIMENTS))
    run_parser.add_argument("--output-root", default=str(HERE / "runs"))
    run_parser.add_argument("--dry-run", action="store_true")
    index_parser = subparsers.add_parser("index", help="Generate a gallery of old and new figures")
    index_parser.add_argument("--output-root", default=str(HERE / "runs"))
    index_parser.add_argument("--destination", default=str(HERE / "RESULTS_INDEX.md"))
    args, extras = parser.parse_known_args()
    if args.action != "run" and extras:
        parser.error(f"unrecognized arguments: {' '.join(extras)}")
    args.script_args = extras
    if args.action == "list":
        for name, (script, purpose) in EXPERIMENTS.items():
            print(f"{name:26} {script:29} {purpose}")
        return 0
    if args.action == "index":
        index(Path(args.output_root).resolve(), Path(args.destination).resolve())
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
