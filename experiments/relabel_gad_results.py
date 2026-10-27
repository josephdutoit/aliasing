"""Copy legacy finite_support_gad CSVs to a new directory with GAD-v2 labels.

python experiments/relabel_gad_results.py OLD_RUN --output-dir CORRECTED_RUN
Only CSVs are copied; legacy figures and manifests retain their old meanings.
No simulation is rerun. Applies to experiments 05--13, not linear.py or GLMs.
"""
import argparse
import csv
import json
from pathlib import Path


MAPPING = {
    "aliasing": "data_insufficiency",
    "data_algorithm": "aliasing",
    "cross_model_alias": "cross_model_data",
    "cross_model_data": "cross_model_alias",
    "alias_operator_rms": "data_insufficiency_operator_frobenius",
    "balanced_exact_alias_control": "balanced_exact_data_control",
}


def new_name(key):
    for old, new in MAPPING.items():
        if key == old or key.startswith(old + "_ci95_"):
            return new + key[len(old):]
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output_dir.resolve()
    if not source.is_dir() or output.exists() or source in output.parents:
        parser.error("source must exist; choose a new destination outside the source tree")
    files = sorted(source.rglob("*.csv"))
    if not files:
        parser.error("no CSV files found")
    output.mkdir(parents=True)
    changed = []
    for path in files:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames or [], list(reader)
        legacy = "data_algorithm" in fields or "data_algorithm_ci95_low" in fields
        destination = output / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if legacy:
            rows = [{new_name(k): v for k, v in row.items()} | {"gad_definition_version": "2"} for row in rows]
            fields = [new_name(k) for k in fields] + ["gad_definition_version"]
            changed.append(path.relative_to(source).as_posix())
        with destination.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    (output / "GAD_RELABEL_MANIFEST.json").write_text(json.dumps({
        "source": str(source), "gad_definition_version": 2,
        "mapping": MAPPING, "relabeled_csvs": changed,
        "note": "Numerical values unchanged. Cross-term names and CI bounds relabeled consistently. Legacy images not copied."
    }, indent=2) + "\n")
    print(f"Relabeled {len(changed)} CSVs; source files preserved. Output: {output}")


if __name__ == "__main__":
    main()
