import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from crypto.evaluation import metric_tables, validate_predictions
from crypto.evaluation_plots import render_bundle, require_volatility_baseline


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_fields(path):
    resolved = Path(path).resolve()
    try:
        return {"path": resolved.relative_to(Path.cwd().resolve()).as_posix()}
    except ValueError:
        return {"path": resolved.name, "absolute_path": resolved.as_posix()}


def _input_record(path):
    path = Path(path)
    metadata_path = path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    record = _path_fields(path) | {"sha256": _sha256(path), "metadata": metadata}
    metadata_fields = _path_fields(metadata_path)
    record["metadata_path"] = metadata_fields.pop("path")
    if metadata_fields:
        record["metadata_absolute_path"] = metadata_fields["absolute_path"]
    record["metadata_sha256"] = _sha256(metadata_path)
    return record


def _manifest(frame, inputs, output_paths):
    cutoffs = {pd.Timestamp(item["metadata"]["data_end"]).isoformat()
               for item in inputs}
    if len(cutoffs) != 1:
        raise ValueError(f"input metadata data cutoffs differ: {sorted(cutoffs)}")
    folds = sorted(pd.Timestamp(value).isoformat() for value in frame.fold.unique())
    return {
        "manifest_version": 1,
        "hash_algorithm": "sha256",
        "evaluation": {
            "data_cutoff": cutoffs.pop(),
            "oos_start": pd.Timestamp(frame.origin.min()).isoformat(),
            "oos_end": pd.Timestamp(frame.origin.max()).isoformat(),
            "folds": folds,
            "fold_count": len(folds),
            "distinct_origins": frame[["asset", "origin"]].drop_duplicates().shape[0],
            "row_count": len(frame),
            "models": sorted(frame.model.unique().tolist()),
            "horizons": sorted(int(value) for value in frame.h.unique()),
        },
        "inputs": inputs,
        "outputs": [
            _path_fields(path) | {"sha256": _sha256(path)}
            for path in sorted(output_paths)
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render daily OOS evaluation charts.")
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    frames = [pd.read_parquet(path) for path in args.predictions]
    frame = pd.concat(frames, ignore_index=True)
    validate_predictions(frame)
    require_volatility_baseline(frame)
    inputs = [_input_record(path) for path in args.predictions]
    cutoffs = {pd.Timestamp(item["metadata"]["data_end"]).isoformat()
               for item in inputs}
    if len(cutoffs) != 1:
        raise ValueError(f"input metadata data cutoffs differ: {sorted(cutoffs)}")

    args.out.mkdir(parents=True, exist_ok=False)
    output_paths = []
    for name, table in metric_tables(frame).items():
        path = args.out / f"metrics_{name}.csv"
        table.to_csv(path)
        output_paths.append(path)
    render_bundle(frame, args.out)
    output_paths.extend(sorted(args.out.glob("*.png")))

    manifest = _manifest(frame, inputs, output_paths)
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
