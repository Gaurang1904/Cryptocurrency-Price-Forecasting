import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from crypto.evaluation import (
    metric_tables, validate_comparable_predictions, validate_predictions,
)
from crypto.evaluation_plots import render_bundle, require_volatility_baseline


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root_relative(path, provenance_root):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(provenance_root).as_posix()
    except ValueError:
        raise ValueError(
            f"path is outside provenance root {provenance_root}: {resolved}"
        ) from None


def _portable_metadata(value, provenance_root):
    if isinstance(value, dict):
        return {
            key: _portable_metadata(item, provenance_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portable_metadata(item, provenance_root) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return _root_relative(value, provenance_root)
    return value


def _input_record(path, provenance_root):
    path = Path(path)
    metadata_path = path.with_name("metadata.json")
    prediction_path = _root_relative(path, provenance_root)
    relative_metadata_path = _root_relative(metadata_path, provenance_root)
    metadata = _portable_metadata(
        json.loads(metadata_path.read_text(encoding="utf-8")),
        provenance_root,
    )
    return {
        "path": prediction_path,
        "sha256": _sha256(path),
        "metadata": metadata,
        "metadata_path": relative_metadata_path,
        "metadata_sha256": _sha256(metadata_path),
    }


def _manifest(frame, inputs, output_paths, provenance_root):
    cutoffs = {pd.Timestamp(item["metadata"]["data_end"]).isoformat()
               for item in inputs}
    if len(cutoffs) != 1:
        raise ValueError(f"input metadata data cutoffs differ: {sorted(cutoffs)}")
    folds = sorted(pd.Timestamp(value).isoformat() for value in frame.fold.unique())
    return {
        "manifest_version": 1,
        "hash_algorithm": "sha256",
        "provenance": {
            "root": ".",
            "path_format": "provenance-root-relative-posix",
        },
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
            {
                "path": _root_relative(path, provenance_root),
                "sha256": _sha256(path),
            }
            for path in sorted(output_paths)
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render daily OOS evaluation charts.")
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--provenance-root", type=Path, default=Path.cwd(),
        help="Root for portable manifest paths; defaults to the invocation directory.",
    )
    args = parser.parse_args(argv)

    provenance_root = args.provenance_root.resolve()
    if not provenance_root.is_dir():
        raise ValueError(
            f"provenance root is not a directory: {provenance_root}"
        )
    for path in [*args.predictions, args.out]:
        _root_relative(path, provenance_root)

    frames = [pd.read_parquet(path) for path in args.predictions]
    inputs = [
        _input_record(path, provenance_root) for path in args.predictions
    ]
    validate_comparable_predictions(
        frames, [item["metadata"] for item in inputs]
    )
    frame = pd.concat(frames, ignore_index=True)
    validate_predictions(frame)
    require_volatility_baseline(frame)
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

    manifest = _manifest(frame, inputs, output_paths, provenance_root)
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
