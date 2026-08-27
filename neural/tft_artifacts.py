import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import uuid4

from crypto.evaluation import reserve_run_dir


REQUIRED_METADATA = {
    "run_id",
    "pipeline",
    "family",
    "data_path",
    "data_end",
    "data_start",
    "assets",
    "rows",
    "gap_stats",
    "features",
    "future_features",
    "config",
    "train_end",
    "validation_start",
    "validation_end",
    "calibration_start",
    "calibration_end",
    "test_start",
    "test_end",
    "package_versions",
    "git_commit",
    "elapsed_seconds",
    "device",
}

_CORE_FILES = {
    "raw_cv.parquet",
    "predictions_raw_test.parquet",
    "predictions_calibrated_test.parquet",
    "metadata.json",
}
_UNHASHED_FILES = {"manifest.json", "status.json"}


def _atomic_json(path, value, *, overwrite):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8"
        )
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(path, frame):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.link(temporary, path)
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance_relative(path, provenance_root):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(provenance_root).as_posix()
    except ValueError:
        raise ValueError(
            f"path is outside provenance root {provenance_root}: {resolved}"
        ) from None


def _is_path_key(key):
    return isinstance(key, str) and (
        key in {"path", "paths"} or key.endswith(("_path", "_paths"))
    )


def _portable_path(value, provenance_root):
    path = Path(value)
    serialized = str(value)
    if not path.is_absolute() and (
        "\\" in serialized or PureWindowsPath(serialized).drive
    ):
        raise ValueError(f"metadata path uses Windows path syntax: {value}")
    if path.anchor and not path.is_absolute():
        raise ValueError(f"metadata path is not a fully relative path: {value}")
    resolved = path if path.is_absolute() else provenance_root / path
    return _provenance_relative(resolved, provenance_root)


def _portable_metadata(value, provenance_root, path_value=False):
    if isinstance(value, dict):
        return {
            key: _portable_metadata(
                item, provenance_root, path_value=_is_path_key(key)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _portable_metadata(item, provenance_root, path_value=path_value)
            for item in value
        ]
    if isinstance(value, Path):
        return _portable_path(value, provenance_root)
    if isinstance(value, str) and (path_value or Path(value).is_absolute()):
        return _portable_path(value, provenance_root)
    return value


def _contained_path(path, output_dir):
    path = Path(path)
    if not path.is_absolute():
        path = output_dir / path
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(output_dir)
    except ValueError:
        raise ValueError(f"evidence path is outside run directory: {resolved}") from None
    return relative.as_posix(), resolved


def _contained_file(path, output_dir):
    relative, resolved = _contained_path(path, output_dir)
    if not resolved.is_file():
        raise FileNotFoundError(f"evidence file is missing: {resolved}")
    return relative, resolved


def _manifest_file(path, output_dir):
    if not isinstance(path, str):
        raise ValueError("manifest path must be a canonical relative POSIX path")
    posix = PurePosixPath(path)
    windows = PureWindowsPath(path)
    if (
        not path
        or "\\" in path
        or posix.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or posix.as_posix() != path
    ):
        raise ValueError(
            f"manifest path must be a canonical relative POSIX path: {path!r}"
        )
    resolved = (output_dir / path).resolve()
    try:
        resolved.relative_to(output_dir)
    except ValueError:
        raise ValueError(f"evidence path is outside run directory: {resolved}") from None
    return path, resolved


def _evidence_files(output_dir):
    files = {}
    for path in output_dir.rglob("*"):
        relative = path.relative_to(output_dir).as_posix()
        if path.is_file() and relative not in _UNHASHED_FILES:
            files[relative] = path
    return files


def reserve_tft_run(data_end, run_id, root=Path("artifacts/evaluation")):
    output_dir = reserve_run_dir(
        "tft", data_end, run_id, root, pipeline="hf15m"
    )
    _atomic_json(output_dir / "status.json", {"state": "incomplete"}, overwrite=True)
    return output_dir


def write_raw_cv(output_dir, raw_cv):
    _atomic_parquet(Path(output_dir) / "raw_cv.parquet", raw_cv)


def save_tft_core(output_dir, nf, raw_test, calibrated_test):
    output_dir = Path(output_dir)
    targets = [
        output_dir / "model",
        output_dir / "predictions_raw_test.parquet",
        output_dir / "predictions_calibrated_test.parquet",
    ]
    for target in targets:
        if target.exists():
            raise FileExistsError(f"artifact already exists: {target}")

    nf.save(str(targets[0]), save_dataset=True, overwrite=False)
    _atomic_parquet(targets[1], raw_test)
    _atomic_parquet(targets[2], calibrated_test)


def finalize_tft_run(
    output_dir, metadata, extra_paths=(), provenance_root=None
):
    output_dir = Path(output_dir).resolve()
    missing = sorted(REQUIRED_METADATA - set(metadata))
    if missing:
        raise ValueError(f"missing required metadata fields: {missing}")

    provenance_root = Path.cwd().resolve() if provenance_root is None else Path(
        provenance_root
    ).resolve()
    portable_metadata = _portable_metadata(metadata, provenance_root)

    required_paths = [output_dir / name for name in _CORE_FILES - {"metadata.json"}]
    model_files = [path for path in (output_dir / "model").rglob("*") if path.is_file()]
    if not model_files:
        raise FileNotFoundError(f"checkpoint files are missing: {output_dir / 'model'}")
    for path in required_paths:
        _contained_file(path, output_dir)
    for path in extra_paths:
        _contained_file(path, output_dir)

    metadata_path = output_dir / "metadata.json"
    _atomic_json(metadata_path, portable_metadata, overwrite=False)
    files = _evidence_files(output_dir)
    manifest = {
        "manifest_version": 1,
        "hash_algorithm": "sha256",
        "files": [
            {"path": relative, "sha256": _sha256(path)}
            for relative, path in sorted(files.items())
        ],
    }
    _atomic_json(output_dir / "manifest.json", manifest, overwrite=False)
    verify_tft_manifest(output_dir)
    _atomic_json(output_dir / "status.json", {"state": "complete"}, overwrite=True)


def verify_tft_manifest(output_dir):
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("hash_algorithm") != "sha256":
        raise ValueError("manifest hash algorithm is not sha256")

    recorded = {}
    for record in manifest.get("files", []):
        relative, path = _manifest_file(record["path"], output_dir)
        if relative in recorded:
            raise ValueError(f"duplicate manifest path: {relative}")
        recorded[relative] = (path, record["sha256"])

    actual = _evidence_files(output_dir)
    missing = sorted(set(recorded) - set(actual))
    extra = sorted(set(actual) - set(recorded))
    if missing:
        raise ValueError(f"missing evidence files: {missing}")
    if extra:
        raise ValueError(f"extra evidence files: {extra}")
    for relative, (path, expected) in recorded.items():
        if _sha256(path) != expected:
            raise ValueError(f"hash mismatch for evidence file: {relative}")
