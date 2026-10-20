"""Repair legacy diagonal-only Independent matrices without rerunning models.

The diagonal contains every measured task accuracy.  Because the predictors are
independent and frozen, the unique canonical CL representation copies each
diagonal value down its column.  This migration preserves an adjacent backup
and emits a JSON audit record with hashes and repaired keys.
"""

import argparse
import hashlib
import json
import pathlib

import numpy as np


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(diagonal):
    diagonal = np.asarray(diagonal, dtype=float)
    return np.tril(np.broadcast_to(diagonal, (len(diagonal), len(diagonal)))).copy()


def repair(path):
    path = pathlib.Path(path)
    backup = path.with_name(path.stem + ".legacy-diagonal" + path.suffix)
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")

    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    repaired = []
    for key, matrix in list(arrays.items()):
        if "/independent/" not in key:
            continue
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{path}:{key} is not a square result matrix")
        expected = _canonical(np.diag(matrix))
        if np.array_equal(matrix, expected):
            continue
        off_diagonal = matrix.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        if np.count_nonzero(off_diagonal):
            raise ValueError(f"{path}:{key} is not diagonal-only; refusing repair")
        arrays[key] = expected
        repaired.append(key)

    if not repaired:
        return None

    original_hash = _sha256(path)
    path.rename(backup)
    try:
        np.savez(path, **arrays)
    except Exception:
        backup.rename(path)
        raise
    return {
        "file": str(path),
        "backup": str(backup),
        "original_sha256": original_hash,
        "repaired_sha256": _sha256(path),
        "repaired_keys": repaired,
        "transformation": "R[i,j]=diag(R)[j] for i>=j; zero above diagonal",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="NPZ files to repair in place")
    parser.add_argument("--audit", required=True, help="Output JSON audit record")
    args = parser.parse_args()
    records = [record for record in (repair(path) for path in args.paths)
               if record is not None]
    payload = {"migration": "independent_matrix_v1_to_v2", "records": records}
    with open(args.audit, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Repaired {sum(len(r['repaired_keys']) for r in records)} matrices "
          f"across {len(records)} archives; audit: {args.audit}")


if __name__ == "__main__":
    main()
