"""Backfill quantile stats (q01/q10/q50/q90/q99) into converted datasets.

Newer LeRobot versions compute per-feature quantiles into ``meta/stats.json``
natively; datasets converted from older sources carry only
count/max/mean/min/std. This tool computes the missing quantiles for the
parquet-backed ``action`` and ``observation.state`` features (exact, over
every frame — image features are left untouched: they would need video
decoding and no downstream consumer reads image quantiles here) and merges
them into each sub-dataset's ``meta/stats.json``, preserving all existing
keys.

Usage:
    uv run python -m ldtools.backfill_quantile_stats \
        --root /data/community_dataset_v1_v3

    # a subset / recompute-and-overwrite / inspect without writing
    ... --datasets ZGGZZG/so100_drop0 ad330/cubePlace
    ... --force
    ... --dry-run

Properties:
  - idempotent: sub-datasets whose stats already carry every quantile key
    for both features are skipped (unless ``--force``).
  - **use --force to standardize a corpus**: LeRobot's native quantiles are
    aggregated from per-episode stats, which SHRINKS ranges badly on
    dimensions that are near-constant within episodes (measured on a
    50-episode SO-101 set: native q01 −54° vs exact −120°). Exact corpus
    quantiles are the FAST paper's definition and the standard here; a
    corpus mixing native and exact provenance normalizes inconsistently.
  - ``--dry-run`` prints the max absolute difference vs existing values
    without writing anything.
  - failures are collected and reported at the end; exit code 1 if any.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILE_KEYS: dict[str, float] = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}
FEATURES = ("action", "observation.state")


def log(name: str, message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{name}] {message}", flush=True)


@dataclass(frozen=True)
class BackfillResult:
    name: str
    status: str  # "backfilled" | "skipped" | "dry-run" | "failed"
    detail: str


def discover(root: Path) -> list[tuple[str, Path]]:
    """(name, path) for every ``<root>/<user>/<dataset>`` with meta/stats.json."""
    found = [
        (f"{stats.parent.parent.parent.name}/{stats.parent.parent.name}", stats.parent.parent)
        for stats in sorted(root.glob("*/*/meta/stats.json"))
    ]
    if not found:
        raise SystemExit(f"no sub-datasets with meta/stats.json under {root}")
    return found


def has_all_quantiles(stats: dict) -> bool:
    return all(key in stats.get(feature, {}) for feature in FEATURES for key in QUANTILE_KEYS)


def feature_quantiles(dataset_dir: Path) -> dict[str, dict[str, list[float]]]:
    """Exact per-dimension quantiles over every frame of each feature."""
    frames = [
        pd.read_parquet(path, columns=list(FEATURES))
        for path in sorted((dataset_dir / "data").rglob("*.parquet"))
    ]
    if not frames:
        raise ValueError("no parquet files under data/")
    table = pd.concat(frames)
    result: dict[str, dict[str, list[float]]] = {}
    for feature in FEATURES:
        values = np.stack(list(table[feature].to_numpy())).astype(np.float64)
        result[feature] = {
            key: np.quantile(values, q, axis=0).tolist() for key, q in QUANTILE_KEYS.items()
        }
    return result


def compare_existing(stats: dict, computed: dict[str, dict[str, list[float]]]) -> str:
    """Max |computed - existing| across features/keys present in both."""
    deltas = [
        float(np.max(np.abs(np.asarray(computed[feature][key]) - np.asarray(existing[key]))))
        for feature in FEATURES
        if (existing := stats.get(feature)) is not None
        for key in QUANTILE_KEYS
        if key in existing
    ]
    if not deltas:
        return "no existing quantiles"
    return f"max |computed - existing| = {max(deltas):.4f} over {len(deltas)} entries"


def backfill_one(name: str, dataset_dir: Path, *, force: bool, dry_run: bool) -> BackfillResult:
    stats_path = dataset_dir / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    missing = [feature for feature in FEATURES if feature not in stats]
    if missing:
        return BackfillResult(name, "failed", f"features missing from stats.json: {missing}")
    if has_all_quantiles(stats) and not force:
        return BackfillResult(name, "skipped", "quantiles already present")

    computed = feature_quantiles(dataset_dir)
    if dry_run:
        return BackfillResult(name, "dry-run", compare_existing(stats, computed))
    for feature in FEATURES:
        stats[feature].update(computed[feature])
    stats_path.write_text(json.dumps(stats, indent=4))
    n_frames = int(stats["action"].get("count", [0])[0])
    return BackfillResult(name, "backfilled", f"{n_frames} frames")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill exact quantile stats (q01/q10/q50/q90/q99) "
        "into converted datasets' meta/stats.json",
    )
    parser.add_argument("--root", type=Path, required=True, help="collection root")
    parser.add_argument("--datasets", nargs="*", help="subset of <user>/<dataset> names")
    parser.add_argument("--force", action="store_true", help="recompute even when present")
    parser.add_argument("--dry-run", action="store_true", help="compute and report, never write")
    parser.add_argument("--workers", type=int, default=8, help="parallel dataset scans")
    args = parser.parse_args()

    pairs = discover(args.root)
    if args.datasets:
        wanted = set(args.datasets)
        pairs = [(name, path) for name, path in pairs if name in wanted]
        unknown = wanted - {name for name, _ in pairs}
        if unknown:
            raise SystemExit(f"not found under {args.root}: {sorted(unknown)}")
    log("backfill", f"{len(pairs)} sub-datasets under {args.root}")

    results: list[BackfillResult] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(backfill_one, name, path, force=args.force, dry_run=args.dry_run): name
            for name, path in pairs
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 - quarantine, report at the end
                result = BackfillResult(name, "failed", f"{type(error).__name__}: {error}")
            results.append(result)
            log(name, f"{result.status}: {result.detail}")

    by_status: dict[str, int] = {}
    for result in results:
        by_status[result.status] = by_status.get(result.status, 0) + 1
    log("backfill", f"done: {by_status}")
    failed = [result for result in results if result.status == "failed"]
    for result in failed:
        log(result.name, f"FAILED: {result.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
