"""Mechanical episode filtering + merge of collections into one curated root.

Applies metadata/trajectory-level filters (no judge, no GPU) to every
dataset under the source collection roots and writes survivors into a
single combined collection ``<output>/<user>/<dataset>``:

- datasets with NO dropped episodes are HARDLINKED (zero copy; all our
  writers replace files atomically, so shared inodes are safe);
- datasets with drops are REBUILT: kept episodes renumbered 0..N-1, data
  parquet rewritten (one row group per episode), videos REMUXED with
  packet copy (episode boundaries are keyframe-aligned in the converted
  collections — measured 130/130 on 6 datasets; files that violate this
  fail the dataset loudly rather than re-encode, which would be lossy),
  per-episode stats carried over, dataset stats re-aggregated and exact
  action/state quantiles recomputed from the kept frames;
- datasets left with fewer than --min-episodes survivors are dropped
  entirely (tiny datasets carry per-dataset stats/holdout overhead for
  negligible frames).

Episode filters (first matching reason recorded in the manifest):

  nan_actions      NaN/inf anywhere in action or observation.state
  short            length < --min-frames (cannot fill one action chunk)
  brief            duration < --min-seconds
  marathon         length > the --max-frames-quantile of the INPUT scope
                   (operator forgot to stop recording)
  zero_travel      total action path < --min-travel (arm never moves)
  idle             fraction of steps where every motor moves < 1% of its
                   episode range >= --idle-max

Everything is loud: per-episode drop reasons, per-dataset outcomes and
validation results land in ``<output>/filter_manifest.jsonl``; a summary
prints at the end. Idempotent: datasets already present in the output are
skipped (--force rebuilds). Dataset-level scope (action/state dims, fps)
is asserted, not filtered — the download was already scoped; anything
out of scope in the sources is a loud error.

Usage:
    uv run python -m ldtools.filter_collections \
        --sources /data/source/v1 /data/source/v2 /data/source/v3 \
        --output /data/curated_v0 [--datasets u/d ...] [--dry-run] \
        [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd
from lerobot.datasets.compute_stats import aggregate_stats

from .backfill_quantile_stats import feature_quantiles

EXPECTED_DIMS = 6
EXPECTED_FPS = 30.0


@dataclass(frozen=True)
class Thresholds:
    """Episode-filter knobs (recorded verbatim in the manifest)."""

    min_frames: int
    min_seconds: float
    max_frames: int  # resolved from --max-frames-quantile over the scope
    min_travel: float
    idle_max: float
    min_episodes: int


@dataclass(frozen=True)
class EpisodePlan:
    keep: list[int]  # original episode indices, sorted
    drops: dict[int, str]  # original episode index -> first matching reason


def discover(sources: list[Path]) -> list[Path]:
    found: dict[str, Path] = {}
    for source in sources:
        for info in sorted(source.glob("*/*/meta/info.json")):
            dataset_dir = info.parent.parent
            repo_id = f"{dataset_dir.parent.name}/{dataset_dir.name}"
            if repo_id in found:
                raise SystemExit(f"duplicate dataset {repo_id}: {found[repo_id]} and {dataset_dir}")
            found[repo_id] = dataset_dir
    if not found:
        raise SystemExit(f"no datasets under {sources}")
    return [found[name] for name in sorted(found)]


def episode_lengths(dataset_dir: Path) -> pd.DataFrame:
    frames = [
        pd.read_parquet(p, columns=["episode_index", "length"])
        for p in sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    ]
    return pd.concat(frames).sort_values("episode_index")


def plan_episodes(dataset_dir: Path, thresholds: Thresholds, fps: float) -> EpisodePlan:
    """Decide keep/drop per episode from meta lengths + data parquet."""
    frame = episode_lengths(dataset_dir)
    lengths = {
        int(episode): int(length)
        for episode, length in zip(
            np.asarray(frame["episode_index"]).tolist(),
            np.asarray(frame["length"]).tolist(),
            strict=True,
        )
    }
    drops: dict[int, str] = {}

    # Metadata-only checks first (cheap).
    for episode, length in lengths.items():
        if length < thresholds.min_frames:
            drops[episode] = f"short ({length} frames)"
        elif length < thresholds.min_seconds * fps:
            drops[episode] = f"brief ({length / fps:.2f}s)"
        elif length > thresholds.max_frames:
            drops[episode] = f"marathon ({length} frames > {thresholds.max_frames})"

    # Trajectory checks over data parquet (skip episodes already dropped).
    data_frames = [
        pd.read_parquet(p, columns=["episode_index", "action", "observation.state"])
        for p in sorted((dataset_dir / "data").rglob("*.parquet"))
    ]
    table = pd.concat(data_frames)
    for episode_key, group in table.groupby("episode_index", sort=True):
        episode = int(episode_key)  # type: ignore[arg-type]  # stubs widen groupby keys to Hashable
        if episode in drops:
            continue
        action = np.stack(list(group["action"].to_numpy())).astype(np.float64)
        state = np.stack(list(group["observation.state"].to_numpy())).astype(np.float64)
        if not (np.isfinite(action).all() and np.isfinite(state).all()):
            drops[episode] = "nan_actions"
            continue
        delta = np.abs(np.diff(action, axis=0))
        travel = float(delta.sum())
        if travel < thresholds.min_travel:
            drops[episode] = f"zero_travel (path {travel:.4f})"
            continue
        ranges = action.max(axis=0) - action.min(axis=0)
        idle_threshold = np.maximum(ranges * 0.01, 1e-6)
        idle = float((delta < idle_threshold).all(axis=1).mean())
        if idle >= thresholds.idle_max:
            drops[episode] = f"idle ({idle:.0%})"

    keep = sorted(set(lengths) - set(drops))
    return EpisodePlan(keep=keep, drops=drops)


def hardlink_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)


def remux_camera(
    dataset_dir: Path,
    out_dir: Path,
    camera: str,
    episodes_meta: pd.DataFrame,
    keep: list[int],
    fps: float,
) -> dict[int, tuple[int, int, float, float]]:
    """Packet-copy kept episode spans into fresh video files.

    Returns per (original) episode: (chunk_index, file_index,
    from_timestamp, to_timestamp) in the OUTPUT layout — one output file
    per source file that still holds kept episodes, renumbered densely.
    Raises when a kept span does not start on a keyframe (re-encoding is
    lossy; such files are a loud dataset failure, not a silent fallback).
    """
    meta = episodes_meta.set_index("episode_index")
    by_source: dict[tuple[int, int], list[int]] = {}
    for episode in keep:
        key = (
            int(meta.loc[episode, f"videos/{camera}/chunk_index"]),
            int(meta.loc[episode, f"videos/{camera}/file_index"]),
        )
        by_source.setdefault(key, []).append(episode)

    placement: dict[int, tuple[int, int, float, float]] = {}
    half_frame = 0.5 / fps
    for out_file_index, (source_key, source_episodes) in enumerate(sorted(by_source.items())):
        chunk_index, file_index = source_key
        source_path = (
            dataset_dir
            / "videos"
            / camera
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.mp4"
        )
        out_path = out_dir / "videos" / camera / "chunk-000" / f"file-{out_file_index:03d}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        spans = sorted(
            (
                float(meta.loc[episode, f"videos/{camera}/from_timestamp"]),
                float(meta.loc[episode, f"videos/{camera}/to_timestamp"]),
                episode,
            )
            for episode in source_episodes
        )
        with av.open(str(source_path)) as source, av.open(str(out_path), "w") as out:
            in_stream = source.streams.video[0]
            time_base = in_stream.time_base or Fraction(1, 90000)
            out_stream = out.add_stream_from_template(in_stream)
            offset = 0.0
            span_pts: dict[
                int, tuple[int, int, int]
            ] = {}  # episode -> (from_pts, to_pts, new_base_pts)
            for from_ts, to_ts, episode in spans:
                placement[episode] = (0, out_file_index, offset, offset + (to_ts - from_ts))
                span_pts[episode] = (
                    round(from_ts / time_base),
                    round(to_ts / time_base),
                    round(offset / time_base),
                )
                offset += to_ts - from_ts

            ordered = sorted(span_pts.items(), key=lambda kv: kv[1][0])
            span_cursor = 0
            first_packet_of_span = True
            for packet in source.demux(in_stream):
                if packet.pts is None:
                    continue
                while span_cursor < len(ordered) and packet.pts >= ordered[span_cursor][1][1]:
                    span_cursor += 1
                    first_packet_of_span = True
                if span_cursor >= len(ordered):
                    break
                from_pts, to_pts, new_base = ordered[span_cursor][1]
                if not from_pts <= packet.pts < to_pts:
                    continue
                if first_packet_of_span:
                    if not packet.is_keyframe or packet.pts > from_pts + round(
                        half_frame / time_base
                    ):
                        raise ValueError(
                            f"{source_path.name}: episode {ordered[span_cursor][0]} span does "
                            f"not start on a keyframe (pts {packet.pts}, expected {from_pts}) — "
                            "remux impossible without lossy re-encode"
                        )
                    first_packet_of_span = False
                shift = new_base - from_pts
                packet.pts += shift
                if packet.dts is not None:
                    packet.dts += shift
                packet.stream = out_stream
                out.mux(packet)
    return placement


def unflatten_episode_stats(row: pd.Series) -> dict[str, dict[str, np.ndarray]]:
    stats: dict[str, dict[str, np.ndarray]] = {}
    for column_name in row.index:
        column = str(column_name)
        if not column.startswith("stats/"):
            continue
        _, feature, key = column.split("/", 2)
        value = row[column]
        if value is None:
            continue
        stats.setdefault(feature, {})[key] = np.asarray(value)
    return stats


def rebuild_dataset(
    dataset_dir: Path,
    out_dir: Path,
    plan: EpisodePlan,
    fps: float,
) -> None:
    """Write a new dataset containing only kept episodes, renumbered."""
    episodes_meta = pd.concat(
        [pd.read_parquet(p) for p in sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))]
    ).sort_values("episode_index")
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    cameras = sorted(
        key.removeprefix("observation.images.")
        for key, feature in info["features"].items()
        if feature.get("dtype") == "video"
    )
    new_index = {old: new for new, old in enumerate(plan.keep)}

    # --- videos (remux) ---
    placements = {
        camera: remux_camera(dataset_dir, out_dir, camera, episodes_meta, plan.keep, fps)
        for camera in [f"observation.images.{c}" for c in cameras]
    }

    # --- data parquet (drop rows, renumber, one row group per episode) ---
    data_frames = [pd.read_parquet(p) for p in sorted((dataset_dir / "data").rglob("*.parquet"))]
    data = pd.concat(data_frames)
    data = data[data["episode_index"].isin(plan.keep)].copy()
    data["episode_index"] = [new_index[int(e)] for e in np.asarray(data["episode_index"]).tolist()]
    # type ignores: the bundled pandas stubs mis-infer boolean-mask getitem
    # results, cascading bogus overload errors into sort_values.
    data = data.sort_values(["episode_index", "frame_index"])  # type: ignore[reportCallIssue]
    data["index"] = np.arange(len(data), dtype=np.int64)
    data_path = out_dir / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(data_path, engine="pyarrow", compression="snappy", index=False)

    # --- meta/episodes ---
    kept_meta = episodes_meta[episodes_meta["episode_index"].isin(plan.keep)].copy()
    kept_meta = kept_meta.sort_values("episode_index")  # type: ignore[reportCallIssue]  # same stub mis-inference
    lengths = np.asarray(kept_meta["length"], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    original_order = list(kept_meta["episode_index"])
    kept_meta["episode_index"] = [new_index[e] for e in original_order]
    kept_meta["dataset_from_index"] = starts
    kept_meta["dataset_to_index"] = starts + lengths
    kept_meta["data/chunk_index"] = 0
    kept_meta["data/file_index"] = 0
    kept_meta["meta/episodes/chunk_index"] = 0
    kept_meta["meta/episodes/file_index"] = 0
    for camera in [f"observation.images.{c}" for c in cameras]:
        placement = placements[camera]
        kept_meta[f"videos/{camera}/chunk_index"] = [placement[e][0] for e in original_order]
        kept_meta[f"videos/{camera}/file_index"] = [placement[e][1] for e in original_order]
        kept_meta[f"videos/{camera}/from_timestamp"] = [placement[e][2] for e in original_order]
        kept_meta[f"videos/{camera}/to_timestamp"] = [placement[e][3] for e in original_order]
    meta_path = out_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    kept_meta.to_parquet(meta_path, engine="pyarrow", compression="snappy", index=False)

    # --- tasks (kept whole: task_index references stay valid) ---
    shutil.copy2(dataset_dir / "meta" / "tasks.parquet", out_dir / "meta" / "tasks.parquet")

    # --- stats: re-aggregate + exact quantiles over kept frames ---
    per_episode = [unflatten_episode_stats(row) for _, row in kept_meta.iterrows()]
    aggregated = aggregate_stats(per_episode)
    stats = {
        feature: {key: np.asarray(value).tolist() for key, value in entry.items()}
        for feature, entry in aggregated.items()
    }
    for feature, quantiles in feature_quantiles(out_dir).items():
        stats.setdefault(feature, {}).update(quantiles)
    (out_dir / "meta" / "stats.json").write_text(json.dumps(stats, indent=1))

    # --- info.json ---
    info["total_episodes"] = len(plan.keep)
    info["total_frames"] = int(lengths.sum())
    info["splits"] = {"train": f"0:{len(plan.keep)}"}
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=1))


def validate_output(out_dir: Path, expected_episodes: int, fps: float) -> None:
    """Reload with lerobot and decode boundary frames of the first and
    last episode from every camera — catches remux timestamp bugs."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = f"{out_dir.parent.name}/{out_dir.name}"
    dataset = LeRobotDataset(repo_id, root=str(out_dir), tolerance_s=0.5 / fps)
    if dataset.num_episodes != expected_episodes:
        raise ValueError(
            f"reload has {dataset.num_episodes} episodes, expected {expected_episodes}"
        )
    lengths = episode_lengths(out_dir)["length"].to_numpy()
    if len(dataset.hf_dataset) != int(lengths.sum()):
        raise ValueError(
            f"data rows {len(dataset.hf_dataset)} != sum of lengths {int(lengths.sum())}"
        )
    for episode in (0, expected_episodes - 1):
        row = dataset.meta.episodes[episode]
        for index in (int(row["dataset_from_index"]), int(row["dataset_to_index"]) - 1):
            item = dataset[index]  # decodes every camera at this frame
            for key in dataset.meta.camera_keys:
                if item[key].shape[0] != 3:
                    raise ValueError(f"bad frame for {key} at index {index}")


def process_dataset(
    dataset_dir: Path,
    output: Path,
    thresholds: Thresholds,
    *,
    dry_run: bool,
) -> dict:
    repo_id = f"{dataset_dir.parent.name}/{dataset_dir.name}"
    started = time.perf_counter()
    result: dict = {"dataset": repo_id, "time": time.strftime("%F %T", time.gmtime())}
    try:
        info = json.loads((dataset_dir / "meta" / "info.json").read_text())
        fps = float(info["fps"])
        action_dim = int(info["features"]["action"]["shape"][0])
        state_dim = int(info["features"]["observation.state"]["shape"][0])
        if (action_dim, state_dim) != (EXPECTED_DIMS, EXPECTED_DIMS) or fps != EXPECTED_FPS:
            raise ValueError(
                f"out of scope: dims {action_dim}/{state_dim}, fps {fps:g} — the download "
                "was supposed to be scoped; investigate before continuing"
            )
        action = json.loads((dataset_dir / "meta" / "stats.json").read_text()).get("action", {})
        if "q01" not in action or "q99" not in action:
            raise ValueError("stats.json lacks exact q01/q99 — backfill before filtering")

        plan = plan_episodes(dataset_dir, thresholds, fps)
        result["episodes"] = len(plan.keep) + len(plan.drops)
        result["kept"] = len(plan.keep)
        result["drops"] = {str(e): reason for e, reason in sorted(plan.drops.items())}

        if len(plan.keep) < thresholds.min_episodes:
            result["status"] = "dataset_dropped"
            result["reason"] = (
                f"{len(plan.keep)} surviving episode(s) < --min-episodes {thresholds.min_episodes}"
            )
            return result
        if dry_run:
            result["status"] = "dry_run"
            return result

        out_dir = output / repo_id
        staging = output / ".staging" / repo_id.replace("/", "_")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        (staging / "meta").mkdir()

        if not plan.drops:
            hardlink_tree(dataset_dir, staging)
            result["mode"] = "hardlink"
        else:
            rebuild_dataset(dataset_dir, staging, plan, fps)
            result["mode"] = "rebuild"
        validate_output(staging, len(plan.keep), fps)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(out_dir)
        result["status"] = "ok"
    except Exception as error:  # noqa: BLE001 - quarantine and continue the sweep
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    result["seconds"] = round(time.perf_counter() - started, 2)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mechanical episode filtering + merge into one curated collection.",
    )
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets", type=str, nargs="*", default=None, help="Subset as <user>/<dataset>."
    )
    parser.add_argument("--min-frames", type=int, default=50)
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument(
        "--max-frames-quantile",
        type=float,
        default=0.995,
        help="Episodes longer than this quantile of the input scope are dropped.",
    )
    parser.add_argument("--min-travel", type=float, default=1e-3)
    parser.add_argument("--idle-max", type=float, default=0.8)
    parser.add_argument("--min-episodes", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = discover(args.sources)
    # Length-outlier threshold over the whole INPUT scope, computed BEFORE
    # any --datasets subsetting: the marathon cut is a property of the
    # corpus, not of whichever subset this invocation processes (a 6-dataset
    # test run once derived max_frames=656 and would have shredded normal
    # episodes).
    all_lengths = np.concatenate([episode_lengths(d)["length"].to_numpy() for d in datasets])
    max_frames = int(np.quantile(all_lengths, args.max_frames_quantile))
    if args.datasets is not None:
        by_id = {f"{d.parent.name}/{d.name}": d for d in datasets}
        unknown = set(args.datasets) - set(by_id)
        if unknown:
            raise SystemExit(f"unknown datasets: {sorted(unknown)}")
        datasets = [by_id[name] for name in sorted(set(args.datasets))]
    thresholds = Thresholds(
        min_frames=args.min_frames,
        min_seconds=args.min_seconds,
        max_frames=max_frames,
        min_travel=args.min_travel,
        idle_max=args.idle_max,
        min_episodes=args.min_episodes,
    )
    print(f"{len(datasets)} dataset(s), {len(all_lengths):,} episodes | thresholds: {thresholds}")

    todo: list[Path] = []
    skipped_existing = 0
    for dataset_dir in datasets:
        repo_id = f"{dataset_dir.parent.name}/{dataset_dir.name}"
        if not args.force and not args.dry_run and (args.output / repo_id).exists():
            skipped_existing += 1
            continue
        todo.append(dataset_dir)
    if skipped_existing:
        print(f"skipping {skipped_existing} dataset(s) already in {args.output}")

    manifest = args.output / "filter_manifest.jsonl"
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
    outcomes: dict[str, int] = {}
    episodes_kept = episodes_dropped = 0

    def record(result: dict) -> None:
        nonlocal episodes_kept, episodes_dropped
        outcomes[result["status"]] = outcomes.get(result["status"], 0) + 1
        episodes_kept += result.get("kept", 0)
        episodes_dropped += len(result.get("drops", {}))
        if result["status"] in ("failed", "dataset_dropped"):
            print(
                f"{result['status'].upper()} {result['dataset']}: {result.get('reason') or result.get('error')}"
            )
        if args.dry_run:
            for episode, reason in result.get("drops", {}).items():
                print(f"  drop {result['dataset']} ep {episode}: {reason}")
        if not args.dry_run:
            with manifest.open("a") as f:
                f.write(json.dumps(result) + "\n")

    if args.workers <= 1:
        for dataset_dir in todo:
            record(process_dataset(dataset_dir, args.output, thresholds, dry_run=args.dry_run))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(process_dataset, d, args.output, thresholds, dry_run=args.dry_run)
                for d in todo
            ]
            for future in as_completed(futures):
                record(future.result())

    staging_root = args.output / ".staging"
    if staging_root.exists() and not any(staging_root.iterdir()):
        staging_root.rmdir()
    print(
        f"done: {outcomes} | episodes kept {episodes_kept:,} / dropped {episodes_dropped:,}"
        + ("" if args.dry_run else f" | manifest: {manifest}")
    )
    return 1 if outcomes.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
