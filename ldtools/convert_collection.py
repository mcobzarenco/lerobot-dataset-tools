"""Migrate a collection of LeRobot sub-datasets to dataset format v3.0.

Points at a collection root laid out as ``<root>/<user>/<dataset>`` (e.g. the
HuggingFaceVLA community_dataset_v1 download), prints a census of the
LeRobot format versions found, then converts every selected sub-dataset:

    v2.0 --[synthesize per-episode stats]--> v2.1 --[official converter]--> v3.0

Results land in ``<output>/<user>/<dataset>``. The source is never modified.

Usage:
    uv run python -m fmatch.convert_collection \
        --source /home/marius/w/community_dataset_v1 \
        --output /home/marius/w/community_dataset_v1_v3

    # census only / a subset / redo
    ... --stats-only
    ... --datasets ZGGZZG/so100_drop0 ad330/cubePlace
    ... --force

Properties:
  - idempotent: sub-datasets whose output already exists as valid v3.0 are
    skipped; interrupted work is staged in ``<output>/.staging`` and redone.
  - repairs the known collection quirk where stats keys dropped the
    ``images.`` segment (``observation.image`` vs feature
    ``observation.images.image``) before converting.
  - copies only real dataset files (skips the duplicate flat video trees and
    ``*.bak`` files present in the community collections).
  - appends one JSON line per processed dataset to
    ``<output>/conversion_manifest.jsonl``.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

SUPPORTED_SOURCE_VERSIONS = {"v2.0", "v2.1"}


def log(name: str, message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{name}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Discovery / census
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubDataset:
    name: str  # "<user>/<dataset>"
    path: Path
    version: str
    episodes: int
    frames: int
    size_bytes: int


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def discover(source: Path) -> list[SubDataset]:
    found = []
    for info_path in sorted(source.glob("*/*/meta/info.json")):
        ds_dir = info_path.parent.parent
        info = json.loads(info_path.read_text())
        found.append(
            SubDataset(
                name=f"{ds_dir.parent.name}/{ds_dir.name}",
                path=ds_dir,
                version=str(info.get("codebase_version", "?")),
                episodes=int(info.get("total_episodes", 0)),
                frames=int(info.get("total_frames", 0)),
                size_bytes=dir_size(ds_dir),
            )
        )
    return found


def print_census(datasets: list[SubDataset], output: Path) -> None:
    by_version = Counter(d.version for d in datasets)
    print(f"\n{len(datasets)} sub-datasets found")
    print(f"{'version':<10}{'datasets':>10}{'episodes':>12}{'frames':>14}{'size':>10}")
    for version in sorted(by_version):
        group = [d for d in datasets if d.version == version]
        marker = "" if version in SUPPORTED_SOURCE_VERSIONS else "  (unsupported!)"
        print(
            f"{version:<10}{len(group):>10}{sum(d.episodes for d in group):>12}"
            f"{sum(d.frames for d in group):>14}{sum(d.size_bytes for d in group) / 1e9:>9.1f}G"
            f"{marker}"
        )
    done = sum(1 for d in datasets if is_converted(output, d.name))
    print(f"already converted in {output}: {done}/{len(datasets)}\n")


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def is_converted(output: Path, name: str) -> bool:
    info_path = output / name / "meta" / "info.json"
    if not info_path.is_file():
        return False
    try:
        return json.loads(info_path.read_text()).get("codebase_version") == "v3.0"
    except json.JSONDecodeError:
        return False


# ---------------------------------------------------------------------------
# Staging copy (skips junk: duplicate flat video dirs, *.bak files)
# ---------------------------------------------------------------------------


def stage_copy(src: Path, dst: Path, info: dict) -> None:
    video_keys = {k for k, f in info["features"].items() if f.get("dtype") == "video"}
    dst.mkdir(parents=True, exist_ok=True)

    meta_dst = dst / "meta"
    shutil.copytree(
        src / "meta",
        meta_dst,
        ignore=shutil.ignore_patterns("*.bak"),
        dirs_exist_ok=True,
    )
    shutil.copytree(src / "data", dst / "data", dirs_exist_ok=True)

    # v2.x layout: videos/chunk-XXX/<video_key>/episode_XXXXXX.mp4.
    # Copy only directories matching real video features; the community
    # collections also contain stray duplicates under non-feature names.
    for chunk_dir in sorted((src / "videos").glob("chunk-*")):
        for key_dir in sorted(chunk_dir.iterdir()):
            if key_dir.name in video_keys:
                shutil.copytree(
                    key_dir,
                    dst / "videos" / chunk_dir.name / key_dir.name,
                    dirs_exist_ok=True,
                )


# ---------------------------------------------------------------------------
# Metadata reconciliation: episodes.jsonl is the source of truth
# ---------------------------------------------------------------------------


def reconcile_with_episodes_metadata(root: Path, info: dict, name: str) -> dict:
    """Make files and declared totals agree with ``meta/episodes.jsonl``.

    Two failure patterns exist in the community collections (the cleaning
    pipeline filtered episodes without removing files or fixing totals):

    1. orphaned per-episode parquet/mp4 files for episode indices that are
       absent from ``episodes.jsonl`` -> the official converter refuses
       ("Number of episodes is not the same"). We delete the orphans.
    2. ``info.json`` totals that disagree with the actual data
       (``total_frames`` off by a few) -> the converted dataset reloads as
       "insufficient cache" and lerobot falls back to the hub with a
       misleading error. We recompute totals from ``episodes.jsonl``.

    Returns a structured summary of corrections (for the manifest / dataset
    card); empty dict when nothing needed fixing.
    """
    corrections: dict = {}
    episodes = [
        json.loads(line)
        for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]

    # Episodes whose parquet row count disagrees with their declared length:
    # if the videos agree with the parquet (only the metadata is stale, e.g.
    # trimmed datasets), repair the declared length; otherwise drop the
    # episode as corrupt. Loud either way.
    consistent = []
    repaired = 0
    dropped_episode_ids: list[int] = []
    for episode in episodes:
        data_path, video_paths = _episode_paths(root, info, int(episode["episode_index"]))
        rows = pq.ParquetFile(data_path).metadata.num_rows if data_path.is_file() else -1
        if rows == int(episode["length"]):
            consistent.append(episode)
            continue
        if rows > 0 and all(_video_frame_count(v) == rows for v in video_paths):
            log(
                name,
                f"repairing declared length of episode {episode['episode_index']}: "
                f"{episode['length']} -> {rows} (parquet and videos agree)",
            )
            episode["length"] = rows
            consistent.append(episode)
            repaired += 1
        else:
            log(
                name,
                f"dropping corrupt episode {episode['episode_index']} "
                f"(parquet rows {rows} != declared length {episode['length']})",
            )
            dropped_episode_ids.append(int(episode["episode_index"]))
    if dropped_episode_ids:
        log(name, f"dropped {len(dropped_episode_ids)} corrupt episode(s)")
        corrections["dropped_corrupt_episodes"] = dropped_episode_ids
    if repaired:
        corrections["repaired_episode_lengths"] = repaired
    episodes = consistent
    if repaired:
        (root / "meta" / "episodes.jsonl").write_text(
            "\n".join(json.dumps(e) for e in episodes) + "\n"
        )
    if not episodes:
        raise RuntimeError("no internally-consistent episodes left after reconciliation")
    valid = {int(e["episode_index"]) for e in episodes}

    orphans = 0
    for path in sorted(root.glob("data/chunk-*/episode_*.parquet")) + sorted(
        root.glob("videos/chunk-*/*/episode_*.mp4")
    ):
        if int(path.stem.rsplit("_", 1)[1]) not in valid:
            path.unlink()
            orphans += 1
    if orphans:
        log(name, f"removed {orphans} orphaned episode files not in episodes.jsonl")

    # Densify: the official v21->v30 converter assumes contiguous episode
    # indices, and the v3 reader assumes the global frame `index` column is
    # the row counter. Community datasets whose episodes were filtered (or
    # whose indices were renumbered inconsistently) violate both.
    episodes.sort(key=lambda e: int(e["episode_index"]))
    mapping = {int(e["episode_index"]): new for new, e in enumerate(episodes)}
    if any(old != new for old, new in mapping.items()) or frame_index_broken(root, info, episodes):
        log(name, f"densifying {len(episodes)} episodes (sparse or broken indices)")
        densify_indices(root, info, episodes, mapping)

    # Recompute declared totals from the episode metadata.
    true_episodes = len(valid)
    true_frames = sum(int(e["length"]) for e in episodes)
    video_keys = [k for k, f in info["features"].items() if f.get("dtype") == "video"]
    updates: dict[str, object] = {}
    # Aggregates are never trusted from source info.json: recompute all of
    # them from the reconciled episode table (several source datasets ship
    # inflated total_frames, stale splits or total_videos=0).
    if info.get("total_episodes") != true_episodes:
        updates["total_episodes"] = true_episodes
    if info.get("total_frames") != true_frames:
        updates["total_frames"] = true_frames
    expected_videos = true_episodes * len(video_keys)
    if "total_videos" in info and info["total_videos"] != expected_videos:
        updates["total_videos"] = expected_videos
    expected_splits = {"train": f"0:{true_episodes}"}
    if info.get("splits") != expected_splits:
        updates["splits"] = expected_splits
    if updates:
        log(name, f"reconciled info.json aggregates from episodes.jsonl: {updates}")
        corrections["aggregates"] = {
            key: {"from": info.get(key), "to": value} for key, value in updates.items()
        }
        info.update(updates)
        (root / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    # episodes_stats.jsonl may carry entries for removed episodes, and must
    # follow any renumbering.
    stats_path = root / "meta" / "episodes_stats.jsonl"
    if stats_path.is_file():
        records = [
            json.loads(line)
            for line in stats_path.read_text().splitlines()
            if line.strip() and int(json.loads(line)["episode_index"]) in valid
        ]
        for record in records:
            record["episode_index"] = mapping[int(record["episode_index"])]
        records.sort(key=lambda r: int(r["episode_index"]))
        stats_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    return corrections


def _video_frame_count(video_path: Path) -> int:
    if not video_path.is_file():
        return -1
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return -1


def _resolve_chunked(root: Path, template_path: str, **fmt: object) -> Path:
    """Resolve a v2.x chunked path, tolerating wrong chunk placement.

    Some datasets violate their own ``chunks_size`` (e.g. 1504 episodes all
    in ``chunk-000``); if the computed chunk doesn't contain the file, search
    the other chunk dirs for it.
    """
    computed = root / template_path.format(**fmt)
    if computed.is_file():
        return computed
    wildcard = re.sub(r"\{episode_chunk[^}]*\}", "*", template_path)
    matches = sorted(
        root.glob(wildcard.format(**{k: v for k, v in fmt.items() if k != "episode_chunk"}))
    )
    return matches[0] if matches else computed


def _episode_paths(root: Path, info: dict, episode_index: int) -> tuple[Path, list[Path]]:
    chunks_size = int(info.get("chunks_size", 1000))
    chunk = episode_index // chunks_size
    data = _resolve_chunked(
        root, info["data_path"], episode_chunk=chunk, episode_index=episode_index
    )
    videos = [
        _resolve_chunked(
            root,
            info["video_path"],
            episode_chunk=chunk,
            video_key=key,
            episode_index=episode_index,
        )
        for key, f in info["features"].items()
        if f.get("dtype") == "video"
    ]
    return data, videos


def frame_index_broken(root: Path, info: dict, episodes: list[dict]) -> bool:
    """True if the global `index` column doesn't equal the row counter.

    Uses parquet footer statistics (no data read) where available.
    """
    start = 0
    for episode in episodes:
        data_path, _ = _episode_paths(root, info, int(episode["episode_index"]))
        metadata = pq.ParquetFile(data_path).metadata
        column = metadata.schema.to_arrow_schema().get_field_index("index")
        lo = hi = None
        for group_index in range(metadata.num_row_groups):
            stats = metadata.row_group(group_index).column(column).statistics
            if stats is None or not stats.has_min_max:
                lo, hi = None, None
                break
            lo = stats.min if lo is None else min(lo, stats.min)
            hi = stats.max if hi is None else max(hi, stats.max)
        if lo is None or hi is None:  # no stats: read the column
            values = pq.read_table(data_path, columns=["index"])["index"].to_numpy()
            lo, hi = int(values.min()), int(values.max())
        length = int(episode["length"])
        if lo != start or hi != start + length - 1:
            return True
        start += length
    return False


def densify_indices(root: Path, info: dict, episodes: list[dict], mapping: dict[int, int]) -> None:
    """Renumber episodes 0..K-1 and rebuild the global frame index.

    Processes episodes in ascending original order; since new <= old always
    holds when closing gaps, renames never clobber unprocessed files.
    """
    start = 0
    for episode in episodes:
        old = int(episode["episode_index"])
        new = mapping[old]
        old_data, old_videos = _episode_paths(root, info, old)
        new_data, new_videos = _episode_paths(root, info, new)

        table = pq.read_table(old_data)
        length = table.num_rows
        if length != int(episode["length"]):
            raise RuntimeError(
                f"episode {old}: parquet has {length} rows but episodes.jsonl "
                f"declares {episode['length']}"
            )
        table = table.set_column(
            table.schema.get_field_index("episode_index"),
            "episode_index",
            pa.array(np.full(length, new, dtype=np.int64)),
        )
        table = table.set_column(
            table.schema.get_field_index("index"),
            "index",
            pa.array(np.arange(start, start + length, dtype=np.int64)),
        )
        new_data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, new_data)
        if new_data != old_data:
            old_data.unlink()

        for old_video, new_video in zip(old_videos, new_videos):
            if old_video != new_video:
                new_video.parent.mkdir(parents=True, exist_ok=True)
                old_video.rename(new_video)

        episode["episode_index"] = new
        start += length

    episodes_path = root / "meta" / "episodes.jsonl"
    episodes_path.write_text("\n".join(json.dumps(e) for e in episodes) + "\n")


# ---------------------------------------------------------------------------
# Feature sanitation: v3.0 cannot represent some legacy/exotic features
# ---------------------------------------------------------------------------

# v3.0 features must be image/video or fixed-shape numeric arrays
# (lerobot feature_utils supports shapes of rank 1..5 plus scalars).
UNSUPPORTED_FEATURE_DTYPES = {"string", "list", "dict"}


def drop_hollow_cameras(root: Path, info: dict, name: str) -> list[str]:
    """Drop video features whose files are mostly missing.

    Some datasets added a camera mid-recording (e.g. 269 videos for 1504
    episodes); the converter requires every camera to cover every episode.
    Cameras covering every episode are kept; partial ones are dropped with
    their files.
    """
    total = int(info.get("total_episodes", 0))
    dropped: list[str] = []
    for key in [k for k, f in info["features"].items() if f.get("dtype") == "video"]:
        count = len(list(root.glob(f"videos/chunk-*/{key}/episode_*.mp4")))
        if 0 < count < total:
            log(name, f"dropping camera {key}: only {count}/{total} episode videos exist")
            for path in root.glob(f"videos/chunk-*/{key}"):
                shutil.rmtree(path)
            del info["features"][key]
            dropped.append(key)
    if dropped:
        if "total_videos" in info:
            video_keys = [k for k, f in info["features"].items() if f.get("dtype") == "video"]
            info["total_videos"] = total * len(video_keys)
        (root / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    return dropped


def shrink_oversized_stats(root: Path, name: str, max_elements: int = 64) -> None:
    """Reduce per-episode stats arrays with more than ``max_elements`` values
    to scalars (exact aggregation over the original array).

    Depth-map features get per-pixel stats in v2.1 (e.g. 480x640 per stat per
    episode); carried into v3 episode metadata they exceed lerobot's 100 MB
    single-file limit. Scalar min/max/mean/std keep normalization sane.
    """
    stats_path = root / "meta" / "episodes_stats.jsonl"
    if not stats_path.is_file():
        return
    shrunk: set[str] = set()
    records = []
    for line in stats_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for key, stats in record.get("stats", {}).items():
            minimum = np.asarray(stats.get("min"))
            if minimum.size <= max_elements:
                continue
            mean = np.asarray(stats["mean"], dtype=np.float64)
            std = np.asarray(stats["std"], dtype=np.float64)
            # Exact total variance: pixels have equal frame counts.
            global_mean = float(mean.mean())
            global_var = float((std**2 + mean**2).mean() - global_mean**2)
            record["stats"][key] = {
                "min": [float(np.asarray(stats["min"]).min())],
                "max": [float(np.asarray(stats["max"]).max())],
                "mean": [global_mean],
                "std": [max(global_var, 0.0) ** 0.5],
                "count": stats.get("count", [1]),
            }
            shrunk.add(key)
        records.append(record)
    if shrunk:
        log(name, f"shrunk oversized per-episode stats to scalars: {sorted(shrunk)}")
        stats_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def drop_unsupported_features(root: Path, info: dict, name: str) -> list[str]:
    """Remove features the v3.0 format cannot represent (e.g. per-frame
    ``string`` subtask annotations, ``list`` fields) from ``info.json`` and
    from the data parquets. The source repos remain the home of this data.
    """
    doomed = [
        key
        for key, feature in info["features"].items()
        if feature.get("dtype") in UNSUPPORTED_FEATURE_DTYPES
    ]
    if not doomed:
        return []
    log(name, f"dropping features unsupported by v3.0: {doomed}")
    for key in doomed:
        del info["features"][key]
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    # the columns themselves are removed by sanitize_data_columns (they are
    # now undeclared)
    return doomed


def normalize_list_wrapped_scalars(root: Path, info: dict, name: str) -> None:
    """Unwrap columns stored as list<T> where the feature declares shape [1].

    Some datasets store scalar features (e.g. ``next.done`` bool, shape [1])
    wrapped in single-element lists; the v3.0 schema cast then fails with
    "Couldn't cast list<element: bool> to bool".
    """
    scalar_keys = {
        key
        for key, feature in info["features"].items()
        if feature.get("dtype") not in ("video", "image")
        and list(feature.get("shape") or []) == [1]
    }
    if not scalar_keys:
        return

    def is_listy(t: pa.DataType) -> bool:
        return pa.types.is_list(t) or pa.types.is_fixed_size_list(t) or pa.types.is_large_list(t)

    fixed_columns: set[str] = set()
    for parquet_path in sorted(root.glob("data/chunk-*/episode_*.parquet")):
        schema = pq.read_schema(parquet_path)
        wrapped = [
            key for key in scalar_keys if key in schema.names and is_listy(schema.field(key).type)
        ]
        if not wrapped:
            continue
        table = pq.read_table(parquet_path)
        for key in wrapped:
            column = table[key].combine_chunks()
            flat = column.flatten()
            if len(flat) != len(table):
                raise RuntimeError(f"{key}: list-wrapped column has non-singleton lists")
            table = table.set_column(table.schema.get_field_index(key), key, flat)
        pq.write_table(table, parquet_path)
        fixed_columns.update(wrapped)
    if fixed_columns:
        log(name, f"unwrapped list-stored scalar columns: {sorted(fixed_columns)}")


# ---------------------------------------------------------------------------
# Data-column sanitation (drop columns not declared in info features)
# ---------------------------------------------------------------------------


def sanitize_data_columns(root: Path, info: dict) -> set[str]:
    """Drop parquet columns that aren't declared features.

    Some community datasets carry legacy columns (e.g. ``next.done``) in the
    per-episode parquets that are absent from ``info.json`` features; the
    datasets library later refuses to cast the consolidated v3 file against
    the declared schema.
    """
    declared = {k for k, f in info["features"].items() if f.get("dtype") != "video"}
    dropped: set[str] = set()
    for parquet_path in sorted(root.glob("data/chunk-*/episode_*.parquet")):
        names = pq.read_schema(parquet_path).names  # cheap: footer only
        extras = [c for c in names if c not in declared]
        if extras:
            dropped.update(extras)
            table = pq.read_table(parquet_path)
            table = table.select([c for c in names if c in declared])
            pq.write_table(table, parquet_path)
    return dropped


# ---------------------------------------------------------------------------
# Video concat: ffmpeg fallback for streams PyAV refuses to mux
# ---------------------------------------------------------------------------

_original_concatenate = None


_original_concat_data = None


def concat_data_files_with_arrow_fallback(
    paths_to_cat: list, new_root: Path, chunk_idx: int, file_idx: int, image_keys: list
) -> None:
    """lerobot's pandas-based data concat, with a pyarrow-native fallback.

    Datasets storing depth maps (or other large arrays) as parquet columns
    break the pandas round-trip (`Table.from_pandas` cannot infer the
    extension dtype). Arrow-native concatenation preserves the source schema
    exactly and needs no inference.
    """
    assert _original_concat_data is not None
    try:
        _original_concat_data(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)
        return
    except Exception as error:  # noqa: BLE001 - deliberate fallback
        log("data-concat", f"pandas failed ({type(error).__name__}); retrying with pyarrow")
    import lerobot.scripts.convert_dataset_v21_to_v30 as converter_module

    tables = [pq.read_table(p) for p in paths_to_cat]
    table = pa.concat_tables(tables, promote_options="default")
    path = Path(new_root) / converter_module.DEFAULT_DATA_PATH.format(
        chunk_index=chunk_idx, file_index=file_idx
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def concatenate_with_ffmpeg_fallback(
    input_video_paths: list,
    output_video_path: Path | str,
    overwrite: bool = True,
    compatibility_check: bool = False,
) -> None:
    """lerobot's PyAV concat, falling back to ffmpeg's concat demuxer.

    Some episodes contain duplicate/non-monotonic DTS at file boundaries;
    PyAV's mux raises EINVAL on these while ffmpeg repairs them in stream
    copy mode.
    """
    assert _original_concatenate is not None
    try:
        _original_concatenate(
            input_video_paths, Path(output_video_path), overwrite, compatibility_check
        )
        return
    except Exception as error:  # noqa: BLE001 - deliberate fallback
        log("video-concat", f"pyav failed ({error}); retrying with ffmpeg")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        for path in input_video_paths:
            escaped = str(Path(path).resolve()).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
        list_path = Path(handle.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(output_video_path),
            ],
            check=True,
        )
    finally:
        list_path.unlink(missing_ok=True)


def install_concat_fallback() -> None:
    global _original_concatenate, _original_concat_data
    import lerobot.scripts.convert_dataset_v21_to_v30 as converter_module

    if converter_module.concatenate_video_files is not concatenate_with_ffmpeg_fallback:
        _original_concatenate = converter_module.concatenate_video_files
        converter_module.concatenate_video_files = concatenate_with_ffmpeg_fallback
    if converter_module.concat_data_files is not concat_data_files_with_arrow_fallback:
        _original_concat_data = converter_module.concat_data_files
        converter_module.concat_data_files = concat_data_files_with_arrow_fallback


# ---------------------------------------------------------------------------
# Stats-key repair (observation.X -> observation.images.X)
# ---------------------------------------------------------------------------


def build_stats_key_repairs(features: dict) -> dict[str, str]:
    repairs = {}
    for key in features:
        if key.startswith("observation.images."):
            flat = key.replace("observation.images.", "observation.", 1)
            if flat not in features:
                repairs[flat] = key
    return repairs


def build_ghost_camera_repairs(stats_keys: list[str], features: dict) -> dict[str, str]:
    """Positionally map stats keys for renamed cameras onto declared ones.

    The collection standardizer renamed camera features (e.g. ``arm`` ->
    ``image``, ``context`` -> ``image2``) without rewriting per-episode
    stats. When the sets are disjoint and equally sized, map by order.
    """
    declared = [k for k, f in features.items() if f.get("dtype") in ("video", "image")]
    ghost = [
        k for k in stats_keys if k.startswith("observation.") and k not in features and "image" in k
    ]
    ghost = [k for k in ghost if k not in declared]
    unmatched_declared = [k for k in declared if k not in stats_keys]
    if ghost and len(ghost) == len(unmatched_declared):
        return dict(zip(sorted(ghost), sorted(unmatched_declared)))
    return {}


def repair_stats_keys(root: Path, features: dict) -> int:
    """Re-key camera stats entries that lost the ``images.`` segment or were
    left under pre-standardization camera names (ghost cameras)."""
    repairs = build_stats_key_repairs(features)
    fixed = 0

    stats_path = root / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        if any(old in stats for old in repairs):
            stats = {repairs.get(k, k): v for k, v in stats.items()}
            stats_path.write_text(json.dumps(stats, indent=4))
            fixed += 1

    ep_stats_path = root / "meta" / "episodes_stats.jsonl"
    if ep_stats_path.is_file():
        lines = ep_stats_path.read_text().splitlines()
        rewritten, changed = [], False
        for line in lines:
            record = json.loads(line)
            stats = record.get("stats", {})
            mapping = dict(repairs)
            mapping.update(build_ghost_camera_repairs(list(stats.keys()), features))
            if any(old in stats for old in mapping):
                record["stats"] = {mapping.get(k, k): v for k, v in stats.items()}
                changed = True
            rewritten.append(json.dumps(record))
        if changed:
            ep_stats_path.write_text("\n".join(rewritten) + "\n")
            fixed += 1
    return fixed


def prune_undeclared_stats(root: Path, features: dict, name: str) -> None:
    """Drop per-episode stats entries for keys that are no longer features
    (e.g. cameras removed by the hollow-camera step). Mixed stats columns
    crash the v3 episodes-metadata writer with a KeyError."""
    ep_stats_path = root / "meta" / "episodes_stats.jsonl"
    if not ep_stats_path.is_file():
        return
    pruned: set[str] = set()
    rewritten = []
    for line in ep_stats_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        stats = record.get("stats", {})
        doomed = [k for k in stats if k not in features]
        for key in doomed:
            del stats[key]
            pruned.add(key)
        rewritten.append(json.dumps(record))
    if pruned:
        log(name, f"pruned stats for undeclared keys: {sorted(pruned)}")
        ep_stats_path.write_text("\n".join(rewritten) + "\n")


# ---------------------------------------------------------------------------
# v2.0 -> v2.1: synthesize meta/episodes_stats.jsonl
# ---------------------------------------------------------------------------


def _normalize_stats_types(record: dict) -> None:
    """Force float leaves for min/max/mean/std and int for count.

    Mixed int/float stats across episodes (original entries vs synthesized
    gap-fills) make the arrow writer refuse the episodes-metadata table.
    """
    for stats in record.get("stats", {}).values():
        for stat_name, value in stats.items():
            if stat_name == "count":
                stats[stat_name] = np.asarray(value, dtype=np.int64).tolist()
            else:
                stats[stat_name] = np.asarray(value, dtype=np.float64).tolist()


def episode_video_stats(video_path: Path) -> dict[str, np.ndarray]:
    """Per-channel stats over sampled frames of one episode video, in [0,1].

    Note: this is the CPU hot spot of the whole pipeline — frame-accurate
    sampling of AV1 video decodes from the nearest keyframe for every sampled
    index. Parallelism is applied at the dataset level (--workers) instead of
    here to avoid oversubscription.
    """
    from lerobot.datasets.compute_stats import get_feature_stats, sample_indices
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(video_path))
    num_frames = decoder.metadata.num_frames
    if not num_frames:
        raise RuntimeError(f"no frames in {video_path}")
    indices = sample_indices(int(num_frames))
    frames = decoder.get_frames_at(indices).data.numpy().astype(np.float32)  # (N,C,H,W)
    stats = get_feature_stats(frames, axis=(0, 2, 3), keepdims=True, quantile_list=[])
    return {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in stats.items()}


def synthesize_episodes_stats(root: Path, info: dict, name: str) -> None:
    """Ensure every episode in episodes.jsonl has an episodes_stats entry.

    Covers two cases: v2.0 datasets (no episodes_stats.jsonl at all) and
    datasets with *partial* coverage (e.g. stats missing for a tail of
    episodes; the official converter indexes stats positionally and crashes).
    Existing entries are kept; only gaps are computed."""
    from lerobot.datasets.compute_stats import get_feature_stats

    features = info["features"]
    episodes = [
        json.loads(line)
        for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]

    stats_path = root / "meta" / "episodes_stats.jsonl"
    existing: dict[int, dict] = {}
    if stats_path.is_file():
        for line in stats_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                existing[int(record["episode_index"])] = record

    missing = [e for e in episodes if int(e["episode_index"]) not in existing]
    if not missing:
        return
    log(name, f"synthesizing episode stats for {len(missing)}/{len(episodes)} episodes")

    for i, episode in enumerate(missing):
        if i % 10 == 0 and len(missing) > 10:
            log(name, f"  stats progress: {i}/{len(missing)}")
        ep_idx = int(episode["episode_index"])
        data_path, video_paths = _episode_paths(root, info, ep_idx)
        df = pd.read_parquet(data_path)

        ep_stats: dict[str, dict] = {}
        video_iter = iter(video_paths)
        for key, feature in features.items():
            dtype = feature.get("dtype")
            if dtype in ("string", "language"):
                continue
            if dtype in ("video", "image"):
                ep_stats[key] = episode_video_stats(next(video_iter))
            else:
                column = df[key].to_numpy()
                array = np.stack(list(column)) if column.dtype == object else column
                ep_stats[key] = get_feature_stats(
                    array, axis=0, keepdims=array.ndim == 1, quantile_list=[]
                )

        serialized = {
            key: {k: np.asarray(v).tolist() for k, v in stats.items()}
            for key, stats in ep_stats.items()
        }
        existing[ep_idx] = {"episode_index": ep_idx, "stats": serialized}

    ordered = [existing[int(e["episode_index"])] for e in episodes]
    for record in ordered:
        _normalize_stats_types(record)
    stats_path.write_text("\n".join(json.dumps(r) for r in ordered) + "\n")
    if info.get("codebase_version") == "v2.0":
        info["codebase_version"] = "v2.1"
        (root / "meta" / "info.json").write_text(json.dumps(info, indent=4))


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------


def validate_v3(root: Path, name: str) -> dict:
    """Load the converted dataset and cross-check its bookkeeping.

    The frame/episode invariants are checked against *independent* sources
    (physical parquet row counts and the episode table), not just the values
    lerobot itself reads from info.json — `dataset.num_frames == total_frames`
    alone would be circular, and inflated source counters have shipped real
    IndexErrors to training runs.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = json.loads((root / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise RuntimeError(f"expected v3.0 after conversion, got {info.get('codebase_version')}")

    physical_rows = sum(
        pq.read_metadata(f).num_rows for f in sorted(root.glob("data/chunk-*/file-*.parquet"))
    )
    episode_table = pd.concat(
        pd.read_parquet(p) for p in sorted(root.glob("meta/episodes/chunk-*/file-*.parquet"))
    )
    episode_length_sum = int(episode_table["length"].sum())

    dataset = LeRobotDataset(name, root=root)
    checks = {
        "info.total_episodes": info["total_episodes"],
        "episode table rows": len(episode_table),
        "dataset.num_episodes": dataset.num_episodes,
    }
    if len(set(checks.values())) != 1:
        raise RuntimeError(f"episode count mismatch: {checks}")
    checks = {
        "info.total_frames": info["total_frames"],
        "sum(episode lengths)": episode_length_sum,
        "physical parquet rows": physical_rows,
        "dataset.num_frames": dataset.num_frames,
    }
    if len(set(checks.values())) != 1:
        raise RuntimeError(f"frame count mismatch: {checks}")
    return {"episodes": dataset.num_episodes, "frames": dataset.num_frames}


def convert_one(ds: SubDataset, output: Path) -> dict:
    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

    started = time.time()
    staging_parent = output / ".staging" / ds.name.split("/")[0]
    staging = staging_parent / ds.name.split("/")[1]
    if staging_parent.parent.exists() and staging.exists():
        shutil.rmtree(staging)  # stale partial work from an interrupted run
    old_leftover = staging.parent / f"{staging.name}_old"
    if old_leftover.exists():
        shutil.rmtree(old_leftover)

    info = json.loads((ds.path / "meta" / "info.json").read_text())

    video_keys = [k for k, f in info["features"].items() if f.get("dtype") == "video"]
    if video_keys and not (ds.path / "videos").is_dir():
        raise RuntimeError(f"declares video features {video_keys} but has no videos/ directory")

    log(ds.name, f"staging copy ({ds.size_bytes / 1e9:.1f} GB, {ds.version})")
    stage_copy(ds.path, staging, info)
    if repair_stats_keys(staging, info["features"]):
        log(ds.name, "repaired renamed/flat camera stats keys")
    dropped_features = drop_hollow_cameras(staging, info, ds.name)
    dropped_features += drop_unsupported_features(staging, info, ds.name)
    normalize_list_wrapped_scalars(staging, info, ds.name)
    shrink_oversized_stats(staging, ds.name)
    corrections = reconcile_with_episodes_metadata(staging, info, ds.name)
    dropped = sanitize_data_columns(staging, info)
    if dropped:
        log(ds.name, f"dropped undeclared parquet columns: {sorted(dropped)}")
    prune_undeclared_stats(staging, info["features"], ds.name)
    synthesize_episodes_stats(staging, info, ds.name)  # gap-fills; no-op when complete

    log(ds.name, "converting v2.1 -> v3.0")
    install_concat_fallback()
    convert_dataset(repo_id=ds.name, root=staging, push_to_hub=False)

    if old_leftover.exists():
        shutil.rmtree(old_leftover)

    log(ds.name, "validating")
    counts = validate_v3(staging, ds.name)

    final = output / ds.name
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        shutil.rmtree(final)
    shutil.move(str(staging), str(final))
    log(ds.name, f"done in {time.time() - started:.0f}s -> {final}")

    result = {
        "status": "converted",
        "source_version": ds.version,
        "seconds": round(time.time() - started, 1),
        **counts,
    }
    if dropped_features:
        result["dropped_features"] = dropped_features
    if corrections:
        result["corrections"] = corrections
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Collection root (<root>/<user>/<dataset>).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where converted v3.0 datasets are written.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Subset to process, as '<user>/<dataset>' names (default: all).",
    )
    parser.add_argument("--stats-only", action="store_true", help="Print the census and exit.")
    parser.add_argument("--force", action="store_true", help="Reconvert even if output exists.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Convert this many sub-datasets in parallel (processes). The "
        "CPU-heavy phase is v2.0 stats synthesis (AV1 frame sampling); "
        "4-8 workers is a good laptop setting.",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Post-conversion validation materializes datasets through the HF
    # `datasets` cache; keep that on the output volume (boot disks are small
    # and depth datasets can need tens of GB).
    os.environ.setdefault("HF_DATASETS_CACHE", str(output / ".hf_datasets_cache"))

    datasets = discover(source)
    if not datasets:
        raise SystemExit(
            f"no sub-datasets found under {source} (expected <user>/<ds>/meta/info.json)"
        )

    if args.datasets is not None:
        by_name = {d.name: d for d in datasets}
        unknown = [n for n in args.datasets if n not in by_name]
        if unknown:
            raise SystemExit(f"unknown datasets: {unknown}")
        datasets = [by_name[n] for n in args.datasets]

    print_census(datasets, output)
    if args.stats_only:
        return

    manifest_path = output / "conversion_manifest.jsonl"
    outcomes: Counter = Counter()

    todo: list[SubDataset] = []
    for ds in datasets:
        if not args.force and is_converted(output, ds.name):
            outcomes["skipped"] += 1
        elif ds.version not in SUPPORTED_SOURCE_VERSIONS:
            outcomes["unsupported"] += 1
            append_manifest(
                manifest_path,
                ds.name,
                {"status": "unsupported", "source_version": ds.version},
            )
        else:
            todo.append(ds)
    if outcomes["skipped"]:
        print(f"skipping {outcomes['skipped']} already-converted dataset(s)")

    if args.workers <= 1:
        for ds in tqdm(todo, unit="dataset"):
            result = run_safely(ds, output)
            outcomes[result["status"]] += 1
            append_manifest(manifest_path, ds.name, result)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_safely, ds, output): ds for ds in todo}
            for future in tqdm(as_completed(futures), total=len(futures), unit="dataset"):
                ds = futures[future]
                result = future.result()
                outcomes[result["status"]] += 1
                append_manifest(manifest_path, ds.name, result)

    staging_root = output / ".staging"
    if staging_root.exists() and not any(staging_root.rglob("*")):
        shutil.rmtree(staging_root)
    cache_root = output / ".hf_datasets_cache"
    if os.environ.get("HF_DATASETS_CACHE") == str(cache_root) and cache_root.exists():
        shutil.rmtree(cache_root, ignore_errors=True)

    print(f"\ndone: {dict(outcomes)}")
    if outcomes["failed"]:
        print(f"failures are quarantined in the manifest: {manifest_path}")


def run_safely(ds: SubDataset, output: Path) -> dict:
    try:
        return convert_one(ds, output)
    except Exception as error:  # noqa: BLE001 - quarantine and continue the sweep
        traceback.print_exc()
        return {
            "status": "failed",
            "source_version": ds.version,
            "error": str(error),
        }


def append_manifest(manifest_path: Path, name: str, result: dict) -> None:
    with manifest_path.open("a") as f:
        f.write(json.dumps({"dataset": name, "time": time.strftime("%F %T"), **result}) + "\n")


if __name__ == "__main__":
    main()
