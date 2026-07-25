"""Generate a dataset card (README.md) for a converted collection.

Reads the source tree, the converted tree, and the conversion manifest
written by ``ldtools.convert_collection``, and emits a README documenting:
what the transformation did, before/after statistics, per-dataset failures,
and exact reproduction commands.

Usage:
    uv run python -m ldtools.dataset_card \
        --source /data/community_dataset_v2 \
        --output /data/community_dataset_v2_v3 \
        --source-repo HuggingFaceVLA/community_dataset_v2 \
        --target-repo <user>/community_dataset_v2_v3 \
        --write          # write <output>/README.md (default: stdout)
"""

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TOOLS_URL = "https://github.com/mcobzarenco/lerobot-dataset-tools"


@dataclass(frozen=True)
class TreeCensus:
    datasets: int
    episodes: int
    frames: int
    size_bytes: int
    by_version: dict[str, int]


def census(root: Path) -> TreeCensus:
    by_version: Counter[str] = Counter()
    datasets = episodes = frames = size = 0
    for info_path in sorted(root.glob("*/*/meta/info.json")):
        info = json.loads(info_path.read_text())
        datasets += 1
        episodes += int(info.get("total_episodes", 0))
        frames += int(info.get("total_frames", 0))
        by_version[str(info.get("codebase_version", "?"))] += 1
        ds_dir = info_path.parent.parent
        size += sum(f.stat().st_size for f in ds_dir.rglob("*") if f.is_file())
    return TreeCensus(datasets, episodes, frames, size, dict(by_version))


def load_final_manifest(manifest_path: Path) -> dict[str, dict]:
    """Last entry per dataset wins (reruns append)."""
    final: dict[str, dict] = {}
    if manifest_path.is_file():
        for line in manifest_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                final[record["dataset"]] = record
    return final


def fmt_gb(size_bytes: int) -> str:
    return f"{size_bytes / 1e9:.1f} GB"


def build_correction_rows(manifest: dict[str, dict]) -> list[tuple[str, str]]:
    """Human-readable per-dataset corrections from structured manifest fields."""
    rows: list[tuple[str, str]] = []
    for name, record in sorted(manifest.items()):
        parts: list[str] = []
        aggregates = record.get("corrections", {}).get("aggregates", {})
        for key, change in aggregates.items():
            if key == "splits":
                parts.append("stale `splits` recomputed")
            else:
                parts.append(f"`{key}` {change['from']} → {change['to']}")
        dropped = record.get("corrections", {}).get("dropped_corrupt_episodes")
        if dropped:
            parts.append(f"dropped {len(dropped)} corrupt episode(s) {dropped}")
        repaired = record.get("corrections", {}).get("repaired_episode_lengths")
        if repaired:
            parts.append(f"repaired {repaired} declared episode length(s)")
        if record.get("dropped_features"):
            parts.append(f"dropped v3-incompatible features {record['dropped_features']}")
        if record.get("status") == "patched" and record.get("note"):
            parts.append(record["note"])
        if parts:
            rows.append((name, "; ".join(parts)))
    return rows


def build_card(
    source: Path,
    output: Path,
    source_repo: str,
    target_repo: str,
) -> str:
    src = census(source)
    dst = census(output)
    converted_names = sorted(
        "/".join(p.parent.parent.parts[-2:]) for p in output.glob("*/*/meta/info.json")
    )
    example_name = converted_names[0] if converted_names else "<user>/<dataset>"
    manifest = load_final_manifest(output / "conversion_manifest.jsonl")
    failed = {k: v for k, v in manifest.items() if v["status"] == "failed"}
    unsupported = {k: v for k, v in manifest.items() if v["status"] == "unsupported"}

    src_versions = ", ".join(f"{n}× {v}" for v, n in sorted(src.by_version.items()))
    collection = source_repo.split("/")[-1]

    lines: list[str] = []
    a = lines.append

    a("---")
    a("license: apache-2.0")
    a("viewer: false")  # umbrella repo: many independent datasets, no single schema
    a("tags:")
    a("- LeRobot")
    a("- robotics")
    a(f"- {collection}")
    a("task_categories:")
    a("- robotics")
    a("---")
    a("")
    a(f"# {target_repo.split('/')[-1]}")
    a("")
    a(
        f"[`{source_repo}`](https://huggingface.co/datasets/{source_repo}) with every "
        "sub-dataset migrated to the **LeRobot v3.0 dataset format**, so it loads "
        "directly with lerobot ≥ 0.4 (`LeRobotDataset(repo_id, root=...)`) without "
        "per-dataset conversion."
    )
    a("")
    a(
        f"Conversion is fully reproducible with [lerobot-dataset-tools]({TOOLS_URL}); "
        "the exact per-dataset log is in [`conversion_manifest.jsonl`](./conversion_manifest.jsonl)."
    )
    a("")
    a("## Usage")
    a("")
    a(
        "This is an *umbrella* repo: every `<user>/<dataset>` subtree is an "
        "independent LeRobotDataset (hence no dataset viewer). Download a "
        "subtree and point lerobot at it:"
    )
    a("")
    a("```python")
    a("from huggingface_hub import snapshot_download")
    a("from lerobot.datasets.lerobot_dataset import LeRobotDataset")
    a("")
    a(f'name = "{example_name}"  # any sub-dataset in this repo')
    a("root = snapshot_download(")
    a(f'    "{target_repo}",')
    a('    repo_type="dataset",')
    a('    allow_patterns=[f"{name}/**"],')
    a('    local_dir="./data",')
    a(")")
    a('dataset = LeRobotDataset(name, root=f"./data/{name}")')
    a("```")
    a("")
    a("or with the CLI:")
    a("")
    a("```bash")
    a(
        f'hf download {target_repo} --repo-type=dataset --include "{example_name}/**" --local-dir ./data'
    )
    a("```")
    a("")
    a("## Statistics")
    a("")
    a("| | source | converted |")
    a("|---|---|---|")
    a(f"| sub-datasets | {src.datasets} ({src_versions}) | {dst.datasets} (all v3.0) |")
    a(f"| episodes | {src.episodes:,} | {dst.episodes:,} |")
    a(f"| frames | {src.frames:,} | {dst.frames:,} |")
    a(f"| size | {fmt_gb(src.size_bytes)} | {fmt_gb(dst.size_bytes)} |")
    a("")
    if src.size_bytes > dst.size_bytes * 1.2:
        a(
            "The size reduction is not compression: the source repos ship most videos "
            "twice (a stray flat tree next to the real one); only the canonical copies "
            "are converted. Videos are stream-copied, never re-encoded."
        )
        a("")
    a("## What the conversion does")
    a("")
    a("Per sub-dataset (see the tools repo for code):")
    a("")
    a("1. Copy the dataset, skipping stray duplicate video trees and `*.bak` files.")
    a("2. Drop parquet columns not declared in `info.json` features (legacy leftovers")
    a("   such as `next.done` break v3.0 schema casting).")
    a(
        "3. Repair the collection-wide stats-key bug (stats keyed `observation.<cam>` "
        "while features are `observation.images.<cam>`)."
    )
    a(
        "4. For v2.0 sources: synthesize `meta/episodes_stats.jsonl` (numeric stats "
        "exact from parquet; per-episode image stats from ~100 sampled frames per "
        "camera, matching lerobot's own conventions) and bump to v2.1."
    )
    a(
        "5. Run the official `lerobot.scripts.convert_dataset_v21_to_v30` converter "
        "(with an ffmpeg concat fallback for episodes whose packets carry "
        "duplicate/non-monotonic DTS, which PyAV refuses to mux)."
    )
    a(
        "6. Validate: reload with `LeRobotDataset`, check episode/frame counts, and "
        "verify per-episode `length == index-range == video-span × fps` per camera."
    )
    a("")
    if failed or unsupported:
        a("## Datasets not included")
        a("")
        a("| dataset | source version | reason |")
        a("|---|---|---|")
        for name, record in sorted({**failed, **unsupported}.items()):
            reason = record.get("error", record["status"]).replace("|", "\\|")[:160]
            a(f"| `{name}` | {record.get('source_version', '?')} | {reason} |")
        a("")
    a("## Reproduce")
    a("")
    a("```bash")
    a(f"git clone {TOOLS_URL} && cd lerobot-dataset-tools")
    a(f"uv run hf download {source_repo} --repo-type=dataset --local-dir ./source")
    a("uv run python -m ldtools.convert_collection \\")
    a("    --source ./source --output ./converted --workers 12")
    a("uv run python -m ldtools.dataset_card \\")
    a("    --source ./source --output ./converted \\")
    a(f"    --source-repo {source_repo} --target-repo {target_repo} --write")
    a("```")
    a("")
    a("## Source-data corrections")
    a("")
    a(
        "Some source datasets ship metadata that contradicts their own data "
        "(e.g. inflated `total_frames` counters, which make "
        "`LeRobotDataset.__len__` overreport and crash shuffled samplers with "
        "IndexError). The converter recomputes every aggregate from the actual "
        "episode table and records each correction in the manifest. Corrections "
        "applied in this collection:"
    )
    a("")
    correction_rows = build_correction_rows(manifest)
    if correction_rows:
        a("| dataset | correction |")
        a("|---|---|")
        for dataset_name, description in correction_rows:
            a(f"| `{dataset_name}` | {description} |")
    else:
        a("*(none needed)*")
    a("")
    a(
        "Remaining known quirk: a handful of episodes across the collections "
        "have a few trailing video frames beyond their declared length (source "
        "recording artifacts, not repairable without re-encoding). They load, "
        "decode and train normally — frames are indexed via episode metadata, so "
        "the tail frames are simply never read."
    )
    a("")
    a("## Provenance & license")
    a("")
    a(
        f"All data originates from [`{source_repo}`](https://huggingface.co/datasets/{source_repo}) "
        "(community-contributed robot demonstrations, Apache-2.0). This repo changes "
        "the storage format only: actions, states, task strings and video content "
        "are unmodified (videos are stream-copied; v2.0 per-episode image stats are "
        "sampled approximations)."
    )
    a("")
    a(f"*Generated {time.strftime('%Y-%m-%d')} by [lerobot-dataset-tools]({TOOLS_URL}).*")
    a("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-repo", type=str, required=True)
    parser.add_argument("--target-repo", type=str, required=True)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write <output>/README.md instead of printing to stdout.",
    )
    args = parser.parse_args()

    card = build_card(
        args.source.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.source_repo,
        args.target_repo,
    )
    if args.write:
        readme = args.output.expanduser().resolve() / "README.md"
        readme.write_text(card)
        print(f"wrote {readme}")
    else:
        print(card)


if __name__ == "__main__":
    main()
