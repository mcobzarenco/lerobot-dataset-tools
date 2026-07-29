"""Batch episode-quality judging across LeRobot v3.0 collections (Anthropic API).

Drives ldtools.judge_episode's evidence gathering + prompt over many datasets
and episodes in parallel, writing one JSON line per episode to a resumable
verdict log. Episodes shorter than --min-frames are skipped loudly (recorded
with a reason, counted) — they cannot fill one action chunk and are filtered
mechanically, no judge needed.

Usage:
    # plan only: what would run, rough token/cost estimate
    uv run python -m ldtools.judge_sweep \
        --roots /data/community_dataset_v1_v3 --output verdicts.jsonl --dry-run

    # pilot: 2 episodes per dataset, 4 concurrent workers
    uv run python -m ldtools.judge_sweep \
        --roots /data/community_dataset_v1_v3 /data/community_dataset_v2_v3 \
        --output verdicts.jsonl --episodes-per-dataset 2 --workers 4

    # everything (resumable: already-recorded episodes are not re-judged)
    uv run python -m ldtools.judge_sweep --roots ... --output verdicts.jsonl

Records carry the model id and prompt version; the sweep refuses to append
records that would mix prompt versions in one log unless --allow-mixed.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from anthropic import Anthropic, APIError

from ldtools.judge_episode import (
    DEFAULT_MAX_IMAGE_DIM,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_NUM_FRAMES,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    EpisodeJudgment,
    build_user_content,
    load_episode_summary,
)

# Sweep-specific CLI defaults (the judge knobs above are shared with the
# single-episode CLI so the two can never drift apart).
DEFAULT_MIN_FRAMES = 50  # = one action chunk; shorter episodes are filtered
DEFAULT_WORKERS = 4

# Rough $/Mtok (input, output) as of 2026-07; used only for --dry-run and the
# end-of-run summary, clearly labeled as estimates. Unknown models get none.
MODEL_PRICES = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
# ~(512*384)/750 vision tokens for a 640x480 frame thumbnailed to 512, plus
# text overhead measured on pilot episodes; rough by design. Calibrated on
# the Sonnet 4.5 tokenizer — models >= 4.7 tokenize text ~30% heavier, so
# expect estimates to run a bit low there.
EST_TOKENS_PER_IMAGE = 262
EST_TEXT_TOKENS = 1400
EST_OUTPUT_TOKENS = 450


@dataclass(frozen=True)
class JudgeTask:
    """One episode to judge (picklable work unit for the process pool)."""

    root: str  # dataset directory
    repo_id: str
    episode: int
    num_timesteps: int
    max_image_dim: int
    model: str
    max_tokens: int


@dataclass(frozen=True)
class DatasetPlan:
    """Planning outcome for one dataset."""

    root: Path
    repo_id: str
    cameras: int
    to_judge: list[int]
    skipped: list[tuple[int, int]]  # (episode, length) below --min-frames


def discover_datasets(roots: list[Path]) -> list[Path]:
    """Dataset dirs under collection roots (or roots that are datasets)."""
    found: list[Path] = []
    for root in roots:
        root = root.expanduser().resolve()
        if (root / "meta" / "info.json").exists():
            found.append(root)
            continue
        nested = sorted(p.parent.parent for p in root.glob("*/*/meta/info.json"))
        if not nested:
            raise SystemExit(f"no LeRobot datasets under {root}")
        found.extend(nested)
    return found


def plan_dataset(
    dataset_dir: Path,
    min_frames: int,
    episodes_per_dataset: int | None,
) -> DatasetPlan:
    """Choose episodes to judge from metadata only (no video access).

    Episodes below ``min_frames`` are skipped with a reason. When
    ``episodes_per_dataset`` is set, eligible episodes are subsampled evenly
    across the episode index range (deterministic, covers session drift
    within a recording day better than the first N).
    """
    repo_id = f"{dataset_dir.parent.name}/{dataset_dir.name}"
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    cameras = sum(
        1 for feature in (info.get("features") or {}).values() if feature.get("dtype") == "video"
    )
    parquets = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not parquets:
        raise ValueError(f"{repo_id}: no meta/episodes/**/*.parquet")
    episodes = pd.concat(
        [pd.read_parquet(p, columns=["episode_index", "length"]) for p in parquets]
    ).sort_values("episode_index")

    eligible: list[int] = []
    skipped: list[tuple[int, int]] = []
    for episode, length in zip(episodes["episode_index"], episodes["length"]):
        if int(length) < min_frames:
            skipped.append((int(episode), int(length)))
        else:
            eligible.append(int(episode))

    if episodes_per_dataset is not None and len(eligible) > episodes_per_dataset:
        picks = np.unique(
            np.linspace(0, len(eligible) - 1, episodes_per_dataset).round().astype(int)
        )
        eligible = [eligible[i] for i in picks]

    return DatasetPlan(
        root=dataset_dir,
        repo_id=repo_id,
        cameras=cameras,
        to_judge=eligible,
        skipped=skipped,
    )


# --- worker ----------------------------------------------------------------
# One process judges one episode end-to-end: decode frames, call the API,
# parse + validate the verdict. Process isolation also contains the AV1
# decoder on corrupt community videos (segfaults kill the worker, not the
# sweep). Caches below are per-process.

_CLIENT: Anthropic | None = None


def _client() -> Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Anthropic(max_retries=5)  # SDK backoff handles 429/529
    return _CLIENT


def judge_one(task: JudgeTask) -> dict:
    started = time.perf_counter()
    record: dict = {
        "dataset": task.repo_id,
        "episode": task.episode,
        "time": time.strftime("%F %T", time.gmtime()),
        "model": task.model,
        "prompt_version": PROMPT_VERSION,
    }
    try:
        summary = load_episode_summary(
            root=Path(task.root),
            repo_id=task.repo_id,
            episode=task.episode,
            num_timesteps=task.num_timesteps,
            max_image_dim=task.max_image_dim,
            cameras=None,
        )
        content = build_user_content(summary)
        response = _client().messages.create(
            model=task.model,
            max_tokens=task.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,  # reproducible verdicts
        )
        raw = "".join(block.text for block in response.content if block.type == "text")
        judgment = EpisodeJudgment.from_response_text(raw)
        judgment.check_cameras(summary.camera_names)
        record.update(
            status="ok",
            task=summary.task,
            num_frames=summary.num_frames,
            duration_s=round(summary.duration_s, 2),
            fps=summary.fps,
            cameras=summary.camera_names,
            judgment=judgment.to_dict(),
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )
    except APIError as error:
        record.update(status="failed", error=f"api: {error}")
    except Exception as error:  # noqa: BLE001 - quarantine and continue the sweep
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    record["seconds"] = round(time.perf_counter() - started, 2)
    return record


# --- driver -----------------------------------------------------------------


def load_done(output: Path, retry_failed: bool) -> tuple[set[tuple[str, int]], set[int]]:
    """Keys already recorded (to skip) and prompt versions seen in the log."""
    done: set[tuple[str, int]] = set()
    versions: set[int] = set()
    if not output.exists():
        return done, versions
    with output.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            versions.add(int(record.get("prompt_version", 1)))
            if retry_failed and record.get("status") == "failed":
                continue
            done.add((record["dataset"], int(record["episode"])))
    return done, versions


def estimate_cost(episodes: int, images: int, model: str) -> str:
    input_tokens = images * EST_TOKENS_PER_IMAGE + episodes * EST_TEXT_TOKENS
    output_tokens = episodes * EST_OUTPUT_TOKENS
    tokens = f"~{input_tokens:,} in / ~{output_tokens:,} out tokens"
    for prefix, (in_price, out_price) in MODEL_PRICES.items():
        if model.startswith(prefix):
            dollars = (input_tokens * in_price + output_tokens * out_price) / 1e6
            return f"{tokens}, ~${dollars:,.2f} ({model}, rough)"
    return f"{tokens} (no price table for {model})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge many LeRobot episodes with the Anthropic API, resumably."
    )
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        required=True,
        help="Collection roots (<root>/<user>/<dataset>) and/or dataset dirs.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Subset as '<user>/<dataset>' repo ids (default: all discovered).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Verdict JSONL, one record per episode (appended; existing records "
        "are not re-judged).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=DEFAULT_MIN_FRAMES,
        help="Skip (and record) episodes shorter than this many frames; the default "
        "matches the 50-step action chunk — shorter episodes are mechanically "
        "filtered, no judge needed (default: %(default)s).",
    )
    parser.add_argument(
        "--episodes-per-dataset",
        type=int,
        default=None,
        help="Judge at most N episodes per dataset, evenly spaced over the episode "
        "index range (default: all eligible).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Hard cap on API calls this run, a safety valve for pilots (default: no cap).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Sampled timesteps per episode, each shown for every camera (default: %(default)s).",
    )
    parser.add_argument(
        "--max-image-dim",
        type=int,
        default=DEFAULT_MAX_IMAGE_DIM,
        help="Frames are downscaled so the longer side is at most this many pixels "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Anthropic model id (default: %(default)s).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum response tokens per verdict (default: %(default)s).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Concurrent judge processes; each decodes frames and holds one API "
        "call in flight (default: %(default)s).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-attempt episodes whose existing record has status=failed.",
    )
    parser.add_argument(
        "--allow-mixed",
        action="store_true",
        help="Append to a log recorded with a different prompt version.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan and estimate only.")
    args = parser.parse_args()

    dataset_dirs = discover_datasets(args.roots)
    if args.datasets is not None:
        wanted = set(args.datasets)
        by_id = {f"{d.parent.name}/{d.name}": d for d in dataset_dirs}
        unknown = wanted - set(by_id)
        if unknown:
            raise SystemExit(f"unknown datasets: {sorted(unknown)}")
        dataset_dirs = [by_id[name] for name in sorted(wanted)]

    plans: list[DatasetPlan] = []
    plan_failures: list[tuple[str, str]] = []
    for dataset_dir in dataset_dirs:
        try:
            plans.append(plan_dataset(dataset_dir, args.min_frames, args.episodes_per_dataset))
        except Exception as error:  # noqa: BLE001 - record and continue planning
            repo_id = f"{dataset_dir.parent.name}/{dataset_dir.name}"
            plan_failures.append((repo_id, str(error)))
            print(f"PLAN FAILED {repo_id}: {error}", file=sys.stderr)

    done, versions = load_done(args.output, retry_failed=args.retry_failed)
    if versions and versions != {PROMPT_VERSION} and not args.allow_mixed:
        raise SystemExit(
            f"{args.output} contains prompt version(s) {sorted(versions)} but this code is "
            f"version {PROMPT_VERSION}; use a fresh --output or --allow-mixed"
        )

    cameras_by_repo = {plan.repo_id: plan.cameras for plan in plans}
    tasks: list[JudgeTask] = []
    new_skips: list[dict] = []
    for plan in plans:
        for episode, length in plan.skipped:
            if (plan.repo_id, episode) not in done:
                new_skips.append(
                    {
                        "dataset": plan.repo_id,
                        "episode": episode,
                        "time": time.strftime("%F %T", time.gmtime()),
                        "status": "skipped",
                        "reason": f"length {length} < --min-frames {args.min_frames}",
                        "prompt_version": PROMPT_VERSION,
                    }
                )
        for episode in plan.to_judge:
            if (plan.repo_id, episode) in done:
                continue
            tasks.append(
                JudgeTask(
                    root=str(plan.root),
                    repo_id=plan.repo_id,
                    episode=episode,
                    num_timesteps=args.num_frames,
                    max_image_dim=args.max_image_dim,
                    model=args.model,
                    max_tokens=args.max_tokens,
                )
            )

    if args.max_episodes is not None and len(tasks) > args.max_episodes:
        tasks = tasks[: args.max_episodes]
    total_images = sum(args.num_frames * cameras_by_repo[task.repo_id] for task in tasks)

    planned = sum(len(p.to_judge) for p in plans)
    skipped_total = sum(len(p.skipped) for p in plans)
    print(
        f"plan: {len(plans)} datasets | {planned} episodes eligible | "
        f"{skipped_total} below {args.min_frames} frames | {len(done)} already recorded | "
        f"{len(tasks)} to judge now | {len(plan_failures)} datasets failed to plan"
    )
    print(f"cost: {estimate_cost(len(tasks), total_images, args.model)}")
    if args.dry_run:
        for repo_id, error in plan_failures:
            print(f"  plan failure: {repo_id}: {error}")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (required unless --dry-run)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    outcomes = {"ok": 0, "failed": 0, "skipped": len(new_skips)}
    tokens_in = tokens_out = 0
    with args.output.open("a") as log:
        for skip in new_skips:
            log.write(json.dumps(skip) + "\n")
        for repo_id, error in plan_failures:
            log.write(
                json.dumps(
                    {
                        "dataset": repo_id,
                        "episode": -1,
                        "time": time.strftime("%F %T", time.gmtime()),
                        "status": "failed",
                        "error": f"planning: {error}",
                        "prompt_version": PROMPT_VERSION,
                    }
                )
                + "\n"
            )
        log.flush()
        if not tasks:
            print("nothing to do")
            return

        # spawn (not fork): workers decode video; forking a torch-loaded
        # parent into AV1 decoders is asking for latent corruption.
        context = mp.get_context("spawn")
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = {pool.submit(judge_one, task): task for task in tasks}
            for i, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                log.write(json.dumps(record) + "\n")
                log.flush()
                outcomes[record["status"]] += 1
                usage = record.get("usage")
                if usage:
                    tokens_in += usage["input_tokens"]
                    tokens_out += usage["output_tokens"]
                if record["status"] == "failed":
                    print(
                        f"FAILED {record['dataset']} ep {record['episode']}: {record['error']}",
                        file=sys.stderr,
                    )
                if i % 25 == 0 or i == len(tasks):
                    rate = i / (time.perf_counter() - started)
                    print(
                        f"[{i}/{len(tasks)}] ok={outcomes['ok']} failed={outcomes['failed']} "
                        f"| {tokens_in:,} in / {tokens_out:,} out tokens | {rate:.2f} eps/s",
                        flush=True,
                    )

    for prefix, (in_price, out_price) in MODEL_PRICES.items():
        if args.model.startswith(prefix):
            spent = (tokens_in * in_price + tokens_out * out_price) / 1e6
            print(f"spent: ~${spent:,.2f} ({tokens_in:,} in / {tokens_out:,} out tokens, rough)")
            break
    print(f"done: {outcomes} -> {args.output}")


if __name__ == "__main__":
    main()
