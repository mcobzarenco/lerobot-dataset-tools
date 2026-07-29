# lerobot-dataset-tools

Tools for migrating [LeRobot](https://github.com/huggingface/lerobot)
community datasets to the **v3.0 dataset format** and curating their episodes
with VLM judges.

Built to convert the crowdsourced collections
[`community_dataset_v1`](https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v1) /
[`v2`](https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v2) /
[`v3`](https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v3)
(~1,250 sub-datasets, ~68k episodes, ~1.3 TB, in a mix of legacy v2.0/v2.1
formats) into directly-loadable v3.0 mirrors, and to filter them down to a
high-quality training corpus.

**If you just want the data**, the converted collections are published on the
Hugging Face hub — no conversion needed, each with a card documenting exactly
what was repaired along the way:

| collection | datasets | episodes | size |
|---|---|---|---|
| [`mcobzarenco/community_dataset_v1_v3`](https://huggingface.co/datasets/mcobzarenco/community_dataset_v1_v3) | 128 | 11,132 | 129 GB |
| [`mcobzarenco/community_dataset_v2_v3`](https://huggingface.co/datasets/mcobzarenco/community_dataset_v2_v3) | 323 | 12,912 | 122 GB |
| [`mcobzarenco/community_dataset_v3_v3`](https://huggingface.co/datasets/mcobzarenco/community_dataset_v3_v3) | 791 | 50,614 | 737 GB |

## Install

Requires [uv](https://docs.astral.sh/uv/) and system `ffmpeg`.

```bash
uv sync
```

## Convert a collection to v3.0

A "collection" is a directory tree of independent LeRobot datasets laid out
as `<root>/<user>/<dataset>` (each with `meta/`, `data/`, `videos/`).

```bash
# census only: version/episode/frame/size breakdown
uv run python -m ldtools.convert_collection \
    --source /data/community_dataset_v2 --output /data/community_dataset_v2_v3 --stats-only

# full sweep (idempotent + resumable; failures quarantined to a manifest)
uv run python -m ldtools.convert_collection \
    --source /data/community_dataset_v2 --output /data/community_dataset_v2_v3 --workers 12

# subset / redo
... --datasets ZGGZZG/so100_drop0 ad330/cubePlace [--force]
```

Per sub-dataset the pipeline: stages a copy (skipping stray duplicate video
trees and junk files) → drops parquet columns not declared as features →
repairs the collection-wide flat-stats-key bug → synthesizes per-episode
stats for v2.0 sources (v2.0→v2.1 hop) → runs the official lerobot
v2.1→v3.0 converter with an ffmpeg concat fallback for PyAV-hostile
timestamps → validates the result end-to-end (`LeRobotDataset` reload +
episode/frame/video-span consistency). One JSON line per dataset lands in
`<output>/conversion_manifest.jsonl`.

Known source-data issues it handles automatically (all found in the wild):

| issue | handling |
|---|---|
| stats keyed `observation.<cam>` but features `observation.images.<cam>` | re-keyed before conversion |
| every video duplicated in a stray flat tree | stray trees skipped (≈2× smaller output) |
| undeclared legacy parquet columns (`next.done`, …) | dropped |
| duplicate/non-monotonic DTS at episode boundaries | ffmpeg concat fallback |
| v2.0 sources (no `episodes_stats.jsonl`) | per-episode stats synthesized |
| datasets declaring video features but shipping no videos | fail fast, quarantined |

## Backfill quantile stats

Newer LeRobot versions write per-feature quantiles (`q01/q10/q50/q90/q99`)
into `meta/stats.json`; older sources (and the converted collections) lack
them — and where they DO exist natively they are **wrong for corpus use**:
LeRobot aggregates dataset-level quantiles as a count-weighted *mean* of
per-episode quantiles, but quantiles don't compose by averaging, so extreme
quantiles regress toward the median on dimensions that are near-constant
within episodes (measured: native q01 −54° vs exact −120° on a 50-episode
SO-101 set; mean/std/min/max aggregate correctly and are unaffected). This
tool computes **exact corpus quantiles** for `action` and
`observation.state` and merges them in place; use `--force` to correct
datasets carrying native aggregated values (do this before fitting anything
that consumes quantile normalization, e.g. FAST action tokenizers):

```bash
uv run python -m ldtools.backfill_quantile_stats \
    --root /data/community_dataset_v1_v3 --force

# subset / inspect without writing
... --datasets ZGGZZG/so100_drop0 --dry-run
```

## Generate the dataset card for a converted collection

```bash
uv run python -m ldtools.dataset_card \
    --source /data/community_dataset_v2 --output /data/community_dataset_v2_v3 \
    --source-repo HuggingFaceVLA/community_dataset_v2 \
    --target-repo <user>/community_dataset_v2_v3 --write
```

Produces a README with before/after statistics, the list of quarantined
datasets with reasons, and exact reproduction commands.

## Upload

```bash
uv run hf upload <user>/community_dataset_v2_v3 \
    /data/community_dataset_v2_v3 --repo-type=dataset
```

## Judge episode quality (for filtering, relabeling and camera tagging)

The judges emit a strict-JSON verdict from sampled frames + the task
instruction + full-trajectory statistics. Schema v2 (`judge_episode`, the
Anthropic judge) contains:

- `overall_score` 1–10, `verdict` keep/review/discard, per-aspect scores,
  `issues`, `summary` — quality of the **demonstration**;
- `instruction_quality` good/vague/mismatched/placeholder — quality of the
  **label**, judged separately (community task strings are frequently junk
  like "test1" on top of usable demos: relabel, don't discard);
- `observed_task` + `suggested_instructions` — grounded relabeling
  candidates, usable directly as training instructions;
- `camera_kinds` — per-camera viewpoint map (`wrist`/`top`/`front`/`side`/
  `unknown`), judged **visually**: the converted collections use anonymized
  camera names (`image`, `image2`, ...) whose order is inconsistent across
  datasets, so names carry no signal. Enables train-time camera annotations
  (with `unknown` as a natural dropout target).

```bash
# Anthropic API (needs ANTHROPIC_API_KEY)
uv run python -m ldtools.judge_episode \
    --root /data/community_dataset_v2_v3/<user>/<ds> --episode 3 [--json] [--dry-run]

# Local Gemma 4 12B (transformers; --load-in-4bit for small GPUs).
# NOTE: still on schema v1 (no instruction/camera fields yet).
uv run python -m ldtools.judge_episode_gemma \
    --root /data/community_dataset_v2_v3/<user>/<ds> --episode 3 --image-token-budget 280
```

## Judge sweeps (many datasets, resumable)

`judge_sweep` drives the Anthropic judge over whole collections: skips
episodes shorter than `--min-frames` (default 50 = one action chunk) loudly
with a recorded reason, subsamples `--episodes-per-dataset` evenly across
each dataset's episode range, runs `--workers` episodes concurrently
(process-isolated — a decoder crash on a corrupt video fails one episode,
not the sweep), and appends one JSON line per episode to `--output`.
Re-running skips already-recorded episodes; `--retry-failed` re-attempts
failures; mixing prompt versions in one log is refused unless
`--allow-mixed`. API verdicts are non-deterministic (newer models reject
`temperature`, and the API never guaranteed determinism even at
`temperature=0`); every record carries the model id + prompt version for
provenance instead.

```bash
# plan + rough cost, no API calls
uv run python -m ldtools.judge_sweep \
    --roots /data/community_dataset_v1_v3 --output verdicts.jsonl --dry-run

# calibration pilot: 2 episodes per dataset, hard cap 200 calls
uv run python -m ldtools.judge_sweep \
    --roots /data/community_dataset_v1_v3 /data/community_dataset_v2_v3 \
    --output verdicts.jsonl --episodes-per-dataset 2 --max-episodes 200 --workers 4
```

Calibrate before filtering at scale: judge prompts are strict, and verdicts
should be validated against a hand-labeled sample before trusting them on a
full corpus. Camera-kind tags in particular should be aggregated per dataset
(majority vote across episodes) — single-episode tags flip on ambiguous
views (measured on the pilot: downward-looking wrist cams vs `top`).

## License

Apache-2.0. The community datasets themselves are Apache-2.0 per their
collection cards; converted mirrors preserve provenance in their READMEs.
