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

## Judge episode quality (for filtering)

Two independent judges emit the same strict-JSON verdict (overall score 1–10,
keep/review/discard, sub-scores, issues) from sampled frames + the task
instruction + full-trajectory statistics:

```bash
# Anthropic API (needs ANTHROPIC_API_KEY)
uv run python -m ldtools.judge_episode \
    --root /data/community_dataset_v2_v3/<user>/<ds> --episode 3 [--json] [--dry-run]

# Local Gemma 4 12B (transformers; --load-in-4bit for small GPUs)
uv run python -m ldtools.judge_episode_gemma \
    --root /data/community_dataset_v2_v3/<user>/<ds> --episode 3 --image-token-budget 280
```

Calibrate before filtering at scale: judge prompts are strict, and verdicts
should be validated against a hand-labeled sample before trusting them on a
full corpus.

## License

Apache-2.0. The community datasets themselves are Apache-2.0 per their
collection cards; converted mirrors preserve provenance in their READMEs.
