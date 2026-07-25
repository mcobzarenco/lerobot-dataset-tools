"""Local LLM-as-judge for LeRobot v3.0 episodes, powered by Gemma 4.

Runs entirely on this machine with the `transformers` library: samples frames
from an episode, computes trajectory statistics, and asks a local Gemma 4
instance (default: google/gemma-4-12B-it) for a structured quality verdict.

Usage:
    uv run python -m fmatch.judge_episode_gemma \
        --root /home/marius/w/community_dataset_v1_v3/ZGGZZG/so100_drop0 \
        --episode 3 --load-in-4bit

    # tokenize + report context length without loading model weights
    uv run python -m fmatch.judge_episode_gemma --root ... --episode 3 --dry-run

Notes:
  - On an 8 GB GPU, pass --load-in-4bit (bitsandbytes NF4); bf16 weights of
    the 12B model need ~24 GB and would otherwise be offloaded to CPU RAM.
  - Generation is greedy by default so verdicts are reproducible. Pass
    --temperature to use the sampling settings recommended on the model card.
  - Gemma 4's "Unified" (encoder-free) architecture supports a per-image
    token budget; tune it with --image-token-budget (70..1120).
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

GEMMA_MODEL_ID = "google/gemma-4-12B-it"
IMAGE_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)

JUDGE_SYSTEM_PROMPT = """\
You are a strict but fair reviewer of robot teleoperation recordings. Each
recording ("episode") is a demonstration that will be used to train an
imitation-learning policy, so flawed demonstrations pollute the training set.

You will receive: chronologically ordered frames sampled from the episode
(every camera at each sampled time), the task instruction given to the
operator, and statistics computed over the full trajectory. The frames are a
sparse sample — base claims about specific moments only on what you can see,
and use the statistics for whole-episode properties.

Watch out for: tasks that visibly fail or cannot be verified, fumbling and
retries, long idle periods, abrupt jerky motion, poor lighting or focus,
cameras that miss the workspace, and inconsistent scene setups.

Answer with EXACTLY one JSON object and nothing else — no prose, no markdown
code fences. Schema:
{
  "overall_score": <integer 1-10>,
  "verdict": "keep" | "review" | "discard",
  "task_completion_visible": "yes" | "partial" | "no" | "unclear",
  "scores": {
    "visual_quality": <integer 1-10>,
    "smoothness": <integer 1-10>,
    "efficiency": <integer 1-10>,
    "camera_framing": <integer 1-10>
  },
  "issues": ["<short issue>", ...],
  "summary": "<2-4 sentence justification>"
}
"""


# ---------------------------------------------------------------------------
# Verdict model
# ---------------------------------------------------------------------------


class Verdict(StrEnum):
    KEEP = "keep"
    REVIEW = "review"
    DISCARD = "discard"


class TaskCompletion(StrEnum):
    YES = "yes"
    PARTIAL = "partial"
    NO = "no"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class AspectScores:
    visual_quality: int
    smoothness: int
    efficiency: int
    camera_framing: int


@dataclass(frozen=True)
class Judgment:
    """Structured verdict produced by the local judge model."""

    overall_score: int
    verdict: Verdict
    task_completion_visible: TaskCompletion
    scores: AspectScores
    issues: tuple[str, ...]
    summary: str

    @classmethod
    def from_dict(cls, payload: dict) -> "Judgment":
        try:
            aspect = payload["scores"]
            return cls(
                overall_score=int(payload["overall_score"]),
                verdict=Verdict(payload["verdict"]),
                task_completion_visible=TaskCompletion(payload["task_completion_visible"]),
                scores=AspectScores(
                    visual_quality=int(aspect["visual_quality"]),
                    smoothness=int(aspect["smoothness"]),
                    efficiency=int(aspect["efficiency"]),
                    camera_framing=int(aspect["camera_framing"]),
                ),
                issues=tuple(str(item) for item in payload.get("issues", [])),
                summary=str(payload.get("summary", "")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"judge output does not match schema: {error}") from error

    @classmethod
    def from_json(cls, text: str) -> "Judgment":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_model_output(cls, text: str) -> "Judgment":
        """Parse from raw generated text; extracts the outermost JSON object."""
        first, last = text.find("{"), text.rfind("}")
        if first < 0 or last <= first:
            raise ValueError("no JSON object in model output")
        return cls.from_json(text[first : last + 1])

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "verdict": self.verdict.value,
            "task_completion_visible": self.task_completion_visible.value,
            "scores": {
                "visual_quality": self.scores.visual_quality,
                "smoothness": self.scores.smoothness,
                "efficiency": self.scores.efficiency,
                "camera_framing": self.scores.camera_framing,
            },
            "issues": list(self.issues),
            "summary": self.summary,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Episode extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeEvidence:
    """Frames + statistics shown to the judge for one episode."""

    repo_id: str
    episode: int
    instruction: str
    fps: float
    num_frames: int
    cameras: list[str]
    stats_block: str
    # (caption, PIL image), chronological; every camera at each sampled step
    frames: list[tuple[str, Image.Image]]


def to_pil(chw: torch.Tensor, max_dim: int) -> Image.Image:
    """float32 CHW in [0,1] -> PIL RGB, bounded to max_dim on the long side."""
    hwc = (chw.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(hwc)
    image.thumbnail((max_dim, max_dim))
    return image


def trajectory_stats(action: np.ndarray, state: np.ndarray, names: list[str], fps: float) -> str:
    steps = np.abs(np.diff(action, axis=0))
    follow_err = np.abs(action - state).mean(axis=0)

    rows = [
        f"{'motor':<22}{'min':>8}{'max':>8}{'mean':>8}{'travel':>9}{'peak-step':>10}{'follow-err':>11}"
    ]
    for i, name in enumerate(names):
        rows.append(
            f"{name:<22}"
            f"{action[:, i].min():>8.1f}"
            f"{action[:, i].max():>8.1f}"
            f"{action[:, i].mean():>8.1f}"
            f"{steps[:, i].sum():>9.1f}"
            f"{steps[:, i].max():>10.1f}"
            f"{follow_err[i]:>11.1f}"
        )

    span = np.maximum(action.max(axis=0) - action.min(axis=0), 1e-6)
    still = float((steps < 0.01 * span).all(axis=1).mean())
    rows += [
        "",
        f"step interval: {1000 / fps:.0f} ms | travel = sum of |per-step change| | "
        "peak-step = largest single-step change | follow-err = mean |commanded - measured|",
        f"fraction of steps with essentially no motion on any motor: {still:.0%}",
        "units are dataset-specific raw joint values",
    ]
    return "\n".join(rows)


def gather_evidence(
    root: Path,
    repo_id: str,
    episode: int,
    timesteps: int,
    max_dim: int,
    cameras: list[str] | None,
) -> EpisodeEvidence:
    dataset = LeRobotDataset(repo_id, root=root)
    if not 0 <= episode < dataset.num_episodes:
        raise SystemExit(f"episode {episode} out of range 0..{dataset.num_episodes - 1}")

    meta_row = dataset.meta.episodes[episode]
    lo, hi = int(meta_row["dataset_from_index"]), int(meta_row["dataset_to_index"])
    length = hi - lo
    fps = float(dataset.fps)

    tasks = meta_row.get("tasks") or ["<none recorded>"]
    instruction = "; ".join(tasks) if isinstance(tasks, list) else str(tasks)

    columns = dataset.hf_dataset[lo:hi]
    action = np.asarray(columns["action"], dtype=np.float32)
    state = np.asarray(columns["observation.state"], dtype=np.float32)
    names = list(
        dataset.meta.features["action"].get("names")
        or [f"motor_{i}" for i in range(action.shape[1])]
    )

    cams = list(dataset.meta.camera_keys)
    if cameras:
        bad = set(cameras) - set(cams)
        if bad:
            raise SystemExit(f"unknown cameras {sorted(bad)}, dataset has {cams}")
        cams = [c for c in cams if c in cameras]

    picks = np.unique(np.linspace(0, length - 1, timesteps).round().astype(int))
    frames: list[tuple[str, Image.Image]] = []
    for offset in picks:
        item = dataset[lo + int(offset)]
        for cam in cams:
            caption = f"[frame {offset + 1}/{length}, t={offset / fps:.1f}s, camera {cam}]"
            frames.append((caption, to_pil(item[cam], max_dim)))

    return EpisodeEvidence(
        repo_id=repo_id,
        episode=episode,
        instruction=instruction,
        fps=fps,
        num_frames=length,
        cameras=cams,
        stats_block=trajectory_stats(action, state, names, fps),
        frames=frames,
    )


# ---------------------------------------------------------------------------
# Chat assembly + local inference
# ---------------------------------------------------------------------------


def build_messages(evidence: EpisodeEvidence, extra_context: str | None) -> list[dict]:
    # Gemma 4 best practice: image content before text. Each frame is
    # followed by its caption, so every image still precedes its text.
    user_content: list[dict] = []
    for caption, image in evidence.frames:
        user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": caption})

    briefing = (
        f"Episode {evidence.episode} of dataset {evidence.repo_id}: "
        f"{evidence.num_frames} frames, {evidence.num_frames / evidence.fps:.1f}s "
        f"at {evidence.fps:.0f} fps, cameras: {', '.join(evidence.cameras)}.\n"
        f'Operator instruction: "{evidence.instruction}"\n'
    )
    if extra_context:
        briefing += f"Context from the dataset owner: {extra_context}\n"
    briefing += (
        f"\nFull-trajectory statistics:\n{evidence.stats_block}\n\n"
        "Assess this demonstration and reply with the JSON object only."
    )
    user_content.append({"type": "text", "text": briefing})

    return [
        {"role": "system", "content": [{"type": "text", "text": JUDGE_SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]


def strip_thought_channel(text: str) -> str:
    """Drop Gemma 4's thought channel and any special-token markup."""
    text = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", text, flags=re.DOTALL)
    return re.sub(r"<\|[^>]*\|>", "", text).strip()


def run_judge(
    messages: list[dict],
    model_id: str,
    load_in_4bit: bool,
    thinking: bool,
    temperature: float | None,
    max_new_tokens: int,
    image_token_budget: int | None,
) -> tuple[str, dict[str, int], float]:
    """Generate a verdict locally. Returns (text, token counts, seconds)."""
    from transformers import (
        AutoModelForImageTextToText,
        AutoModelForMultimodalLM,
        AutoProcessor,
    )

    processor = AutoProcessor.from_pretrained(model_id)

    load_kwargs: dict = {"dtype": "auto", "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    # `Any`: the HF auto-class stubs don't expose `generate` on their
    # common base, and the two branches return different class families.
    model: Any
    try:
        model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs)
    except ValueError:
        # Non any-to-any checkpoints (useful with --model for smoke tests).
        model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    model.eval()

    template_kwargs: dict = {"enable_thinking": thinking}
    if image_token_budget is not None:
        # transformers >=5.14 wants per-call processor kwargs nested.
        template_kwargs["processor_kwargs"] = {"max_soft_tokens": image_token_budget}
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
    ).to(next(model.parameters()).device)
    input_len = int(inputs["input_ids"].shape[-1])

    generate_kwargs: dict = {"max_new_tokens": max_new_tokens}
    if temperature is None:
        generate_kwargs["do_sample"] = False
    else:
        # Sampling settings recommended by the Gemma 4 model card.
        generate_kwargs |= {
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
        }

    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    elapsed = time.perf_counter() - started

    new_tokens = output_ids[0][input_len:]
    decoded = processor.decode(new_tokens, skip_special_tokens=False)

    text: str | None = None
    try:
        parsed = processor.parse_response(decoded)
        if isinstance(parsed, dict):
            text = str(parsed.get("content", ""))
    except Exception:  # noqa: BLE001 - schema-less tokenizers raise freely here
        pass
    if not text:
        text = strip_thought_channel(decoded)

    usage = {"input_tokens": input_len, "output_tokens": int(new_tokens.shape[-1])}
    return text, usage, elapsed


def count_context_tokens(
    messages: list[dict], model_id: str, image_token_budget: int | None = None
) -> int:
    """Tokenize the prompt (processor only, no model weights)."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    extra: dict = {}
    if image_token_budget is not None:
        extra["processor_kwargs"] = {"max_soft_tokens": image_token_budget}
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **extra,
    )
    return int(inputs["input_ids"].shape[-1])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def report(
    evidence: EpisodeEvidence,
    judgment: Judgment | None,
    raw_text: str,
    usage: dict[str, int],
    seconds: float,
    model_id: str,
    as_json: bool,
) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "dataset": evidence.repo_id,
                    "episode": evidence.episode,
                    "task": evidence.instruction,
                    "model": model_id,
                    "usage": usage,
                    "generation_seconds": round(seconds, 2),
                    "judge": judgment.to_dict() if judgment else {"raw_response": raw_text},
                },
                indent=2,
            )
        )
        return

    print(f"=== {evidence.repo_id} — episode {evidence.episode} (judge: {model_id}) ===")
    print(f'task     : "{evidence.instruction}"')
    print(f"length   : {evidence.num_frames} frames @ {evidence.fps:.0f} fps")
    print(
        f"tokens   : {usage['input_tokens']} in / {usage['output_tokens']} out "
        f"({seconds:.1f}s generation)"
    )
    print()
    if judgment is None:
        print("model output did not parse as a verdict; raw text:")
        print(raw_text)
        return
    print(f"overall  : {judgment.overall_score}/10  ->  {judgment.verdict.value}")
    print(f"task done: {judgment.task_completion_visible.value}")
    s = judgment.scores
    print(
        f"scores   : visual_quality={s.visual_quality}  smoothness={s.smoothness}  "
        f"efficiency={s.efficiency}  camera_framing={s.camera_framing}"
    )
    print("issues   : " + ("none noted" if not judgment.issues else ""))
    for issue in judgment.issues:
        print(f"  - {issue}")
    if judgment.summary:
        print(f"summary  : {judgment.summary}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge a LeRobot v3.0 episode with a local Gemma 4 model."
    )
    parser.add_argument("--root", type=Path, required=True, help="Dataset directory (v3.0).")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Defaults to last two components of --root.",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=6, help="Sampled timesteps.")
    parser.add_argument("--cameras", type=str, nargs="*", default=None)
    parser.add_argument("--max-image-dim", type=int, default=512)
    parser.add_argument("--model", type=str, default=GEMMA_MODEL_ID)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Quantize weights to NF4 via bitsandbytes (recommended on 8 GB GPUs).",
    )
    parser.add_argument(
        "--image-token-budget",
        type=int,
        choices=IMAGE_TOKEN_BUDGETS,
        default=None,
        help="Gemma 4 visual token budget per image (default: model's own default).",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Gemma 4's reasoning mode before the final answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Enable sampling at this temperature (default: greedy/deterministic).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--context", type=str, default=None, help="Extra scene context.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tokenize and report context length without loading model weights.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    repo_id = args.repo_id or "/".join(root.parts[-2:])

    evidence = gather_evidence(
        root=root,
        repo_id=repo_id,
        episode=args.episode,
        timesteps=args.num_frames,
        max_dim=args.max_image_dim,
        cameras=args.cameras,
    )
    messages = build_messages(evidence, args.context)

    if args.dry_run:
        sizes = {img.size for _, img in evidence.frames}
        print(f"[dry run] {len(evidence.frames)} frames at {sizes}, judge model {args.model}")
        print(f'[dry run] instruction: "{evidence.instruction}"')
        print(f"[dry run] stats block:\n{evidence.stats_block}")
        tokens = count_context_tokens(messages, args.model, args.image_token_budget)
        print(f"[dry run] context length: {tokens} input tokens")
        return

    text, usage, seconds = run_judge(
        messages=messages,
        model_id=args.model,
        load_in_4bit=args.load_in_4bit,
        thinking=args.thinking,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        image_token_budget=args.image_token_budget,
    )

    try:
        judgment: Judgment | None = Judgment.from_model_output(text)
    except ValueError as error:
        print(f"warning: {error}", file=sys.stderr)
        judgment = None
    report(evidence, judgment, text, usage, seconds, args.model, as_json=args.json)


if __name__ == "__main__":
    main()
