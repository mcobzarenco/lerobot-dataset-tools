"""LLM-as-judge quality assessment for LeRobot v3.0 dataset episodes.

Samples representative frames from an episode, computes trajectory summary
statistics, and asks Claude for a structured quality verdict — useful for
curating teleoperated demonstrations before training.

Usage:
    uv run python -m fmatch.judge_episode \
        --root /home/marius/w/community_dataset_v1_v3/ZGGZZG/so100_drop0 \
        --episode 3

    # inspect the payload without calling the API
    uv run python -m fmatch.judge_episode --root ... --episode 3 --dry-run

Requires ANTHROPIC_API_KEY in the environment (the SDK reads it directly).
"""

import argparse
import base64
import io
import json
import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from anthropic import Anthropic
from anthropic.types import ImageBlockParam, TextBlockParam
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

# CLI defaults. Named so ldtools.judge_sweep (which drives this judge over
# whole collections) shares the exact same knobs instead of re-hardcoding
# them; help strings render the live values via argparse's %(default)s.
DEFAULT_MODEL = "claude-opus-4-8"  # $5/$25 per MTok (2026-07)
DEFAULT_NUM_FRAMES = 10  # sampled timesteps per episode
DEFAULT_MAX_IMAGE_DIM = 512  # px, longer side after downscaling
DEFAULT_JPEG_QUALITY = 90
DEFAULT_MAX_TOKENS = 1500  # response budget

# Bump when SYSTEM_PROMPT or the verdict schema changes; recorded alongside
# every stored verdict so sweeps/calibration can filter comparable records.
PROMPT_VERSION = 2

SYSTEM_PROMPT = """\
You are a robotics dataset curator reviewing teleoperated demonstration
episodes for imitation-learning training quality. You will see frames sampled
chronologically from one episode (all cameras at each sampled timestep), the
natural-language task instruction, and summary statistics of the recorded
trajectory.

Judge only what is observable. Typical issues worth flagging: the task is not
completed or not visible; operator fumbling (retries, dropped objects,
hesitation); long idle stretches; jerky or erratic motion; occluded or badly
framed cameras; inconsistent scene setup; frames where the robot
is outside the camera view. Remember you only see sampled frames — phrase
temporal claims accordingly (statistics cover the full episode).

Judge the DEMONSTRATION, not the label: a competent demonstration with a
wrong, empty or placeholder instruction is salvageable by relabeling — do
not discard for the instruction alone; reflect label problems in
`instruction_quality` (and rate `task_completion_visible` against the
stated instruction, "unclear" when it is meaningless).

Classify every camera by what you SEE across the sampled frames — the
recorded camera names ("image", "image2", ...) are arbitrary and their
ordering is inconsistent between datasets:
- "wrist": mounted on a robot arm, the viewpoint moves with it; gripper
  jaws/fingers typically protrude from a fixed spot at the frame edge while
  the background shifts between frames.
- "top": fixed camera looking roughly straight down at the workspace.
- "front": fixed external camera facing the workspace/robot roughly
  head-on and horizontally.
- "side": fixed external camera viewing the workspace from the side or a
  three-quarter angle.
- "unknown": genuinely undeterminable from the frames.

For `suggested_instructions`, write 2-3 short imperative commands that
describe what is actually demonstrated (grounded in the visible objects and
outcome, varied phrasing, usable directly as training labels). If the
stated instruction is accurate, include a cleaned-up version of it.

Respond with a single JSON object, no markdown fences, matching:
{
  "overall_score": <int 1-10>,
  "verdict": "keep" | "review" | "discard",
  "task_completion_visible": "yes" | "partial" | "no" | "unclear",
  "scores": {
    "visual_quality": <int 1-10>,
    "smoothness": <int 1-10>,
    "efficiency": <int 1-10>,
    "camera_framing": <int 1-10>
  },
  "instruction_quality": "good" | "vague" | "mismatched" | "placeholder",
  "observed_task": "<1-2 sentences: what actually happens>",
  "suggested_instructions": ["<imperative instruction>", ...],
  "camera_kinds": {"<camera name>": "wrist" | "top" | "front" | "side" | "unknown", ...},
  "issues": [<short strings>],
  "summary": "<2-4 sentences>"
}
`camera_kinds` must contain exactly the camera names listed in the message.
"""


class Verdict(StrEnum):
    """Curation decision for an episode."""

    KEEP = "keep"
    REVIEW = "review"
    DISCARD = "discard"


class TaskCompletion(StrEnum):
    """Whether task completion is observable in the sampled frames."""

    YES = "yes"
    PARTIAL = "partial"
    NO = "no"
    UNCLEAR = "unclear"


class InstructionQuality(StrEnum):
    """How well the stored task string describes the demonstration.

    Community task strings are frequently junk ("test1", "Test Boulon") on
    top of perfectly usable demonstrations; this axis is deliberately
    separate from the quality verdict so good demos with bad labels can be
    relabeled instead of discarded.
    """

    GOOD = "good"  # specific and matches what the episode shows
    VAGUE = "vague"  # generic but compatible ("pick up the object")
    MISMATCHED = "mismatched"  # describes something visibly different
    PLACEHOLDER = "placeholder"  # empty/meaningless ("test", "task1", ...)


class CameraKind(StrEnum):
    """Visually judged camera mount/viewpoint category.

    The converted community collections use anonymized camera names
    ("image", "image2", ...) whose ordering is inconsistent across datasets
    (measured: 99.9% of 1,242 datasets), so viewpoint semantics can only
    come from looking at the frames. UNKNOWN is the honest fallback and is
    also useful as a train-time dropout target for camera annotations.
    """

    WRIST = "wrist"
    TOP = "top"
    FRONT = "front"
    SIDE = "side"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Scores:
    """Per-aspect quality sub-scores on a 1-10 scale."""

    visual_quality: int
    smoothness: int
    efficiency: int
    camera_framing: int


def _score_1_10(data: dict, field: str) -> int:
    """1-10 integer score, strictly — what a jsonschema 'integer' + bounds
    would check, without a second schema document to keep in sync.

    Bare int() coercion lets true -> 1, "7" -> 7 and 7.9 -> 7 slide through
    silently; a silently-wrong score poisons downstream aggregation, which
    is worse than a loud parse failure (the sweep records those for
    --retry-failed). Integer-valued floats (7.0) are accepted — JSON Schema
    itself treats them as integers.
    """
    value = data[field]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field} must be an integer, got {value!r}")
        value = int(value)
    if not 1 <= value <= 10:
        raise ValueError(f"{field} must be in 1..10, got {value}")
    return value


@dataclass(frozen=True)
class EpisodeJudgment:
    """Structured verdict returned by the judge model.

    Mirrors the JSON schema demanded in SYSTEM_PROMPT (PROMPT_VERSION 2);
    `from_dict` enforces it exhaustively (required fields, enum membership,
    integer 1-10 scores, non-empty relabels) — the parser IS the schema,
    there is deliberately no separate jsonschema document to drift out of
    sync. Use `from_response_text` for raw model output (tolerates
    surrounding prose or markdown fences) and `to_json`/`from_json` for
    strict round-trips. A verdict that violates the schema is a parse
    failure to be retried, not silently backfilled or clamped.
    """

    overall_score: int
    verdict: Verdict
    task_completion_visible: TaskCompletion
    scores: Scores
    instruction_quality: InstructionQuality
    observed_task: str
    suggested_instructions: tuple[str, ...]
    camera_kinds: dict[str, CameraKind]
    issues: tuple[str, ...]
    summary: str

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeJudgment":
        try:
            scores = data["scores"]
            if not isinstance(scores, dict):
                raise ValueError(f"scores must be an object, got {type(scores).__name__}")
            camera_kinds = data["camera_kinds"]
            if not isinstance(camera_kinds, dict) or not camera_kinds:
                raise ValueError("camera_kinds must be a non-empty object")
            observed_task = str(data["observed_task"]).strip()
            if not observed_task:
                raise ValueError("observed_task must be a non-empty string")
            suggested = data["suggested_instructions"]
            if not isinstance(suggested, list) or not suggested:
                raise ValueError("suggested_instructions must be a non-empty array")
            instructions = tuple(str(entry).strip() for entry in suggested)
            if not all(instructions):
                raise ValueError(f"suggested_instructions contains empty entries: {suggested!r}")
            return cls(
                overall_score=_score_1_10(data, "overall_score"),
                verdict=Verdict(data["verdict"]),
                task_completion_visible=TaskCompletion(data["task_completion_visible"]),
                scores=Scores(
                    visual_quality=_score_1_10(scores, "visual_quality"),
                    smoothness=_score_1_10(scores, "smoothness"),
                    efficiency=_score_1_10(scores, "efficiency"),
                    camera_framing=_score_1_10(scores, "camera_framing"),
                ),
                instruction_quality=InstructionQuality(data["instruction_quality"]),
                observed_task=observed_task,
                suggested_instructions=instructions,
                camera_kinds={str(name): CameraKind(kind) for name, kind in camera_kinds.items()},
                issues=tuple(str(issue) for issue in data.get("issues", [])),
                summary=str(data.get("summary", "")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed judge verdict: {error}") from error

    def check_cameras(self, expected: list[str]) -> None:
        """Raise if camera_kinds does not cover exactly the shown cameras.

        The map is only usable downstream (train-time camera annotations)
        when keyed by the dataset's actual camera names; a judge that
        renamed or dropped a camera produced an unusable verdict.
        """
        got, want = set(self.camera_kinds), set(expected)
        if got != want:
            raise ValueError(f"camera_kinds keys {sorted(got)} != cameras shown {sorted(want)}")

    @classmethod
    def from_json(cls, text: str) -> "EpisodeJudgment":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_response_text(cls, text: str) -> "EpisodeJudgment":
        """Parse from raw model output by extracting the outermost JSON object."""
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in response")
        return cls.from_json(text[start : end + 1])

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
            "instruction_quality": self.instruction_quality.value,
            "observed_task": self.observed_task,
            "suggested_instructions": list(self.suggested_instructions),
            "camera_kinds": {name: kind.value for name, kind in self.camera_kinds.items()},
            "issues": list(self.issues),
            "summary": self.summary,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def short_camera(key: str) -> str:
    """Dataset camera names without the feature-key boilerplate.

    "observation.images.image2" -> "image2". These short names are what the
    judge sees and what `camera_kinds` is keyed by — they match the names
    other tooling derives from the feature keys.
    """
    return key.removeprefix("observation.images.")


@dataclass(frozen=True)
class EpisodeSummary:
    """Everything extracted from the dataset for one episode."""

    repo_id: str
    episode: int
    task: str
    fps: float
    num_frames: int
    duration_s: float
    motor_names: list[str]
    camera_names: list[str]  # short names, e.g. "image", "image2"
    stats_text: str
    # (timestep label, short camera name, base64 image) in chronological order
    frames: list[tuple[str, str, str]]
    media_type: Literal["image/jpeg", "image/png"]


def tensor_to_image_b64(
    chw: torch.Tensor, max_dim: int, image_format: str, jpeg_quality: int
) -> str:
    """float32 CHW [0,1] image tensor -> base64-encoded JPEG or PNG.

    PNG is lossless w.r.t. the decoded video frame (the dataset's AV1
    compression is already baked in either way); JPEG adds one more lossy
    generation but is ~10x smaller on the wire. Anthropic token cost is
    identical (it depends on pixel dimensions only).
    """
    array = (chw.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(array)
    image.thumbnail((max_dim, max_dim))
    buffer = io.BytesIO()
    if image_format == "png":
        image.save(buffer, format="PNG")
    else:
        image.save(buffer, format="JPEG", quality=jpeg_quality)
    return base64.standard_b64encode(buffer.getvalue()).decode()


def format_stats(
    action: np.ndarray,
    state: np.ndarray,
    motor_names: list[str],
    fps: float,
) -> str:
    """Compact per-motor and whole-trajectory statistics for the prompt."""
    delta = np.abs(np.diff(action, axis=0))
    tracking_error = np.abs(action - state).mean(axis=0)

    header = f"{'motor':<22}{'min':>9}{'max':>9}{'mean':>9}{'std':>8}{'path':>9}{'max|d|':>8}{'|a-s|':>8}"
    lines = [header]
    for i, name in enumerate(motor_names):
        lines.append(
            f"{name:<22}"
            f"{action[:, i].min():>9.1f}"
            f"{action[:, i].max():>9.1f}"
            f"{action[:, i].mean():>9.1f}"
            f"{action[:, i].std():>8.1f}"
            f"{delta[:, i].sum():>9.1f}"
            f"{delta[:, i].max():>8.1f}"
            f"{tracking_error[i]:>8.1f}"
        )

    # Fraction of steps where no motor target moves more than 1% of its
    # episode range: a proxy for idle time.
    ranges = action.max(axis=0) - action.min(axis=0)
    idle_threshold = np.maximum(ranges * 0.01, 1e-6)
    idle_fraction = float((delta < idle_threshold).all(axis=1).mean())

    lines += [
        "",
        "columns: action min/max/mean/std over the episode; path = total travelled "
        "distance sum(|delta|); max|d| = largest single-step jump (jerkiness proxy, "
        f"steps are {1000 / fps:.0f} ms apart); |a-s| = mean |action - state| "
        "(commanded target vs. achieved position tracking error).",
        f"idle steps (all motors move < 1% of their range): {idle_fraction:.0%}",
        "Units are dataset-dependent (raw joint values, often degrees or a "
        "normalized [-100, 100] range).",
    ]
    return "\n".join(lines)


def load_episode_summary(
    root: Path,
    repo_id: str,
    episode: int,
    num_timesteps: int,
    max_image_dim: int,
    cameras: list[str] | None,
    image_format: str = "jpeg",
    jpeg_quality: int = 90,
) -> EpisodeSummary:
    dataset = LeRobotDataset(repo_id, root=root)
    if not 0 <= episode < dataset.num_episodes:
        raise SystemExit(f"episode {episode} out of range (dataset has {dataset.num_episodes})")

    row = dataset.meta.episodes[episode]
    start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    num_frames = stop - start
    fps = float(dataset.fps)

    tasks = row.get("tasks") or ["<no task recorded>"]
    task = "; ".join(tasks) if isinstance(tasks, list) else str(tasks)

    # Trajectory columns straight from parquet (no video decoding).
    table = dataset.hf_dataset[start:stop]
    action = np.asarray(table["action"], dtype=np.float32)
    state = np.asarray(table["observation.state"], dtype=np.float32)

    feature = dataset.meta.features["action"]
    motor_names = list(feature.get("names") or [f"motor_{i}" for i in range(action.shape[1])])

    camera_keys = list(dataset.meta.camera_keys)
    if cameras:
        # Accept either full feature keys or short names in the filter.
        wanted = {short_camera(c) for c in cameras}
        known = {short_camera(k) for k in camera_keys}
        unknown = wanted - known
        if unknown:
            raise SystemExit(f"unknown cameras {sorted(unknown)}; dataset has {sorted(known)}")
        camera_keys = [k for k in camera_keys if short_camera(k) in wanted]

    # Evenly spaced timesteps, always including the first and last frame.
    picks = np.unique(np.linspace(0, num_frames - 1, num_timesteps).round().astype(int))
    frames: list[tuple[str, str, str]] = []
    for local_idx in picks:
        item = dataset[start + int(local_idx)]  # decodes video for this frame only
        label = f"frame {local_idx + 1}/{num_frames} (t={local_idx / fps:.1f}s)"
        for camera in camera_keys:
            frames.append(
                (
                    label,
                    short_camera(camera),
                    tensor_to_image_b64(item[camera], max_image_dim, image_format, jpeg_quality),
                )
            )

    return EpisodeSummary(
        repo_id=repo_id,
        episode=episode,
        task=task,
        fps=fps,
        num_frames=num_frames,
        duration_s=num_frames / fps,
        motor_names=motor_names,
        camera_names=[short_camera(k) for k in camera_keys],
        stats_text=format_stats(action, state, motor_names, fps),
        frames=frames,
        media_type="image/png" if image_format == "png" else "image/jpeg",
    )


def build_user_content(
    summary: EpisodeSummary,
    context: str | None = None,
) -> list[TextBlockParam | ImageBlockParam]:
    """Interleaved text/image blocks for the Anthropic messages API."""
    intro = (
        f"Dataset: {summary.repo_id}, episode {summary.episode}\n"
        f'Task instruction: "{summary.task}"\n'
        f"Length: {summary.num_frames} frames = {summary.duration_s:.1f}s @ {summary.fps:.0f} fps\n"
        f"Cameras: {', '.join(summary.camera_names)}\n"
        f"Sampled timesteps: {len(summary.frames) // max(len(summary.camera_names), 1)} "
        f"(each shown for every camera, chronological order)"
    )
    content: list[TextBlockParam | ImageBlockParam] = [{"type": "text", "text": intro}]
    if context:
        content.append(
            {
                "type": "text",
                "text": f"Additional context from the dataset owner: {context}",
            }
        )
    for label, camera, b64 in summary.frames:
        content.append({"type": "text", "text": f"{label} — camera '{camera}'"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": summary.media_type,
                    "data": b64,
                },
            }
        )
    camera_list = ", ".join(f'"{name}"' for name in summary.camera_names)
    content.append(
        {
            "type": "text",
            "text": "Full-episode trajectory statistics:\n```\n"
            + summary.stats_text
            + "\n```\nNow give your quality assessment as the specified JSON object. "
            f"`camera_kinds` must have exactly these keys: {camera_list}.",
        }
    )
    return content


def print_report(
    summary: EpisodeSummary,
    judgment: EpisodeJudgment | None,
    raw: str,
    as_json: bool,
    usage: dict[str, int] | None = None,
) -> None:
    if as_json:
        payload = {
            "dataset": summary.repo_id,
            "episode": summary.episode,
            "task": summary.task,
            "num_frames": summary.num_frames,
            "duration_s": round(summary.duration_s, 2),
            "judge": judgment.to_dict() if judgment is not None else {"raw_response": raw},
        }
        if usage is not None:
            payload["usage"] = usage
        print(json.dumps(payload, indent=2))
        return

    print(f"=== {summary.repo_id} — episode {summary.episode} ===")
    print(f'task     : "{summary.task}"')
    print(
        f"length   : {summary.num_frames} frames ({summary.duration_s:.1f}s @ {summary.fps:.0f} fps)"
    )
    print(f"cameras  : {', '.join(summary.camera_names)}")
    if usage is not None:
        print(f"tokens   : {usage['input_tokens']} in / {usage['output_tokens']} out")
    print()
    if judgment is None:
        print("could not parse JSON verdict; raw response:")
        print(raw)
        return
    print(f"overall  : {judgment.overall_score}/10  ->  {judgment.verdict.value}")
    print(f"task done: {judgment.task_completion_visible.value}")
    scores = judgment.scores
    print(
        "scores   : "
        f"visual_quality={scores.visual_quality}  smoothness={scores.smoothness}  "
        f"efficiency={scores.efficiency}  camera_framing={scores.camera_framing}"
    )
    print(f"instr    : {judgment.instruction_quality.value}")
    print(f'observed : "{judgment.observed_task}"')
    print(
        "cameras  : "
        + "  ".join(f"{name}={kind.value}" for name, kind in sorted(judgment.camera_kinds.items()))
    )
    for instruction in judgment.suggested_instructions:
        print(f'  suggest: "{instruction}"')
    print("issues   : " + ("none noted" if not judgment.issues else ""))
    for issue in judgment.issues:
        print(f"  - {issue}")
    if judgment.summary:
        print(f"summary  : {judgment.summary}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge the quality of a LeRobot v3.0 episode with Claude."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset directory containing meta/, data/, videos/ (v3.0 format).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Dataset repo id (default: the last two path components of --root).",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to judge (default: %(default)s).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Number of timesteps to sample, each shown for every camera (default: %(default)s).",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="*",
        default=None,
        help="Camera names to include, short ('wrist') or full "
        "('observation.images.wrist') (default: all cameras).",
    )
    parser.add_argument(
        "--max-image-dim",
        type=int,
        default=DEFAULT_MAX_IMAGE_DIM,
        help="Frames are downscaled so the longer side is at most this many pixels "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--image-format",
        choices=["jpeg", "png"],
        default="jpeg",
        help="Encoding for uploaded frames. Token cost is identical (it depends on "
        "pixel dimensions); png is lossless w.r.t. the decoded video frame but "
        "~10x larger on the wire (default: %(default)s).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality, ignored for --image-format=png (default: %(default)s).",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Extra context for the judge, e.g. clarifying an ambiguous task "
        "instruction or describing the scene setup (default: none).",
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
        help="Maximum response tokens for the verdict (default: %(default)s).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report instead of the text report.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the model's verbatim response text instead of any report "
        "(parse problems still warn on stderr).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and describe the payload without requesting a completion. "
        "When ANTHROPIC_API_KEY is set, the exact context length is reported "
        "via the free token-counting endpoint.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    repo_id = args.repo_id or "/".join(root.parts[-2:])

    summary = load_episode_summary(
        root=root,
        repo_id=repo_id,
        episode=args.episode,
        num_timesteps=args.num_frames,
        max_image_dim=args.max_image_dim,
        cameras=args.cameras,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
    )

    content = build_user_content(summary, args.context)
    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if args.dry_run:
        payload_kb = sum(len(b64) for _, _, b64 in summary.frames) * 3 / 4 / 1024
        print(
            f"[dry run] would send {len(summary.frames)} images "
            f"(~{payload_kb:.0f} KB {summary.media_type.removeprefix('image/').upper()})"
        )
        print(f'[dry run] task: "{summary.task}"')
        print(f"[dry run] stats block:\n{summary.stats_text}")
        if have_key:
            count = Anthropic().messages.count_tokens(
                model=args.model,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            print(f"[dry run] context length: {count.input_tokens} input tokens ({args.model})")
        else:
            print("[dry run] set ANTHROPIC_API_KEY to report the exact context length")
        return

    if not have_key:
        print(
            "error: ANTHROPIC_API_KEY is not set. Export it or use --dry-run.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    client = Anthropic()
    # Deliberately no sampling controls: opus 4.7+ rejects `temperature`
    # with a 400, and the API reference never promised determinism even at
    # temperature=0 — API verdicts are inherently non-reproducible. The
    # local Gemma judge (greedy decode) is the path to reproducible
    # verdicts if that ever becomes load-bearing.
    response = client.messages.create(
        model=args.model,
        max_tokens=args.max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    raw = "".join(block.text for block in response.content if block.type == "text")
    try:
        judgment: EpisodeJudgment | None = EpisodeJudgment.from_response_text(raw)
        judgment.check_cameras(summary.camera_names)
    except ValueError as error:
        print(f"warning: {error}", file=sys.stderr)
        judgment = None
    if args.raw:
        print(raw)
        return
    print_report(summary, judgment, raw, as_json=args.json, usage=usage)


if __name__ == "__main__":
    main()
