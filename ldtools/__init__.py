"""Tools for migrating LeRobot community datasets to v3.0 and curating them.

Modules (run as ``python -m ldtools.<module>``):

- ``convert_collection``: census + v2.0/v2.1 -> v3.0 migration of a whole
  collection tree (``<root>/<user>/<dataset>``), idempotent and parallel.
- ``dataset_card``: generate the README/dataset card for a converted
  collection from its conversion manifest and source/output censuses.
- ``judge_episode``: episode-quality judgment via the Anthropic API
  (schema v2: + instruction quality, suggested relabels, camera kinds).
- ``judge_episode_gemma``: episode-quality judgment via a local Gemma 4
  (schema v1).
- ``judge_sweep``: resumable parallel judging across whole collections,
  one JSONL verdict per episode, with loud skip/failure records.

Heavy imports are deliberately kept out of this package root.
"""

__all__: list[str] = []
