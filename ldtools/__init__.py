"""Tools for migrating LeRobot community datasets to v3.0 and repairing them.

Modules (run as ``python -m ldtools.<module>``):

- ``convert_collection``: census + v2.0/v2.1 -> v3.0 migration of a whole
  collection tree (``<root>/<user>/<dataset>``), idempotent and parallel.
- ``backfill_quantile_stats``: exact corpus action/state quantiles into
  ``meta/stats.json`` (lerobot's native episode-averaged quantiles are
  wrong for corpus use).
- ``dataset_card``: generate the README/dataset card for a converted
  collection from its conversion manifest and source/output censuses.

The LLM episode judges that used to live here moved to the training repo
(``bijou/judge`` in flow-matching), next to the training pipeline that
consumes their verdicts.

Heavy imports are deliberately kept out of this package root.
"""

__all__: list[str] = []
