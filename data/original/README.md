# Test sets as first generated, and what changed

`test_v2.tsv` / `test_v3.tsv` in `data/` are the sets used for every reported number. They differ from the files here by
the changes in `../test_changes.json` (32 items):

- test v2: 3 items removed — two sentences that also appear, almost verbatim, as worked examples in the editor's task
  prompt (a leak we found in review; neither edit fixes the DI one anyway), and one ambiguous item.
- test v3: a blind re-classification of all 300 items by Claude Opus 5.5 (`relabel_v3.json`, prompt in the internal
  `relabel.py`) disagreed with 30 labels; after reading each one we relabelled 21 (mostly "DD" items that actually
  describe a situation still going on, e.g. *The emperor had ruled for thirty-one years when he was deposed*) and dropped
  8 ambiguous ones. The same check on test v2 found 1 ambiguous item.

Translations were not regenerated: the same outputs were re-scored on the corrected labels.
