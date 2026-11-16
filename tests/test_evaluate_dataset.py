"""Offline unit test for the shared scoring core (FEAT-002).

Exercises tools.evaluate_labels.score_group on tiny synthetic labels +
answer_key CSVs written to a tmp_path labels/ dir, following the tmp_path
pattern in tests/test_app_core.py. Verifies the precision/recall/F1 math and
that 'unsure'/blank verdicts are excluded (not counted right or wrong).
"""

from __future__ import annotations

import pandas as pd

from tools import evaluate_dataset
from tools.evaluate_labels import score_group


def _write_synthetic(labels_dir):
    """Four scorable rows (one per confusion cell) + one unsure + one blank."""
    labels = pd.DataFrame([
        {"track_id": "t1", "your_verdict": "misattributed"},  # flagged   -> TP
        {"track_id": "t2", "your_verdict": "correct"},        # flagged   -> FP
        {"track_id": "t3", "your_verdict": "misattributed"},  # unflagged -> FN
        {"track_id": "t4", "your_verdict": "correct"},        # unflagged -> TN
        {"track_id": "t5", "your_verdict": "unsure"},         # excluded
        {"track_id": "t6", "your_verdict": ""},               # blank -> excluded
    ])
    key = pd.DataFrame([
        {"track_id": "t1", "track_name": "One", "tool_flagged_candidate": True, "gap": 0.9},
        {"track_id": "t2", "track_name": "Two", "tool_flagged_candidate": True, "gap": 0.8},
        {"track_id": "t3", "track_name": "Three", "tool_flagged_candidate": False, "gap": 0.1},
        {"track_id": "t4", "track_name": "Four", "tool_flagged_candidate": False, "gap": 0.0},
        {"track_id": "t5", "track_name": "Five", "tool_flagged_candidate": True, "gap": 0.5},
        {"track_id": "t6", "track_name": "Six", "tool_flagged_candidate": False, "gap": 0.2},
    ])
    labels.to_csv(labels_dir / "labels_TEST.csv", index=False)
    key.to_csv(labels_dir / "answer_key_TEST.csv", index=False)


def test_score_group_math_and_exclusion(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    _write_synthetic(labels_dir)

    res = score_group("TEST", labels_dir=labels_dir)

    assert res["status"] == "ok"
    # Confusion cells: one row per quadrant.
    assert (res["tp"], res["fp"], res["fn"], res["tn"]) == (1, 1, 1, 1)

    # Only the four correct/misattributed rows are scored; unsure counted apart.
    assert res["n_scored"] == 4
    assert res["n_unsure"] == 1
    # The blank row is dropped entirely (not scored, not counted as unsure).
    assert res["n_total"] == 5

    # Hand-computed: precision = 1/2, recall = 1/2, f1 = 1/2, accuracy = 2/4.
    assert res["precision"] == 0.5
    assert res["recall"] == 0.5
    assert res["f1"] == 0.5
    assert res["accuracy"] == 0.5


def test_score_group_missing_files(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    res = score_group("NOPE", labels_dir=labels_dir)
    assert res["status"] == "missing_files"
    assert res["n_scored"] == 0


def test_score_group_all_unsure(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    pd.DataFrame([
        {"track_id": "t1", "your_verdict": "unsure"},
        {"track_id": "t2", "your_verdict": ""},
    ]).to_csv(labels_dir / "labels_U.csv", index=False)
    pd.DataFrame([
        {"track_id": "t1", "track_name": "One", "tool_flagged_candidate": True, "gap": 0.1},
        {"track_id": "t2", "track_name": "Two", "tool_flagged_candidate": False, "gap": 0.0},
    ]).to_csv(labels_dir / "answer_key_U.csv", index=False)

    res = score_group("U", labels_dir=labels_dir)
    assert res["status"] == "all_unsure"
    assert res["n_scored"] == 0
    assert res["n_unsure"] == 1


def test_discover_groups_requires_both_files(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    # Group A has both files; group B only has labels -> excluded.
    (labels_dir / "labels_A.csv").write_text("track_id,your_verdict\n", encoding="utf-8")
    (labels_dir / "answer_key_A.csv").write_text(
        "track_id,track_name,tool_flagged_candidate,gap\n", encoding="utf-8")
    (labels_dir / "labels_B.csv").write_text("track_id,your_verdict\n", encoding="utf-8")

    assert evaluate_dataset.discover_groups(labels_dir) == ["A"]


def test_pooled_metrics_sums_confusion_cells():
    results = [
        {"tp": 2, "fp": 1, "fn": 1, "tn": 4},
        {"tp": 3, "fp": 0, "fn": 2, "tn": 1},
    ]
    pooled = evaluate_dataset.pooled_metrics(results)
    assert (pooled["tp"], pooled["fp"], pooled["fn"], pooled["tn"]) == (5, 1, 3, 5)
    # precision = 5/6, recall = 5/8.
    assert pooled["precision"] == 5 / 6
    assert pooled["recall"] == 5 / 8
