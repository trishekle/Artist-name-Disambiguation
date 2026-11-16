"""Score the model against your hand labels: precision, recall, confusion.

Reads labels/labels_<group>.csv (your verdicts) and labels/answer_key_<group>.csv
(the tool's flags), joins on track_id, and reports how well the tool's
"review candidate" flag predicts true misattribution.

Convention:
  * Positive class = "misattributed" (the thing we want to catch).
  * Tool prediction = tool_flagged_candidate (True if the tool flagged it).
  * Ground truth    = your_verdict == "misattributed".
  * Rows you marked "unsure" or left blank are excluded from precision/recall
    and reported separately, so they can't silently distort the numbers.

Usage: python tools/evaluate_labels.py [GROUP]
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LABELS_DIR = Path("labels")
VALID_VERDICTS = {"correct", "misattributed", "unsure"}


def score_group(group, labels_dir=LABELS_DIR):
    """Load, join, and score one group's hand labels against the tool's flags.

    Returns a dict describing the result. This is pure computation: it never
    prints and never calls sys.exit, so it can be reused by the cross-group
    aggregator and by tests. ``status`` tells the caller what happened:

      * "missing_files"  - one or both CSVs are absent.
      * "no_verdicts"    - the labels file has no filled-in verdicts.
      * "all_unsure"     - verdicts exist but none are correct/misattributed.
      * "ok"             - at least one row was scored.

    Scoring convention (positive class = "misattributed"):
      * prediction   = answer_key column ``tool_flagged_candidate`` (bool).
      * ground truth = labels column ``your_verdict`` == "misattributed".
      * blank and "unsure" verdicts are excluded and reported via n_unsure.

    The returned dict always carries these keys:
      group, status, n_labeled, n_scored, n_unsure,
      tp, fp, fn, tn, precision, recall, f1, accuracy,
      bad_verdicts (sorted list of unrecognized verdict values),
      merged (the scored DataFrame, or None), n_total (labeled after join).

    Metric values are float('nan') when undefined (e.g. no scored rows).
    """
    labels_dir = Path(labels_dir)
    labels_path = labels_dir / f"labels_{group}.csv"
    key_path = labels_dir / f"answer_key_{group}.csv"

    result = {
        "group": group,
        "status": "missing_files",
        "n_labeled": 0,
        "n_scored": 0,
        "n_unsure": 0,
        "n_total": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "accuracy": float("nan"),
        "bad_verdicts": [],
        "merged": None,
        "labels_path": labels_path,
        "key_path": key_path,
    }

    if not labels_path.exists() or not key_path.exists():
        return result

    labels = pd.read_csv(labels_path, dtype=str)
    key = pd.read_csv(key_path)

    labels["your_verdict"] = labels["your_verdict"].fillna("").str.strip().str.lower()
    result["n_labeled"] = len(labels)
    filled = labels[labels["your_verdict"] != ""]

    if filled.empty:
        result["status"] = "no_verdicts"
        return result

    result["bad_verdicts"] = sorted(set(filled["your_verdict"]) - VALID_VERDICTS)

    merged = filled.merge(key, on="track_id", suffixes=("", "_key"))
    result["n_total"] = len(merged)
    result["n_unsure"] = int((merged["your_verdict"] == "unsure").sum())

    scored = merged[merged["your_verdict"].isin(["correct", "misattributed"])].copy()

    if scored.empty:
        result["status"] = "all_unsure"
        return result

    scored["truth_pos"] = scored["your_verdict"] == "misattributed"
    scored["pred_pos"] = scored["tool_flagged_candidate"].astype(bool)

    tp = int((scored["pred_pos"] & scored["truth_pos"]).sum())
    fp = int((scored["pred_pos"] & ~scored["truth_pos"]).sum())
    fn = int((~scored["pred_pos"] & scored["truth_pos"]).sum())
    tn = int((~scored["pred_pos"] & ~scored["truth_pos"]).sum())

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall)
          else float("nan"))
    accuracy = (tp + tn) / len(scored)

    result.update({
        "status": "ok",
        "n_scored": len(scored),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "merged": scored,
    })
    return result


def pct(x):
    return "n/a" if x != x else f"{x:.1%}"


def main(argv=None):
    """Original CLI: score one group and print the human-readable report."""
    argv = list(sys.argv[1:] if argv is None else argv)
    group = argv[0] if argv else "WITCH"

    res = score_group(group)

    if res["status"] == "missing_files":
        print(f"Missing {res['labels_path']} or {res['key_path']}. "
              f"Run tools/make_labels.py {group} first.")
        return 1

    if res["status"] == "no_verdicts":
        print(f"No verdicts filled in yet in {res['labels_path']}.")
        print("Fill the 'your_verdict' column with: correct | misattributed | unsure")
        return 0

    if res["bad_verdicts"]:
        print(f"WARNING: unrecognized verdict values (ignored): {res['bad_verdicts']}")
        print("Use exactly: correct | misattributed | unsure\n")

    if res["status"] == "all_unsure":
        print(f"Labeled {res['n_total']} tracks, but all are 'unsure' - "
              "nothing to score yet.")
        return 0

    scored = res["merged"]
    n_total = res["n_total"]
    n_unsure = res["n_unsure"]
    tp, fp, fn, tn = res["tp"], res["fp"], res["fn"], res["tn"]

    print(f"=== Evaluation: {group} ===\n")
    print(f"Labeled: {n_total}   (scored: {len(scored)}, unsure/excluded: {n_unsure})")
    print("\nConfusion matrix (positive = misattributed):")
    print(f"                      truth: misattributed   truth: correct")
    print(f"  tool flagged              TP = {tp:<3}            FP = {fp:<3}")
    print(f"  tool said correct         FN = {fn:<3}            TN = {tn:<3}")
    print("\nMetrics:")
    print(f"  Precision (of flags, how many were real): {pct(res['precision'])}")
    print(f"  Recall    (of real errors, how many caught): {pct(res['recall'])}")
    print(f"  F1:        {pct(res['f1'])}")
    print(f"  Accuracy:  {pct(res['accuracy'])}")

    disagreements = scored[scored["pred_pos"] != scored["truth_pos"]]
    if not disagreements.empty:
        print("\nWhere tool and truth disagree (worth a look):")
        show = disagreements[["track_name", "your_verdict", "tool_flagged_candidate", "gap"]]
        print(show.to_string(index=False))

    if n_unsure:
        print(f"\nNote: {n_unsure} track(s) marked 'unsure' were excluded. "
              "Resolve them to tighten the estimate.")
    if len(scored) < 10:
        print("\nCaveat: very small sample - treat these numbers as directional, not final.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
