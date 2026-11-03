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

GROUP = sys.argv[1] if len(sys.argv) > 1 else "WITCH"
labels_path = Path("labels") / f"labels_{GROUP}.csv"
key_path = Path("labels") / f"answer_key_{GROUP}.csv"

if not labels_path.exists() or not key_path.exists():
    print(f"Missing {labels_path} or {key_path}. Run tools/make_labels.py {GROUP} first.")
    sys.exit(1)

labels = pd.read_csv(labels_path, dtype=str)
key = pd.read_csv(key_path)

labels["your_verdict"] = labels["your_verdict"].fillna("").str.strip().str.lower()
filled = labels[labels["your_verdict"] != ""]

if filled.empty:
    print(f"No verdicts filled in yet in {labels_path}.")
    print("Fill the 'your_verdict' column with: correct | misattributed | unsure")
    sys.exit(0)

valid = {"correct", "misattributed", "unsure"}
bad = sorted(set(filled["your_verdict"]) - valid)
if bad:
    print(f"WARNING: unrecognized verdict values (ignored): {bad}")
    print("Use exactly: correct | misattributed | unsure\n")

merged = filled.merge(key, on="track_id", suffixes=("", "_key"))

n_total = len(merged)
n_unsure = int((merged["your_verdict"] == "unsure").sum())
scored = merged[merged["your_verdict"].isin(["correct", "misattributed"])].copy()

if scored.empty:
    print(f"Labeled {n_total} tracks, but all are 'unsure' - nothing to score yet.")
    sys.exit(0)

scored["truth_pos"] = scored["your_verdict"] == "misattributed"
scored["pred_pos"] = scored["tool_flagged_candidate"].astype(bool)

tp = int((scored["pred_pos"] & scored["truth_pos"]).sum())
fp = int((scored["pred_pos"] & ~scored["truth_pos"]).sum())
fn = int((~scored["pred_pos"] & scored["truth_pos"]).sum())
tn = int((~scored["pred_pos"] & ~scored["truth_pos"]).sum())

precision = tp / (tp + fp) if (tp + fp) else float("nan")
recall = tp / (tp + fn) if (tp + fn) else float("nan")
f1 = (2 * precision * recall / (precision + recall)
      if precision == precision and recall == recall and (precision + recall) else float("nan"))
accuracy = (tp + tn) / len(scored)


def pct(x):
    return "n/a" if x != x else f"{x:.1%}"


print(f"=== Evaluation: {GROUP} ===\n")
print(f"Labeled: {n_total}   (scored: {len(scored)}, unsure/excluded: {n_unsure})")
print("\nConfusion matrix (positive = misattributed):")
print(f"                      truth: misattributed   truth: correct")
print(f"  tool flagged              TP = {tp:<3}            FP = {fp:<3}")
print(f"  tool said correct         FN = {fn:<3}            TN = {tn:<3}")
print("\nMetrics:")
print(f"  Precision (of flags, how many were real): {pct(precision)}")
print(f"  Recall    (of real errors, how many caught): {pct(recall)}")
print(f"  F1:        {pct(f1)}")
print(f"  Accuracy:  {pct(accuracy)}")

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
