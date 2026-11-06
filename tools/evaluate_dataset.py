"""Aggregate the tool's misattribution scores across every labeled group.

This pools the per-group confusion matrices produced by
``tools.evaluate_labels.score_group`` into one dataset-wide estimate, plus a
per-group breakdown. It discovers every group that has BOTH
labels/labels_<group>.csv AND labels/answer_key_<group>.csv.

Honesty caveats:
  * Only tracks that MusicBrainz could resolve (and that you therefore labeled)
    are scored. Tracks with no resolvable ground truth are simply absent from
    the labels file; they are NOT counted as right or wrong, so these numbers
    describe the resolvable slice of the catalog, not the whole catalog.
  * Rows marked "unsure" or left blank are excluded from precision/recall and
    reported separately, so uncertainty can't silently inflate the metrics.
  * Pooling sums raw tp/fp/fn/tn across groups, so larger groups weigh more.
    The per-group table is there to expose groups that behave differently.

Usage:
  python tools/evaluate_dataset.py                 # score all discovered groups
  python tools/evaluate_dataset.py WITCH NIRVANA   # score only the named groups
  python tools/evaluate_dataset.py --self-test     # offline sanity check, no data
  python tools/evaluate_dataset.py --threshold 0.06  # real-data FPR at a gap cutoff
"""

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_labels import LABELS_DIR, pct, score_group


def discover_groups(labels_dir=LABELS_DIR):
    """Return sorted group names that have both a labels and answer_key CSV."""
    labels_dir = Path(labels_dir)
    groups = []
    for labels_file in sorted(labels_dir.glob("labels_*.csv")):
        group = labels_file.stem[len("labels_"):]
        if (labels_dir / f"answer_key_{group}.csv").exists():
            groups.append(group)
    return groups


def pooled_metrics(results):
    """Sum confusion counts across scored groups and derive pooled metrics."""
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    tn = sum(r["tn"] for r in results)

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall)
          else float("nan"))
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else float("nan")

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": accuracy, "n_scored": total,
    }


def real_fpr_at_threshold(groups, threshold, labels_dir=LABELS_DIR):
    """False-positive rate on verified-correct tracks if flag = gap > threshold.

    Uses the answer key's ``gap`` (best-fit score minus credited score, >= 0),
    which equals the injection eval's score_gap whenever that is positive, so a
    threshold chosen from tools/injection_eval.py applies here unchanged.
    Returns (per_group list of (group, flagged, n_correct), pooled_flagged, pooled_n).
    """
    labels_dir = Path(labels_dir)
    per_group, total_flagged, total_n = [], 0, 0
    for group in groups:
        labels_path = labels_dir / f"labels_{group}.csv"
        key_path = labels_dir / f"answer_key_{group}.csv"
        if not labels_path.exists() or not key_path.exists():
            continue
        labels = pd.read_csv(labels_path, dtype=str)
        key = pd.read_csv(key_path)
        verdict = labels["your_verdict"].fillna("").str.strip().str.lower()
        correct_ids = set(labels.loc[verdict == "correct", "track_id"])
        clean = key[key["track_id"].astype(str).isin(correct_ids)]
        flagged = int((clean["gap"].astype(float) > threshold).sum())
        per_group.append((group, flagged, len(clean)))
        total_flagged += flagged
        total_n += len(clean)
    return per_group, total_flagged, total_n


def report_threshold(groups, threshold, labels_dir=LABELS_DIR):
    """Print the real-data false-positive rate at a fixed gap threshold."""
    per_group, flagged, n = real_fpr_at_threshold(groups, threshold, labels_dir)
    print(f"\n=== Real-data false-positive rate at gap > {threshold} ===")
    print("(MusicBrainz-verified 'correct' tracks only; compare with the injection")
    print(" eval's FPR at the same threshold to see if it holds on real data.)\n")
    for group, g_flagged, g_n in per_group:
        rate = g_flagged / g_n if g_n else float("nan")
        print(f"  {group:<16} {g_flagged:>3}/{g_n:<4} flagged  ({pct(rate)})")
    if n:
        print(f"  {'POOLED':<16} {flagged:>3}/{n:<4} flagged  ({pct(flagged / n)})")


def report(groups, labels_dir=LABELS_DIR):
    """Score each group, print the per-group table + pooled totals.

    Returns the list of per-group result dicts (for programmatic reuse).
    """
    results = [score_group(g, labels_dir=labels_dir) for g in groups]
    scored = [r for r in results if r["status"] == "ok"]

    print("=== Dataset evaluation (pooled across groups) ===\n")
    print(f"Groups discovered: {len(groups)}   scored: {len(scored)}\n")

    header = f"{'group':<16} {'scored':>6} {'unsure':>6} {'precision':>10} {'recall':>9} {'F1':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r["status"] == "ok":
            print(f"{r['group']:<16} {r['n_scored']:>6} {r['n_unsure']:>6} "
                  f"{pct(r['precision']):>10} {pct(r['recall']):>9} {pct(r['f1']):>8}")
        else:
            print(f"{r['group']:<16} {'n/a':>6} {'n/a':>6} "
                  f"{r['status']:>10} {'':>9} {'':>8}")

    if scored:
        pooled = pooled_metrics(scored)
        print("-" * len(header))
        print(f"{'POOLED':<16} {pooled['n_scored']:>6} {'':>6} "
              f"{pct(pooled['precision']):>10} {pct(pooled['recall']):>9} {pct(pooled['f1']):>8}")
        print("\nPooled confusion matrix (positive = misattributed):")
        print(f"  TP = {pooled['tp']}   FP = {pooled['fp']}   "
              f"FN = {pooled['fn']}   TN = {pooled['tn']}")
        print(f"  Pooled accuracy: {pct(pooled['accuracy'])}")

        negatives = pooled["fp"] + pooled["tn"]
        positives = pooled["tp"] + pooled["fn"]
        if negatives:
            fp_rate = pooled["fp"] / negatives
            print(f"\nFalse-positive rate (clean tracks flagged): "
                  f"{pct(fp_rate)}  ({pooled['fp']}/{negatives})")
        if positives == 0:
            print("\nNOTE: no real misattributions in the labeled data "
                  "(0 'misattributed' verdicts).")
            print("  Recall and F1 are undefined, and precision is meaningless "
                  "(0 true errors to find).")
            print("  Only the false-positive rate above is a valid real-data "
                  "measurement here.")
            print("  Use tools/injection_eval*.py for recall, or add real "
                  "misattributed cases.")
        elif positives < 10:
            print(f"\nCaveat: only {positives} real misattributions; "
                  "recall/precision are very noisy.")
    else:
        print("\nNo scored rows across any group yet - nothing to pool.")

    return results


def self_test():
    """Offline check of score_group on a tiny synthetic labels/answer_key pair.

    Writes a temp labels/ dir, scores it, and asserts the confusion math and
    unsure/blank exclusion. Returns 0 on success, non-zero on failure.
    """
    tmp = Path(tempfile.mkdtemp(prefix="eval_dataset_selftest_"))
    try:
        labels = pd.DataFrame([
            {"track_id": "t1", "your_verdict": "misattributed"},  # flagged -> TP
            {"track_id": "t2", "your_verdict": "correct"},        # flagged -> FP
            {"track_id": "t3", "your_verdict": "misattributed"},  # not flagged -> FN
            {"track_id": "t4", "your_verdict": "correct"},        # not flagged -> TN
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
        labels.to_csv(tmp / "labels_SELFTEST.csv", index=False)
        key.to_csv(tmp / "answer_key_SELFTEST.csv", index=False)

        res = score_group("SELFTEST", labels_dir=tmp)

        assert res["status"] == "ok", res["status"]
        assert res["tp"] == 1 and res["fp"] == 1 and res["fn"] == 1 and res["tn"] == 1, res
        assert res["n_scored"] == 4, res["n_scored"]
        assert res["n_unsure"] == 1, res["n_unsure"]
        # blank row excluded: 6 labeled rows, 1 blank dropped before join
        assert res["n_total"] == 5, res["n_total"]
        assert abs(res["precision"] - 0.5) < 1e-9, res["precision"]
        assert abs(res["recall"] - 0.5) < 1e-9, res["recall"]
        assert abs(res["f1"] - 0.5) < 1e-9, res["f1"]
        assert abs(res["accuracy"] - 0.5) < 1e-9, res["accuracy"]

        # discovery picks up the synthetic pair
        assert discover_groups(tmp) == ["SELFTEST"], discover_groups(tmp)

        print("self-test OK: score_group math and unsure/blank exclusion verified.")
        return 0
    except AssertionError as exc:
        print(f"self-test FAILED: {exc}")
        return 1
    finally:
        for f in tmp.glob("*"):
            f.unlink()
        tmp.rmdir()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("groups", nargs="*",
                        help="Limit scoring to these group names (default: all discovered).")
    parser.add_argument("--self-test", action="store_true",
                        help="Run an offline sanity check of the scoring math and exit.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Also report the real-data false-positive rate when "
                             "flagging only tracks with gap > THRESHOLD "
                             "(e.g. 0.06, chosen from tools/injection_eval.py).")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    groups = args.groups if args.groups else discover_groups()
    if not groups:
        print(f"No labeled groups found in {LABELS_DIR}/ "
              "(need both labels_<group>.csv and answer_key_<group>.csv).")
        return 0

    report(groups)
    if args.threshold is not None:
        report_threshold(groups, args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
