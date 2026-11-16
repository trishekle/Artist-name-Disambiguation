"""Synthetic injection evaluation: can the tool catch KNOWN misattributions?

Idea
----
Start from a group whose tracks are all correctly attributed (e.g. verified
WITCH). In each trial, deliberately relabel a few tracks to a DIFFERENT
same-name artist in the group -- these are known-misattributed "impostors". To
mimic a real misattribution, the impostor's ``credited_artist_ids`` is also
rewritten so the old (true) artist id is replaced by the new (wrong) one; any
other credited ids (guests) are kept. Otherwise the true id would leak back in
as a fake guest feature and make detection artificially easy.

The pipeline is rebuilt on the corrupted labels (so profiles and collaborator
features reflect the wrong assignment, exactly as a real misattribution would)
and the tool is run with leave-one-out scoring.

Metrics
-------
* recall              = flagged impostors / planted impostors   (flag = best-fit != assigned)
* false-positive rate = flagged clean tracks / clean tracks
* ROC AUC             = how well the score gap (best_other - assigned) separates
                        impostors from clean tracks. AUC ~ 0.5 means no signal;
                        clearly > 0.5 means the metadata carries real signal.
* random baseline     = 1 - 1/n_artists, the recall a coin-flip "flag" would get.

Averaged over several random trials so the numbers are stable.

* threshold sweep     = recall / FPR / precision at a realistic base rate as the
                        gap cutoff moves. Injection plants ~10 errors per group,
                        far above reality (~1%), so raw precision is inflated;
                        the sweep rescales it to the assumed BASE_RATE.

Usage: python tools/injection_eval.py [GROUP] [N_INJECT] [N_TRIALS] [BASE_RATE] [THRESHOLD]

THRESHOLD adds a "CHOSEN" row for a fixed gap cutoff, so a value picked on one
group (e.g. 0.06 from Morphine's FPR~5% row) can be checked on another:
  python tools/injection_eval.py WITCH 10 20 0.01 0.06
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import data_source
from src.analysis.features import load_features
from src.analysis.profiles import build_artist_profiles
from src.analysis.scoring import score_tracks

GROUP = sys.argv[1] if len(sys.argv) > 1 else "WITCH"
N_INJECT = int(sys.argv[2]) if len(sys.argv) > 2 else 10
N_TRIALS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
# Assumed real-world share of misattributed tracks, used to rescale precision.
BASE_RATE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
# Optional fixed gap cutoff to test (e.g. one chosen on another group).
THRESHOLD = float(sys.argv[5]) if len(sys.argv) > 5 else None
BASE_SEED = 1234


def _relabel(frame: pd.DataFrame, row: int, new_id: str, true_id: str) -> None:
    """Reassign a row to ``new_id`` and rewrite its credited ids realistically."""
    frame.at[row, "artist_id"] = new_id
    credited = str(frame.at[row, "credited_artist_ids"])
    parts = credited.split("|") if credited and credited != "nan" else []
    # Replace the true id with the new id; keep guests; dedupe preserving order.
    rewritten = [new_id if p == true_id else p for p in parts]
    if new_id not in rewritten:
        rewritten = [new_id] + rewritten
    frame.at[row, "credited_artist_ids"] = "|".join(dict.fromkeys(rewritten))


def _run_once(base_frame: pd.DataFrame, rng: np.random.Generator):
    frame = base_frame.copy().reset_index(drop=True)
    artist_ids = frame["artist_id"].astype(str).unique().tolist()
    if len(artist_ids) < 2:
        raise ValueError(f"{GROUP} has <2 artist IDs; cannot inject misattributions.")

    n = min(N_INJECT, len(frame))
    impostor_rows = rng.choice(len(frame), size=n, replace=False)
    impostor_mask = np.zeros(len(frame), dtype=bool)
    impostor_mask[impostor_rows] = True

    for row in impostor_rows:
        true_id = str(frame.at[row, "artist_id"])
        new_id = rng.choice([a for a in artist_ids if a != true_id])
        _relabel(frame, row, new_id, true_id)

    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / f"{GROUP}_injected.csv"
        frame.to_csv(csv, index=False)
        features = load_features(csv)
        profiles = build_artist_profiles(features)
        scored = score_tracks(features, profiles, leave_one_out=True)

    # Align to feature frame order.
    scored = scored.set_index(scored["track_id"].astype(str))
    track_ids = features.frame["track_id"].astype(str).to_numpy()
    scored = scored.loc[track_ids]

    flagged = (scored["best_other_score"].to_numpy() > scored["assigned_score"].to_numpy())
    gap = scored["score_gap"].to_numpy()  # best_other - assigned (leave-one-out)

    imp = impostor_mask
    clean = ~impostor_mask
    recall = (flagged & imp).sum() / imp.sum() if imp.sum() else float("nan")
    fp = (flagged & clean).sum() / clean.sum() if clean.sum() else float("nan")

    # AUC: gap as detection score, impostor = positive. Needs both classes present.
    auc = float("nan")
    if imp.sum() and clean.sum() and len(np.unique(gap)) > 1:
        auc = roc_auc_score(imp.astype(int), gap)

    return (recall, fp, auc,
            int((flagged & imp).sum()), int(imp.sum()),
            int((flagged & clean).sum()), int(clean.sum()),
            len(artist_ids), gap, imp)


def precision_at_base_rate(recall: float, fpr: float, base_rate: float) -> float:
    """Expected precision if only ``base_rate`` of real tracks are misattributed.

    Injection plants far more errors than reality has, so its raw precision is
    inflated. This rescales: TP ~ recall * base_rate, FP ~ fpr * (1 - base_rate).
    """
    tp = recall * base_rate
    fp = fpr * (1 - base_rate)
    return tp / (tp + fp) if (tp + fp) else float("nan")


def threshold_sweep(gaps: np.ndarray, imp: np.ndarray, base_rate: float) -> None:
    """Print recall / FPR / base-rate precision as the flag threshold moves.

    Flag rule here is ``gap > threshold`` (the default rule is threshold 0).
    Thresholds are chosen from the clean-track gap distribution so each row
    hits a target false-positive rate, plus the default cutoff of 0 (which
    matches the best-fit != assigned rule used in the summary above).
    """
    clean_gaps = gaps[~imp]
    imp_gaps = gaps[imp]
    rows = []
    for target_fpr in [0.20, 0.10, 0.05, 0.02, 0.01]:
        rows.append((f"FPR~{target_fpr:.0%}", float(np.quantile(clean_gaps, 1 - target_fpr))))
    rows.append(("default (0)", 0.0))
    if THRESHOLD is not None:
        rows.append(("CHOSEN", THRESHOLD))

    print(f"\n=== Threshold sweep (pooled over trials; flag = gap > threshold) ===")
    print(f"Precision column assumes a real misattribution base rate of {base_rate:.1%}.\n")
    print(f"{'target':>12} | {'threshold':>9} | {'recall':>7} | {'FPR':>6} | "
          f"{'precision@base':>14} | {'reviews per real error':>22}")
    print("-" * 86)
    for label, thr in rows:
        recall = float((imp_gaps > thr).mean()) if len(imp_gaps) else float("nan")
        fpr = float((clean_gaps > thr).mean()) if len(clean_gaps) else float("nan")
        prec = precision_at_base_rate(recall, fpr, base_rate)
        per_error = f"{1 / prec:.1f}" if prec and prec == prec else "n/a"
        print(f"{label:>12} | {thr:>9.4f} | {recall:>6.1%} | {fpr:>5.1%} | "
              f"{prec:>13.1%} | {per_error:>22}")
    print("\nReading it: pick the row whose 'reviews per real error' a reviewer can live with.")
    print("Recall/FPR come from synthetic injection; check FPR against the real-data")
    print("false-positive rate from tools/evaluate_dataset.py (MusicBrainz-verified tracks).")


def main():
    base = data_source.load_group(GROUP).frame.copy()
    recalls, fps, aucs = [], [], []
    all_gaps, all_imp = [], []
    tfi = ti = tfc = tc = 0
    n_artists = base["artist_id"].astype(str).nunique()

    for t in range(N_TRIALS):
        rng = np.random.default_rng(BASE_SEED + t)
        recall, fp, auc, fi, n_imp, fc, n_clean, n_art, gap, imp = _run_once(base, rng)
        all_gaps.append(gap)
        all_imp.append(imp)
        recalls.append(recall)
        fps.append(fp)
        if auc == auc:
            aucs.append(auc)
        tfi += fi; ti += n_imp; tfc += fc; tc += n_clean

    recalls = np.array(recalls); fps = np.array(fps); aucs = np.array(aucs)
    random_baseline = 1 - 1 / n_artists

    print(f"=== Synthetic injection evaluation: {GROUP} ===\n")
    print(f"Trials: {N_TRIALS}   Injected per trial: {N_INJECT}   Artists: {n_artists}")
    print("Flag rule: best-fit artist != assigned artist. Scoring: leave-one-out.\n")
    print(f"  Recall:              mean {recalls.mean():.1%}  std {recalls.std():.1%}  (pooled {tfi}/{ti})")
    print(f"  False-positive rate: mean {fps.mean():.1%}  std {fps.std():.1%}  (pooled {tfc}/{tc})")
    print(f"  Random-flag baseline recall (1 - 1/n_artists): {random_baseline:.1%}")
    print()
    if len(aucs):
        print(f"  >>> ROC AUC (gap separates impostors from clean): "
              f"mean {aucs.mean():.3f}  std {aucs.std():.3f} <<<")
        print("      AUC ~0.50 = no signal in metadata;  >0.5 = real signal.")
    else:
        print("  ROC AUC: undefined (need both classes with varied gaps).")

    threshold_sweep(np.concatenate(all_gaps), np.concatenate(all_imp), BASE_RATE)


if __name__ == "__main__":
    main()
