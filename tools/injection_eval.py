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

Usage: python tools/injection_eval.py [GROUP] [N_INJECT] [N_TRIALS]
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
            len(artist_ids))


def main():
    base = data_source.load_group(GROUP).frame.copy()
    recalls, fps, aucs = [], [], []
    tfi = ti = tfc = tc = 0
    n_artists = base["artist_id"].astype(str).nunique()

    for t in range(N_TRIALS):
        rng = np.random.default_rng(BASE_SEED + t)
        recall, fp, auc, fi, n_imp, fc, n_clean, n_art = _run_once(base, rng)
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


if __name__ == "__main__":
    main()
