"""Synthetic injection evaluation: can the tool catch KNOWN misattributions?

Idea
----
Start from a group whose tracks are all correctly attributed (e.g. verified
WITCH). In each trial, deliberately relabel a few tracks to a DIFFERENT
same-name artist in the group -- these are known-misattributed "impostors".
Rebuild the pipeline on the corrupted labels (so profiles and collaborator
features reflect the wrong assignment, exactly as a real misattribution would)
and run the tool.

Metric (definition A)
---------------------
A track is "flagged" when the tool's best-fit artist != its (currently assigned)
artist -- i.e. it appears as a review candidate.

  recall              = flagged impostors / total impostors        (planted, wrong rows)
  false-positive rate = flagged clean tracks / total clean tracks  (untouched, correct rows)

Averaged over several random trials so the numbers are stable, not a fluke of
which tracks were picked.

Usage: python tools/injection_eval.py [GROUP] [N_INJECT] [N_TRIALS]
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import data_source
from src.analysis.features import load_features
from src.analysis.graph import agreement_report, build_bipartite_graph
from src.analysis.profiles import build_artist_profiles

GROUP = sys.argv[1] if len(sys.argv) > 1 else "WITCH"
N_INJECT = int(sys.argv[2]) if len(sys.argv) > 2 else 10
N_TRIALS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
BASE_SEED = 1234


def _run_once(base_frame: pd.DataFrame, rng: np.random.Generator):
    """Corrupt N_INJECT rows (reassign to another same-name artist), run the tool,
    and return (recall, fp_rate, n_impostors, n_clean, flagged_impostors, flagged_clean)."""
    frame = base_frame.copy().reset_index(drop=True)
    artist_ids = frame["artist_id"].astype(str).unique().tolist()
    if len(artist_ids) < 2:
        raise ValueError(f"{GROUP} has <2 artist IDs; cannot inject misattributions.")

    n = min(N_INJECT, len(frame))
    impostor_rows = rng.choice(len(frame), size=n, replace=False)

    impostor_mask = np.zeros(len(frame), dtype=bool)
    impostor_mask[impostor_rows] = True

    # Relabel each chosen row to a DIFFERENT artist in the group.
    for row in impostor_rows:
        true_id = str(frame.at[row, "artist_id"])
        others = [a for a in artist_ids if a != true_id]
        frame.at[row, "artist_id"] = rng.choice(others)
        # artist_name is a display label; keep it consistent with the new id's name
        # if we have one, else leave as-is (name is not used as an identifier).

    # Run the pipeline on the corrupted labels via a temp CSV (honest: the whole
    # pipeline sees the wrong assignment, just like a real misattribution).
    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / f"{GROUP}_injected.csv"
        frame.to_csv(csv, index=False)
        features = load_features(csv)
        profiles = build_artist_profiles(features)
        report = agreement_report(build_bipartite_graph(features, profiles))

    flagged_ids = {r["track_id"] for r in report["disagreements"]}
    track_ids = features.frame["track_id"].astype(str).to_numpy()
    flagged = np.array([tid in flagged_ids for tid in track_ids])

    # Align impostor_mask to features.frame order (load_features resets index but
    # preserves row order from the CSV we just wrote).
    imp = impostor_mask
    clean = ~impostor_mask

    flagged_impostors = int((flagged & imp).sum())
    flagged_clean = int((flagged & clean).sum())
    n_imp = int(imp.sum())
    n_clean = int(clean.sum())
    recall = flagged_impostors / n_imp if n_imp else float("nan")
    fp_rate = flagged_clean / n_clean if n_clean else float("nan")
    return recall, fp_rate, n_imp, n_clean, flagged_impostors, flagged_clean


def main():
    base = data_source.load_group(GROUP).frame.copy()
    recalls, fps = [], []
    tot_flag_imp = tot_imp = tot_flag_clean = tot_clean = 0

    for t in range(N_TRIALS):
        rng = np.random.default_rng(BASE_SEED + t)
        recall, fp, n_imp, n_clean, fi, fc = _run_once(base, rng)
        recalls.append(recall)
        fps.append(fp)
        tot_flag_imp += fi
        tot_imp += n_imp
        tot_flag_clean += fc
        tot_clean += n_clean

    recalls = np.array(recalls)
    fps = np.array(fps)

    print(f"=== Synthetic injection evaluation: {GROUP} ===\n")
    print(f"Trials: {N_TRIALS}   Injected per trial: {N_INJECT}")
    print(f"Metric A: a track is 'flagged' when best-fit artist != assigned artist.\n")
    print(f"Recall  (flagged impostors / planted impostors):")
    print(f"  mean = {recalls.mean():.1%}   std = {recalls.std():.1%}   "
          f"(pooled {tot_flag_imp}/{tot_imp})")
    print(f"False-positive rate (flagged clean tracks / clean tracks):")
    print(f"  mean = {fps.mean():.1%}   std = {fps.std():.1%}   "
          f"(pooled {tot_flag_clean}/{tot_clean})")
    print("\nReading: high recall = catches planted errors; low FP = doesn't cry "
          "wolf on correct tracks. On a metadata-only tool, expect modest recall.")


if __name__ == "__main__":
    main()
