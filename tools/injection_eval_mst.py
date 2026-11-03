"""Synthetic injection evaluation for the MST method, swept over cut thresholds.

Same injection protocol as tools/injection_eval.py (so results are comparable),
but detection uses the pairwise + MST splitter instead of centroid scoring.

For each cut threshold we report averaged recall (planted impostors flagged)
and false-positive rate (clean tracks flagged). We sweep the threshold and
print the whole curve rather than cherry-picking, to stay honest.

Usage: python tools/injection_eval_mst.py [GROUP] [N_INJECT] [N_TRIALS]
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import data_source
from src.analysis.features import load_features
from src.analysis import mst

GROUP = sys.argv[1] if len(sys.argv) > 1 else "WITCH"
N_INJECT = int(sys.argv[2]) if len(sys.argv) > 2 else 10
N_TRIALS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
BASE_SEED = 1234
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]


def _trial(base_frame, rng, thresholds):
    frame = base_frame.copy().reset_index(drop=True)
    artist_ids = frame["artist_id"].astype(str).unique().tolist()
    n = min(N_INJECT, len(frame))
    impostor_rows = rng.choice(len(frame), size=n, replace=False)
    impostor_mask = np.zeros(len(frame), dtype=bool)
    impostor_mask[impostor_rows] = True
    for row in impostor_rows:
        true_id = str(frame.at[row, "artist_id"])
        new_id = rng.choice([a for a in artist_ids if a != true_id])
        frame.at[row, "artist_id"] = new_id
        # Realistic misattribution: credited ids point to the wrong artist too.
        credited = str(frame.at[row, "credited_artist_ids"])
        parts = credited.split("|") if credited and credited != "nan" else []
        rewritten = [new_id if p == true_id else p for p in parts]
        if new_id not in rewritten:
            rewritten = [new_id] + rewritten
        frame.at[row, "credited_artist_ids"] = "|".join(dict.fromkeys(rewritten))

    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "inj.csv"
        frame.to_csv(csv, index=False)
        features = load_features(csv)
        track_ids = features.frame["track_id"].astype(str).to_numpy()
        # Align impostor mask to feature frame order (row order preserved).
        out = {}
        for thr in thresholds:
            flags = mst.detect(features, cut_threshold=thr)
            flagged = np.array([flags[t] for t in track_ids])
            imp = impostor_mask
            clean = ~impostor_mask
            out[thr] = (
                int((flagged & imp).sum()), int(imp.sum()),
                int((flagged & clean).sum()), int(clean.sum()),
            )
    return out


def main():
    base = data_source.load_group(GROUP).frame.copy()
    agg = {thr: [0, 0, 0, 0] for thr in THRESHOLDS}
    for t in range(N_TRIALS):
        rng = np.random.default_rng(BASE_SEED + t)
        res = _trial(base, rng, THRESHOLDS)
        for thr, (fi, ni, fc, nc) in res.items():
            agg[thr][0] += fi
            agg[thr][1] += ni
            agg[thr][2] += fc
            agg[thr][3] += nc

    print(f"=== MST injection evaluation: {GROUP} ===\n")
    print(f"Trials: {N_TRIALS}   Injected per trial: {N_INJECT}")
    print("A track is 'flagged' when the MST splits it off its assigned artist's main cluster.\n")
    print(f"{'cut_thr':>8} | {'recall':>8} | {'false_pos':>10}")
    print("-" * 34)
    for thr in THRESHOLDS:
        fi, ni, fc, nc = agg[thr]
        recall = fi / ni if ni else float("nan")
        fp = fc / nc if nc else float("nan")
        print(f"{thr:>8.2f} | {recall:>7.1%} | {fp:>9.1%}")
    print("\nCompare to centroid method baseline: recall ~0.5%, false-pos ~6.5%.")
    print("Lower cut_thr = more aggressive splitting (higher recall, higher false-pos).")


if __name__ == "__main__":
    main()
