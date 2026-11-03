"""Generate a blind labeling template for a same-name group, with evidence.

Produces two files under labels/:
  * labels_<group>.csv      -> you fill this in. Includes factual metadata
                               "evidence" columns to make judging fast, but NOT
                               the tool's scores or flags (stay blind).
  * answer_key_<group>.csv   -> the tool's scores/flags, hidden until you compare.

The labeling CSV mixes every review candidate with a random sample of
tool-"agreed" tracks, so you can measure both false alarms and misses.

Usage: python tools/make_labels.py [GROUP] [N_AGREE_SAMPLE]
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import data_source
from src.app_core import prepare_group

GROUP = sys.argv[1] if len(sys.argv) > 1 else "WITCH"
N_AGREE_SAMPLE = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SEED = 42

out_dir = Path("labels")
out_dir.mkdir(exist_ok=True)

view = prepare_group(GROUP)
rows = pd.DataFrame(view.report["rows"])

candidates = rows[~rows["agrees"]]
agreed = rows[rows["agrees"]]
sample = agreed.sample(n=min(N_AGREE_SAMPLE, len(agreed)), random_state=SEED)

selected = pd.concat([candidates, sample], ignore_index=True)
selected = selected.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # shuffle

# --- Evidence: factual metadata from the group CSV (no tool opinion) ---------
raw = data_source.load_group(GROUP).frame.copy()
raw["release_year"] = (
    raw["album_release_date"].astype("string").str.extract(r"^(\d{4})")[0]
)


def _other_collaborators(row):
    """Credited collaborator names excluding the group's own same-name artists."""
    ids = str(row["credited_artist_ids"]).split("|")
    names = str(row["credited_artist_names"]).split("|")
    group_ids = set(view.profiles)
    guests = [n for i, n in zip(ids, names) if i and i not in group_ids]
    return ", ".join(dict.fromkeys(guests))  # dedupe, keep order


raw["other_collaborators"] = raw.apply(_other_collaborators, axis=1)
evidence_cols = ["track_id", "album_name", "release_year", "album_type",
                 "duration_ms", "other_collaborators", "credited_artist_names"]
evidence = raw[evidence_cols]

merged = selected.merge(evidence, on="track_id", how="left")

# Blind labeling sheet: identity + evidence + empty verdict columns, NO tool scores.
label_sheet = pd.DataFrame({
    "track_id": merged["track_id"],
    "track_name": merged["track_name"],
    "credited_artist_id": merged["credited_artist_id"],
    "credited_artist_names": merged["credited_artist_names"],
    # evidence to speed up judging:
    "album_name": merged["album_name"],
    "release_year": merged["release_year"],
    "album_type": merged["album_type"],
    "duration_ms": merged["duration_ms"],
    "other_collaborators": merged["other_collaborators"],
    # you fill these:
    "your_verdict": "",      # correct | misattributed | unsure
    "true_artist_id": "",    # if misattributed, which artist it really is
    "reason": "",            # one line: what you heard / where you checked
    "source": "",            # e.g. album page, Discogs, band website
})
label_path = out_dir / f"labels_{GROUP}.csv"
label_sheet.to_csv(label_path, index=False)

# Answer key: the tool's view, to compare AFTER you label.
answer_key = selected[[
    "track_id", "track_name", "credited_artist_id",
    "best_fit_artist_id", "credited_score", "best_fit_score", "gap", "agrees",
]].copy()
answer_key = answer_key.rename(columns={"agrees": "tool_says_correct"})
answer_key["tool_flagged_candidate"] = ~answer_key["tool_says_correct"]
key_path = out_dir / f"answer_key_{GROUP}.csv"
answer_key.to_csv(key_path, index=False)

print(f"Group: {GROUP}")
print(f"  review candidates: {len(candidates)}")
print(f"  agreed-sample:     {len(sample)}")
print(f"  total to label:    {len(selected)}")
print(f"Wrote {label_path}")
print(f"Wrote {key_path}  (don't peek until you've labeled!)")
