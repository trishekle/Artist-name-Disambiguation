"""Tests for the evaluation bug fixes: release-year regex and leave-one-out scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.features import _release_year, load_features
from src.analysis.profiles import build_artist_profiles
from src.analysis.scoring import compute_score_matrix


def test_release_year_extracts_years():
    out = _release_year(pd.Series(["2019-05-03", "1998", None]))
    assert out.iloc[0] == 2019
    assert out.iloc[1] == 1998
    assert pd.isna(out.iloc[2])


def _outlier_group_frame() -> pd.DataFrame:
    """Artist A has 3 coherent tracks + 1 blatant outlier; artist B has 2 tracks."""
    def row(tid, name, artist, album, dur, year, num):
        return {
            "track_id": tid, "track_name": name, "artist_id": artist, "artist_name": "Twins",
            "album_id": album, "album_name": album, "duration_ms": dur, "explicit": False,
            "track_number": num, "disc_number": 1, "album_type": "album",
            "album_release_date": f"{year}-01-01", "album_total_tracks": 10,
            "credited_artist_ids": artist, "credited_artist_names": "Twins",
        }
    rows = [
        row("a1", "Iron Thunder", "A", "Steel", 180000, 1985, 1),
        row("a2", "Blood Steel", "A", "Steel", 185000, 1985, 2),
        row("a3", "War Hymn", "A", "Steel", 190000, 1986, 3),
        # Blatant outlier credited to A: very different name, duration, era.
        row("a4", "Gentle Quiet Ambient Drift Clouds", "A", "Calm", 600000, 2022, 1),
        row("b1", "Soft Rain", "B", "Calm2", 420000, 2021, 1),
        row("b2", "Still Ocean", "B", "Calm2", 440000, 2021, 2),
    ]
    return pd.DataFrame(rows)


def test_leave_one_out_lowers_outlier_assigned_score(tmp_path):
    csv = tmp_path / "group.csv"
    _outlier_group_frame().to_csv(csv, index=False)
    features = load_features(csv)
    profiles = build_artist_profiles(features)

    artist_ids = list(profiles)
    a_col = artist_ids.index("A")
    # Row index of the outlier track a4 in frame order.
    outlier_row = int(np.where(features.frame["track_id"].astype(str).to_numpy() == "a4")[0][0])

    with_loo, _ = compute_score_matrix(features, profiles, leave_one_out=True)
    without_loo, _ = compute_score_matrix(features, profiles, leave_one_out=False)

    # Leave-one-out removes the outlier from its own centroid, so its assigned
    # score should drop relative to including itself.
    assert with_loo[outlier_row, a_col] < without_loo[outlier_row, a_col]


def test_non_assigned_scores_unchanged_by_loo(tmp_path):
    csv = tmp_path / "group.csv"
    _outlier_group_frame().to_csv(csv, index=False)
    features = load_features(csv)
    profiles = build_artist_profiles(features)
    artist_ids = list(profiles)

    with_loo, _ = compute_score_matrix(features, profiles, leave_one_out=True)
    without_loo, _ = compute_score_matrix(features, profiles, leave_one_out=False)

    assigned = features.frame["artist_id"].astype(str).to_numpy()
    for i in range(len(features.frame)):
        for j, aid in enumerate(artist_ids):
            if aid != assigned[i]:
                assert abs(with_loo[i, j] - without_loo[i, j]) < 1e-12
