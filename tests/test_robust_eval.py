"""Tests for release-level injection in tools/robust_eval.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.robust_eval import inject_releases


def _frame() -> pd.DataFrame:
    rows = []
    for artist, albums in {"A": {"a1": 3, "a2": 2, "a3": 2}, "B": {"b1": 4, "b2": 2}}.items():
        for album, n in albums.items():
            for k in range(n):
                rows.append({"track_id": f"{album}_{k}", "artist_id": artist, "album_id": album,
                             "credited_artist_ids": f"{artist}|G"})
    return pd.DataFrame(rows)


def test_inject_releases_moves_whole_albums_and_keeps_two_tracks():
    frame = _frame()
    mask, moves = inject_releases(frame, 1, np.random.default_rng(0))
    assert mask is not None and len(moves) == 1
    album, old, new, k = moves[0]
    moved = frame["album_id"] == album
    assert moved.sum() == k and mask.sum() == k and (mask == moved.to_numpy()).all()
    assert (frame.loc[moved, "artist_id"] == new).all() and new != old
    assert all(old not in c.split("|") and new in c.split("|") and "G" in c.split("|")
               for c in frame.loc[moved, "credited_artist_ids"])
    assert (frame["artist_id"].value_counts() >= 2).all()


def test_inject_releases_returns_none_when_impossible():
    frame = _frame()
    mask, moves = inject_releases(frame, 5, np.random.default_rng(0))
    assert mask is None and moves == []
