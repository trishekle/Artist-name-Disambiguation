"""Transparent track-to-artist metadata compatibility scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .features import FeatureSet
from .profiles import ArtistProfile

WEIGHTS = {"text": 0.30, "duration": 0.25, "release": 0.15, "collaboration": 0.30}


def _bounded_distance(values: np.ndarray, centroid: np.ndarray, columns: slice) -> np.ndarray:
    distance = np.mean(np.abs(values[:, columns] - centroid[columns]), axis=1)
    return 1.0 / (1.0 + distance)


def _blocks_to_score(numeric, text_sim, collab_sim, album_sim, numeric_centroid, weights, rows=None):
    """Combine the four block similarities into a weighted score for the given rows.

    ``text_sim``, ``collab_sim``, ``album_sim`` are precomputed per-row similarities;
    the numeric/duration/release blocks are derived from ``numeric`` vs
    ``numeric_centroid``. ``rows`` restricts the numeric computation to a subset.
    """
    values = numeric if rows is None else numeric[rows]
    duration = _bounded_distance(values, numeric_centroid, slice(0, 1))
    release_numeric = _bounded_distance(values, numeric_centroid, slice(1, None))
    release = (release_numeric + album_sim) / 2
    return (weights["text"] * text_sim + weights["duration"] * duration +
            weights["release"] * release + weights["collaboration"] * collab_sim)


def compute_score_matrix(
    features: FeatureSet,
    profiles: dict[str, ArtistProfile],
    weights: dict | None = None,
    leave_one_out: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Return the full per-track x per-artist compatibility matrix and its artist-ID columns.

    ``matrix[i, j]`` is the compatibility of track ``i`` with the artist whose ID is
    ``artist_ids[j]``. Row order matches ``features.frame``; column order matches the
    insertion order of ``profiles``.

    When ``leave_one_out`` is True (default), the score of a track against its OWN
    assigned artist is computed from that artist's centroid with the track removed
    (centroid = (sum of members - track) / (n - 1), per block). This prevents a track
    from inflating its own assigned-artist score, which otherwise makes genuine
    misattributions almost impossible to flag. Scores against non-assigned artists are
    unchanged. For a single-track artist (n = 1) leave-one-out is undefined, so that
    track's assigned score falls back to its mean score against the OTHER artists
    (a neutral value that neither rewards nor penalizes the lone track).
    """
    weights = weights or WEIGHTS
    if set(weights) != set(WEIGHTS) or not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("weights must contain text, duration, release, collaboration and sum to 1.")
    numeric = features.numeric.toarray()
    artist_ids = list(profiles)
    n_tracks = len(features.frame)
    scores = np.empty((n_tracks, len(artist_ids)))

    # --- 1. Standard scores: every track vs every artist's full centroid ----------
    for column, artist_id in enumerate(artist_ids):
        profile = profiles[artist_id]
        text = cosine_similarity(features.text, profile.text_centroid).ravel()
        collaborators = (
            cosine_similarity(features.collaborators, profile.collaborator_centroid).ravel()
            if features.collaborators.shape[1] else np.zeros(n_tracks)
        )
        album = cosine_similarity(features.album, profile.album_centroid.reshape(1, -1)).ravel()
        scores[:, column] = _blocks_to_score(
            numeric, text, collaborators, album, profile.numeric_centroid, weights
        )

    if not leave_one_out:
        return scores, artist_ids

    # --- 2. Overwrite each track's ASSIGNED-artist score with a leave-one-out score -
    col_index = {artist_id: i for i, artist_id in enumerate(artist_ids)}
    assigned_ids = features.frame["artist_id"].astype(str).to_numpy()
    text_dense = features.text.toarray()
    album_dense = features.album.toarray()
    has_collab = features.collaborators.shape[1] > 0
    collab_dense = features.collaborators.toarray() if has_collab else None

    for artist_id, profile in profiles.items():
        members = np.where(assigned_ids == artist_id)[0]
        n_a = len(members)
        column = col_index[artist_id]
        if n_a <= 1:
            # Single-track artist: leave-one-out undefined. Use the neutral fallback
            # of the track's mean score against the other artists.
            for i in members:
                others = [c for c in range(len(artist_ids)) if c != column]
                scores[i, column] = scores[i, others].mean() if others else scores[i, column]
            continue

        # Per-block sums over this artist's members, for O(1) leave-one-out centroids.
        text_sum = text_dense[members].sum(axis=0)
        album_sum = album_dense[members].sum(axis=0)
        numeric_sum = numeric[members].sum(axis=0)
        collab_sum = collab_dense[members].sum(axis=0) if has_collab else None

        for i in members:
            loo_text_centroid = (text_sum - text_dense[i]) / (n_a - 1)
            loo_album_centroid = (album_sum - album_dense[i]) / (n_a - 1)
            loo_numeric_centroid = (numeric_sum - numeric[i]) / (n_a - 1)

            text_sim = cosine_similarity(text_dense[i:i + 1], loo_text_centroid.reshape(1, -1)).ravel()
            album_sim = cosine_similarity(album_dense[i:i + 1], loo_album_centroid.reshape(1, -1)).ravel()
            if has_collab:
                loo_collab_centroid = (collab_sum - collab_dense[i]) / (n_a - 1)
                collab_sim = cosine_similarity(
                    collab_dense[i:i + 1], loo_collab_centroid.reshape(1, -1)
                ).ravel()
            else:
                collab_sim = np.zeros(1)

            scores[i, column] = _blocks_to_score(
                numeric, text_sim, collab_sim, album_sim, loo_numeric_centroid, weights,
                rows=np.array([i]),
            )[0]

    return scores, artist_ids


def score_tracks(
    features: FeatureSet,
    profiles: dict[str, ArtistProfile],
    weights: dict | None = None,
    leave_one_out: bool = True,
) -> pd.DataFrame:
    """Return every track's assigned versus best alternative compatibility."""
    scores, artist_ids = compute_score_matrix(features, profiles, weights, leave_one_out=leave_one_out)
    assigned_ids = features.frame["artist_id"].astype(str).to_numpy()
    assigned_indices = np.array([artist_ids.index(value) for value in assigned_ids])
    assigned = scores[np.arange(len(scores)), assigned_indices]
    masked = scores.copy()
    masked[np.arange(len(scores)), assigned_indices] = -np.inf
    best_indices = masked.argmax(axis=1)
    best_other = masked[np.arange(len(scores)), best_indices]
    result = features.frame[["track_id", "track_name", "artist_id", "artist_name"]].copy()
    result["assigned_score"] = assigned
    result["best_other_score"] = best_other
    result["best_other_artist_id"] = [artist_ids[index] for index in best_indices]
    result["score_gap"] = best_other - assigned
    return result
