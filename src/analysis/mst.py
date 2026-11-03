"""Pairwise + Minimum-Spanning-Tree misattribution detection.

Instead of comparing each track to its artist's *centroid* (which an injected
impostor can contaminate), this compares tracks *pairwise* within a single
assigned artist, builds an MST, and cuts long edges. A track that does not
belong hangs off the tree by a long edge and splits into its own component.

This mirrors the approach in Spotify's "Which Witch?" (ISMIR 2023): pairwise
distances -> MST -> threshold-cut long edges to split a discography.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from .features import FeatureSet

# Same blocks/weights as scoring, so the two methods use a comparable signal.
WEIGHTS = {"text": 0.30, "duration": 0.25, "release": 0.15, "collaboration": 0.30}


def _block_similarity(features: FeatureSet, rows: np.ndarray) -> np.ndarray:
    """Weighted pairwise similarity (0..1) among the given rows of the frame."""
    numeric = features.numeric.toarray()[rows]
    text = cosine_similarity(features.text[rows])
    album = cosine_similarity(features.album[rows])
    collab = (
        cosine_similarity(features.collaborators[rows])
        if features.collaborators.shape[1] else np.zeros((len(rows), len(rows)))
    )
    # Duration block = first numeric column; release block = the rest.
    def _pair_bounded(cols):
        # 1 / (1 + mean abs diff) elementwise over the column slice -> similarity.
        sub = numeric[:, cols]
        diff = np.abs(sub[:, None, :] - sub[None, :, :]).mean(axis=2)
        return 1.0 / (1.0 + diff)

    duration = _pair_bounded(slice(0, 1))
    release = _pair_bounded(slice(1, None))
    sim = (WEIGHTS["text"] * text + WEIGHTS["duration"] * duration +
           WEIGHTS["release"] * release + WEIGHTS["collaboration"] * collab)
    np.fill_diagonal(sim, 1.0)
    return sim


def detect_for_artist(features: FeatureSet, rows: np.ndarray, cut_threshold: float) -> np.ndarray:
    """Return a boolean mask (over ``rows``) of tracks split off the main cluster.

    Builds an MST on distance = 1 - similarity, removes edges whose distance
    exceeds ``cut_threshold``, and flags every track not in the largest
    remaining component.
    """
    n = len(rows)
    if n <= 2:
        return np.zeros(n, dtype=bool)  # too small to split meaningfully
    sim = _block_similarity(features, rows)
    distance = 1.0 - sim

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            graph.add_edge(i, j, weight=distance[i, j])
    mst = nx.minimum_spanning_tree(graph)

    # Cut long edges.
    mst.remove_edges_from([(u, v) for u, v, d in mst.edges(data=True) if d["weight"] > cut_threshold])

    components = list(nx.connected_components(mst))
    if len(components) <= 1:
        return np.zeros(n, dtype=bool)
    main = max(components, key=len)
    flagged = np.ones(n, dtype=bool)
    for idx in main:
        flagged[idx] = False
    return flagged


def detect(features: FeatureSet, cut_threshold: float) -> dict[str, np.ndarray]:
    """Run MST detection per assigned artist.

    Returns a dict mapping track_id -> flagged(bool). A track is flagged if the
    MST split it off from its assigned artist's main cluster.
    """
    frame = features.frame
    assigned = frame["artist_id"].astype(str).to_numpy()
    track_ids = frame["track_id"].astype(str).to_numpy()
    result: dict[str, bool] = {}
    for artist_id in np.unique(assigned):
        rows = np.where(assigned == artist_id)[0]
        mask = detect_for_artist(features, rows, cut_threshold)
        for local, global_row in enumerate(rows):
            result[track_ids[global_row]] = bool(mask[local])
    return result
