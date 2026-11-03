"""Bipartite track<->artist graph built from the compatibility score matrix.

Tier 1 scope: artist nodes and track nodes only, with one weighted edge from
every track to every candidate artist (weight = metadata compatibility score).
There are no track<->track edges yet; those are Tier 2. Keeping the graph a clean
reframing of ``scoring.compute_score_matrix`` means the UI and metric read the same
numbers the CLI does.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .features import FeatureSet
from .profiles import ArtistProfile
from .scoring import compute_score_matrix

ARTIST_NODE = "artist"
TRACK_NODE = "track"


def artist_node_id(artist_id: str) -> str:
    """Namespaced node id so an artist ID never collides with a track ID."""
    return f"artist::{artist_id}"


def track_node_id(track_id: str) -> str:
    return f"track::{track_id}"


def build_bipartite_graph(
    features: FeatureSet,
    profiles: dict[str, ArtistProfile],
    weights: dict | None = None,
    leave_one_out: bool = True,
) -> nx.Graph:
    """Return a weighted bipartite graph of tracks and candidate artists.

    Artist nodes carry ``type="artist"``; track nodes carry ``type="track"`` plus
    their ``credited_artist_id``. Each track connects to every artist with
    ``weight`` equal to that track's compatibility score for the artist. See
    :func:`~src.analysis.scoring.compute_score_matrix` for ``leave_one_out``.
    """
    matrix, artist_ids = compute_score_matrix(features, profiles, weights, leave_one_out=leave_one_out)
    graph = nx.Graph()

    for artist_id in artist_ids:
        profile = profiles[artist_id]
        graph.add_node(
            artist_node_id(artist_id),
            type=ARTIST_NODE,
            artist_id=artist_id,
            artist_name=profile.artist_name,
            track_count=profile.track_count,
        )

    frame = features.frame
    track_ids = frame["track_id"].astype(str).to_numpy()
    track_names = frame["track_name"].astype(str).to_numpy()
    credited = frame["artist_id"].astype(str).to_numpy()

    for row, track_id in enumerate(track_ids):
        t_node = track_node_id(track_id)
        graph.add_node(
            t_node,
            type=TRACK_NODE,
            track_id=track_id,
            track_name=track_names[row],
            credited_artist_id=credited[row],
        )
        for column, artist_id in enumerate(artist_ids):
            graph.add_edge(t_node, artist_node_id(artist_id), weight=float(matrix[row, column]))

    return graph


def track_scores(graph: nx.Graph, track_id: str) -> dict[str, float]:
    """Return {artist_id: weight} for one track, for the UI's inspect panel."""
    t_node = track_node_id(track_id)
    out = {}
    for neighbor, data in graph[t_node].items():
        nd = graph.nodes[neighbor]
        if nd.get("type") == ARTIST_NODE:
            out[nd["artist_id"]] = float(data["weight"])
    return out


def graph_summary(graph: nx.Graph) -> dict:
    """Node/edge counts by type, for quick CLI/UI reporting."""
    tracks = [n for n, d in graph.nodes(data=True) if d.get("type") == TRACK_NODE]
    artists = [n for n, d in graph.nodes(data=True) if d.get("type") == ARTIST_NODE]
    return {
        "track_nodes": len(tracks),
        "artist_nodes": len(artists),
        "edges": graph.number_of_edges(),
    }


def best_fit_artist(graph: nx.Graph, track_node: str) -> tuple[str, float]:
    """Return the (artist_id, weight) of the track's strongest artist edge."""
    best_id, best_weight = None, -np.inf
    for neighbor, data in graph[track_node].items():
        if graph.nodes[neighbor].get("type") != ARTIST_NODE:
            continue
        if data["weight"] > best_weight:
            best_id, best_weight = graph.nodes[neighbor]["artist_id"], data["weight"]
    return best_id, float(best_weight)


def agreement_report(graph: nx.Graph) -> dict:
    """Compare each track's best-fit artist against its credited artist.

    A track "agrees" when its strongest artist edge points to the artist it is
    already credited to. Disagreements are review candidates, ranked by ``gap`` =
    best-fit weight minus credited-artist weight (how much a rival artist fit
    better). This is a review signal, not proof of misattribution: there is no
    ground truth, so a flagged track only means "worth a human look".
    """
    rows = []
    for node, data in graph.nodes(data=True):
        if data.get("type") != TRACK_NODE:
            continue
        credited = data["credited_artist_id"]
        best_id, best_weight = best_fit_artist(graph, node)
        credited_weight = float(graph[node][artist_node_id(credited)]["weight"])
        agrees = best_id == credited
        rows.append(
            {
                "track_id": data["track_id"],
                "track_name": data["track_name"],
                "credited_artist_id": credited,
                "best_fit_artist_id": best_id,
                "credited_score": credited_weight,
                "best_fit_score": best_weight,
                "gap": best_weight - credited_weight,
                "agrees": agrees,
            }
        )

    total = len(rows)
    agree_count = sum(1 for r in rows if r["agrees"])
    disagreements = sorted(
        (r for r in rows if not r["agrees"]), key=lambda r: r["gap"], reverse=True
    )
    return {
        "total_tracks": total,
        "agree_count": agree_count,
        "agreement_rate": (agree_count / total) if total else 0.0,
        "rows": rows,
        "disagreements": disagreements,
    }
