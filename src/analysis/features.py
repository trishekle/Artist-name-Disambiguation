"""Sparse, explainable metadata features for one same-name artist CSV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED_COLUMNS = {
    "track_id", "track_name", "artist_id", "artist_name", "album_name",
    "duration_ms", "explicit", "track_number", "disc_number", "album_type",
    "album_release_date", "album_total_tracks", "credited_artist_ids",
}
NUMERIC_COLUMNS = [
    "duration_ms", "track_number", "disc_number", "album_total_tracks", "release_year",
]


@dataclass
class FeatureSet:
    """Feature blocks remain separate so scoring weights stay interpretable."""

    frame: pd.DataFrame
    numeric: sparse.csr_matrix
    album: sparse.csr_matrix
    text: sparse.csr_matrix
    collaborators: sparse.csr_matrix

    @property
    def track_ids(self) -> pd.Series:
        return self.frame["track_id"]


def _release_year(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype("string").str.extract(r"^(\d{4})")[0], errors="coerce")


def _collaborator_documents(frame: pd.DataFrame) -> list[str]:
    """Encode guests only; otherwise the assigned identity would leak into a score."""
    return [
        " ".join(
            f"artist_{item}" for item in str(credited).split("|")
            if item and item != str(assigned)
        )
        for credited, assigned in zip(
            frame["credited_artist_ids"].fillna(""), frame["artist_id"]
        )
    ]


def load_features(csv_path: str | Path) -> FeatureSet:
    """Load and transform one enriched CSV, preserving row/track-ID alignment."""
    frame = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            f"{csv_path} is not an enriched metadata CSV; missing: {', '.join(missing)}. "
            "Regenerate it with src.ingestion.ingestion first."
        )
    if frame.empty:
        raise ValueError(f"{csv_path} contains no tracks.")
    if frame["track_id"].isna().any() or frame["artist_id"].isna().any():
        raise ValueError("track_id and artist_id must be present for every row.")

    frame = frame.copy().reset_index(drop=True)
    frame["release_year"] = _release_year(frame["album_release_date"])
    numeric_values = frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric_values = numeric_values.fillna(numeric_values.median()).fillna(0.0)
    numeric_values["explicit"] = frame["explicit"].fillna(False).astype(bool).astype(float)
    numeric = sparse.csr_matrix(StandardScaler().fit_transform(numeric_values))
    # Include the binary flag in the numeric block after the normalized numeric fields.
    numeric = sparse.hstack([numeric, sparse.csr_matrix(numeric_values[["explicit"]])], format="csr")

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    album = encoder.fit_transform(frame[["album_type"]].fillna("unknown"))
    text_values = (frame["track_name"].fillna("") + " " + frame["album_name"].fillna(""))
    text = TfidfVectorizer(lowercase=True, ngram_range=(1, 2)).fit_transform(text_values)
    collaborator_docs = _collaborator_documents(frame)
    collaborators = (
        TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False).fit_transform(collaborator_docs)
        if any(collaborator_docs) else sparse.csr_matrix((len(frame), 0))
    )
    return FeatureSet(frame, numeric, album.tocsr(), text.tocsr(), collaborators.tocsr())
