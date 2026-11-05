"""MusicBrainz ground-truth labeler: TRUSTWORTHY, traceable labels, no guessing.

This REPLACES hand-labeling for a same-name group. Instead of a human listening
and judging, it derives each verdict from MusicBrainz FACTS and records exactly
which MB fact produced it, so every label is traceable back to an MBID.

Honesty caveats (read these before trusting the output):
  * Only MusicBrainz-resolvable tracks get a correct/misattributed verdict.
    Anything we cannot pin to a MusicBrainz recording -> ``unsure`` (excluded by
    tools/evaluate_labels.py, never silently counted right or wrong).
  * We NEVER fabricate a label with an LLM. The verdict is a MusicBrainz fact or
    ``unsure`` -- nothing in between.
  * The preferred key is the track's ISRC (an exact recording identity). The
    title/artist search fallback is weaker; in --match-strictness strict it is
    only accepted when the MB artist cross-references to the SAME Spotify artist.

Output matches the EXACT schema tools/evaluate_labels.py consumes, so
``tools/evaluate_labels.py <group>`` and ``tools/evaluate_dataset.py`` run
unchanged against the files this writes:
  * labels/labels_<group>.csv       -> columns: track_id, track_name,
        credited_artist_id, your_verdict, true_artist_id, reason, source
  * labels/answer_key_<group>.csv   -> the tool's flags (reuses
        tools/make_labels.build_answer_key so the schema is identical).

Usage: python tools/musicbrainz_labels.py GROUP [--match-strictness {strict,loose}]
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import data_source
from src.app_core import prepare_group
from tools.make_labels import build_answer_key

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Credentials/contact load from the PROJECT's PARENT directory .env, consistent
# with src/ingestion/ingestion.py (ENV_FILE = PROJECT_ROOT.parent / ".env").
load_dotenv(PROJECT_ROOT.parent / ".env")

MB_BASE_URL = "https://musicbrainz.org/ws/2"
DEFAULT_CONTACT = "music-disambiguation-eval/1.0 (contact)"
# MusicBrainz asks for at most 1 request/second; mirror ingestion.py pacing.
MIN_REQUEST_INTERVAL_SECONDS = 1.0
MAX_RETRIES = 5
REQUEST_TIMEOUT = 20
# Title/artist fallback: how close the MB recording length must be to Spotify's.
DURATION_TOLERANCE_MS = 3000

LABELS_DIR = Path("labels")
CACHE_DIR = LABELS_DIR / ".mb_cache"
SPOTIFY_ARTIST_URL_RE = re.compile(r"open\.spotify\.com/artist/([A-Za-z0-9]+)")

LABEL_COLUMNS = [
    "track_id",
    "track_name",
    "credited_artist_id",
    "your_verdict",
    "true_artist_id",
    "reason",
    "source",
]


def user_agent() -> str:
    """Descriptive User-Agent MusicBrainz requires, from MUSICBRAINZ_CONTACT."""
    contact = os.getenv("MUSICBRAINZ_CONTACT", "").strip()
    if not contact:
        return DEFAULT_CONTACT
    return f"music-disambiguation-eval/1.0 ({contact})"


class MusicBrainzClient:
    """Minimal MusicBrainz read client honoring MB's required conventions.

    Conventions: base https://musicbrainz.org/ws/2, fmt=json, no auth for reads,
    a descriptive User-Agent, >=1s pacing, and HTTP 503 treated as retryable
    (honoring Retry-After before capped backoff). Lookups are cached on disk
    under labels/.mb_cache/ keyed by ISRC and by MBID so re-runs are cheap.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent()})
        self._last_request_at = 0.0

    # -- pacing / HTTP --------------------------------------------------------
    def _wait_for_request_slot(self) -> None:
        """Pace requests to at most 1/sec, mirroring ingestion.py."""
        remaining = MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()

    @staticmethod
    def _retry_delay(response: requests.Response, attempt: int) -> float:
        """Honor Retry-After, else capped exponential backoff."""
        try:
            return max(float(response.headers["Retry-After"]), 1.0)
        except (KeyError, ValueError, TypeError):
            return min(2 ** attempt, 30)

    def _get(self, path: str, params: dict) -> dict | None:
        """GET a MusicBrainz JSON resource; return None on a 404 (not found)."""
        url = f"{MB_BASE_URL}/{path}"
        query = dict(params)
        query["fmt"] = "json"
        for attempt in range(MAX_RETRIES + 1):
            self._wait_for_request_slot()
            response = self.session.get(url, params=query, timeout=REQUEST_TIMEOUT)
            if response.status_code == 503 and attempt < MAX_RETRIES:
                delay = self._retry_delay(response, attempt)
                response.close()
                print(f"  MusicBrainz rate-limited (503) on {path}; waiting {delay:.1f}s...")
                time.sleep(delay)
                continue
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        raise RuntimeError(f"MusicBrainz GET {path} remained rate-limited after retries.")

    # -- on-disk cache --------------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
        return self.cache_dir / f"{safe}.json"

    # -- MB endpoints ---------------------------------------------------------
    def lookup_isrc(self, isrc: str) -> dict | None:
        """GET /isrc/<isrc>?inc=artist-credits : exact recording identity."""
        path = self._cache_path(f"isrc_{isrc}")
        if path.exists():
            return _read_json(path)
        data = self._get(f"isrc/{isrc}", {"inc": "artist-credits"})
        _write_json(path, data)
        return data

    def search_recording(self, title: str, artist_name: str) -> dict | None:
        """Search recordings by title + artist; weaker than an ISRC match."""
        query = f'recording:"{_lucene_escape(title)}" AND artist:"{_lucene_escape(artist_name)}"'
        return self._get("recording", {"query": query, "limit": 10})

    def artist_spotify_id(self, mbid: str) -> str | None:
        """GET /artist/<mbid>?inc=url-rels and return its Spotify artist id, if any."""
        path = self._cache_path(f"artist_{mbid}")
        if path.exists():
            cached = _read_json(path)
            return cached.get("spotify_artist_id") if isinstance(cached, dict) else None
        data = self._get(f"artist/{mbid}", {"inc": "url-rels"})
        spotify_id = None
        if data:
            for relation in data.get("relations", []):
                resource = (relation.get("url") or {}).get("resource", "")
                match = SPOTIFY_ARTIST_URL_RE.search(resource or "")
                if match:
                    spotify_id = match.group(1)
                    break
        _write_json(path, {"mbid": mbid, "spotify_artist_id": spotify_id})
        return spotify_id


# --- small JSON helpers (avoid importing json at module top for brevity) -----
def _read_json(path: Path):
    import json
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)


def _lucene_escape(text: str) -> str:
    """Escape Lucene special characters in a free-text search term."""
    return re.sub(r'([+\-&|!(){}\[\]^"~*?:\\/])', r"\\\1", str(text))


def _recording_artist_mbids(recording: dict) -> list[str]:
    """Pull the artist MBIDs from a recording's artist-credit list."""
    mbids = []
    for credit in recording.get("artist-credit", []) or []:
        artist = credit.get("artist") if isinstance(credit, dict) else None
        if artist and artist.get("id"):
            mbids.append(artist["id"])
    return mbids


def resolve_track(client, track, group_spotify_ids, strictness):
    """Resolve one track to a MusicBrainz-backed verdict.

    ``track`` is a dict with track_id, track_name, duration_ms, isrc, and
    credited_artist_id (the Spotify artist whose catalog this row came from).
    ``group_spotify_ids`` is the set of the group's Spotify artist ids. Returns
    (your_verdict, true_artist_id, reason, source):

      * correct       -> MB resolves this recording to the SAME Spotify artist.
      * misattributed -> MB resolves it to a DIFFERENT confirmed Spotify artist
                         (true_artist_id set to that id).
      * unsure        -> unresolved or low-confidence (excluded from scoring).
    """
    credited = str(track["credited_artist_id"])

    # --- Preferred: ISRC lookup (exact recording identity = TRUSTED) ---------
    isrc = track.get("isrc")
    if isinstance(isrc, str) and isrc.strip():
        isrc = isrc.strip()
        data = client.lookup_isrc(isrc)
        recordings = (data or {}).get("recordings") or []
        for recording in recordings:
            for mbid in _recording_artist_mbids(recording):
                spotify_id = client.artist_spotify_id(mbid)
                if not spotify_id:
                    continue
                reason = f"ISRC {isrc} -> MBID {mbid} -> spotify artist {spotify_id}"
                source = "musicbrainz GET /isrc/<isrc> + GET /artist/<mbid>"
                if spotify_id == credited:
                    return "correct", "", reason, source
                if spotify_id in group_spotify_ids:
                    return "misattributed", spotify_id, reason, source
                # Resolved, but to an artist outside the group: still a
                # confirmed different Spotify artist -> misattributed.
                return "misattributed", spotify_id, reason, source

    # --- Fallback: title + artist search (weaker, only if close) -------------
    title = str(track.get("track_name") or "")
    duration_ms = track.get("duration_ms")
    credited_name = str(track.get("credited_artist_name") or "")
    if title:
        search = client.search_recording(title, credited_name)
        for recording in (search or {}).get("recordings", []) or []:
            if str(recording.get("title", "")).strip().lower() != title.strip().lower():
                continue
            length = recording.get("length")
            if duration_ms is not None and length is not None:
                try:
                    if abs(float(length) - float(duration_ms)) > DURATION_TOLERANCE_MS:
                        continue
                except (TypeError, ValueError):
                    continue
            mbid_candidates = _recording_artist_mbids(recording)
            for mbid in mbid_candidates:
                spotify_id = client.artist_spotify_id(mbid)
                if not spotify_id:
                    continue
                reason = (f"title+duration search -> MBID {mbid} -> "
                          f"spotify artist {spotify_id}")
                source = "musicbrainz GET /recording?query + GET /artist/<mbid>"
                if spotify_id == credited:
                    return "correct", "", reason, source
                if spotify_id in group_spotify_ids:
                    return "misattributed", spotify_id, reason, source
                return "misattributed", spotify_id, reason, source
            # loose mode: accept a strong title+duration match even WITHOUT a
            # Spotify cross-reference (trades trust for coverage).
            if strictness == "loose" and mbid_candidates:
                reason = (f"loose title+duration match -> MBID "
                          f"{mbid_candidates[0]} (no Spotify cross-ref)")
                source = "musicbrainz GET /recording?query"
                return "correct", "", reason, source

    return ("unsure", "",
            "no trusted MusicBrainz match (ISRC/search unresolved)",
            "musicbrainz")


def build_labels(view, client, strictness):
    """Resolve every track in the group and return the labels DataFrame."""
    frame = data_source.load_group(view.name).frame
    group_spotify_ids = set(view.profiles) | set(frame["artist_id"].astype(str))

    label_rows = []
    total = len(frame)
    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        track = {
            "track_id": row["track_id"],
            "track_name": row["track_name"],
            "duration_ms": row.get("duration_ms"),
            "isrc": row.get("isrc"),
            "credited_artist_id": row["artist_id"],
            "credited_artist_name": row.get("artist_name"),
        }
        verdict, true_artist_id, reason, source = resolve_track(
            client, track, group_spotify_ids, strictness
        )
        print(f"  [{position}/{total}] {track['track_name']!r} -> {verdict}")
        label_rows.append({
            "track_id": row["track_id"],
            "track_name": row["track_name"],
            "credited_artist_id": row["artist_id"],
            "your_verdict": verdict,
            "true_artist_id": true_artist_id,
            "reason": reason,
            "source": source,
        })
    return pd.DataFrame(label_rows, columns=LABEL_COLUMNS)


def run(group, strictness):
    """Label one group from MusicBrainz facts and write both CSVs."""
    try:
        view = prepare_group(group)
    except FileNotFoundError:
        print(f"No catalog CSV for group {group!r} "
              f"(expected data/spotify_tracks_{group}.csv). "
              "Run src.ingestion.ingestion first; skipping.")
        return 1

    LABELS_DIR.mkdir(exist_ok=True)
    client = MusicBrainzClient()

    print(f"Resolving {group!r} against MusicBrainz "
          f"(strictness={strictness}, contact={user_agent()!r})...")
    labels = build_labels(view, client, strictness)
    labels_path = LABELS_DIR / f"labels_{group}.csv"
    labels.to_csv(labels_path, index=False)

    answer_key = build_answer_key(view)
    key_path = LABELS_DIR / f"answer_key_{group}.csv"
    answer_key.to_csv(key_path, index=False)

    counts = labels["your_verdict"].value_counts().to_dict()
    print(f"\nGroup: {group}")
    print(f"  tracks labeled: {len(labels)}  ({counts})")
    print(f"Wrote {labels_path}")
    print(f"Wrote {key_path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Label a same-name group's tracks from MusicBrainz facts "
                    "(trustworthy ground truth; never LLM-fabricated).",
    )
    parser.add_argument("group", help="group/artist name, e.g. WITCH")
    parser.add_argument(
        "--match-strictness",
        choices=["strict", "loose"],
        default="strict",
        help="strict (default): title/artist search matches require a "
             "MB->Spotify cross-reference. loose: also accept high-confidence "
             "title/duration matches WITHOUT the Spotify-link cross-reference "
             "(trades trust for coverage).",
    )
    args = parser.parse_args(argv)
    return run(args.group, args.match_strictness)


if __name__ == "__main__":
    sys.exit(main())
