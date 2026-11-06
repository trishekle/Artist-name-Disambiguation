"""Spotify catalog ingestion keyed by verified Spotify artist IDs.

Spotify artist names are display labels, not identifiers. Normal runs search
for one exact artist name, then ingest every exact-name artist ID returned by
Spotify Search.

Spotify metadata is used only for identity and catalog links here. Do not
download Spotify audio or pass Spotify Content to an ML/AI model.
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT.parent / ".env"
ARTISTS_FILE = PROJECT_ROOT / "data" / "same_name_artists.json"
API_BASE_URL = "https://api.spotify.com/v1"
SEARCH_NAME = "Northland"  # Change this name, or pass --artist-name on the command line.
MAX_RATE_LIMIT_RETRIES = 5
MIN_REQUEST_INTERVAL_SECONDS = 1.0
ALBUM_LIMIT = 5
ALBUM_TRACK_LIMIT = 50
TRACK_COLUMNS = [
    "track_id",
    "track_name",
    "artist_id",
    "artist_name",
    "album_id",
    "album_name",
    "duration_ms",
    "isrc",
    "explicit",
    "track_number",
    "disc_number",
    "album_type",
    "album_release_date",
    "album_total_tracks",
    "credited_artist_ids",
    "credited_artist_names",
]

load_dotenv(ENV_FILE)
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
last_request_at = 0.0
headers: dict[str, str] = {}


def raise_for_spotify_error(response: requests.Response, action: str) -> None:
    """Raise an error that includes Spotify's useful error payload."""
    if response.ok:
        return
    try:
        detail = response.json()
    except ValueError:
        detail = response.text
    raise RuntimeError(f"Spotify {action} failed ({response.status_code}): {detail}")


def wait_for_request_slot() -> None:
    """Pace requests so development runs do not burst into a 429."""
    global last_request_at
    remaining = MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - last_request_at)
    if remaining > 0:
        time.sleep(remaining)
    last_request_at = time.monotonic()


def rate_limit_delay(response: requests.Response, attempt: int) -> float:
    """Honor Retry-After, falling back to capped exponential backoff plus jitter."""
    try:
        return max(float(response.headers["Retry-After"]), 1.0)
    except (KeyError, ValueError):
        return min(2**attempt, 30) + random.uniform(0, 0.5)


def get_access_token() -> str:
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError(
            "Spotify credentials were not loaded. Add SPOTIFY_CLIENT_ID and "
            f"SPOTIFY_CLIENT_SECRET to {ENV_FILE}."
        )
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        wait_for_request_slot()
        response = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET),
            timeout=20,
        )
        if response.status_code == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
            delay = rate_limit_delay(response, attempt)
            response.close()
            print(f"Spotify token endpoint rate-limited; waiting {delay:.1f}s...")
            time.sleep(delay)
            continue
        raise_for_spotify_error(response, "token request")
        return response.json()["access_token"]
    raise RuntimeError("Spotify token request failed after retries.")


def authenticate() -> None:
    """Set the authorization header lazily, so this module is safe to import."""
    headers["Authorization"] = f"Bearer {get_access_token()}"


def spotify_get(url: str, params: dict | None = None) -> dict:
    """GET Spotify JSON, retrying 429 responses with Retry-After/backoff."""
    if not headers:
        authenticate()
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        wait_for_request_slot()
        response = requests.get(url, headers=headers, params=params, timeout=20)
        if response.status_code == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
            delay = rate_limit_delay(response, attempt)
            response.close()
            print(f"Spotify rate-limited on {url}; waiting {delay:.1f}s...")
            time.sleep(delay)
            continue
        raise_for_spotify_error(response, f"GET {url}")
        return response.json()
    raise RuntimeError(f"Spotify GET {url} remained rate-limited after retries.")


def normalized_artist_name(name: str) -> str:
    return name.strip().casefold()


def search_exact_artists(search_name: str) -> list[dict]:
    """Return exact-name candidates from one Search response, not an exhaustive list."""
    response = spotify_get(
        f"{API_BASE_URL}/search", {"q": search_name, "type": "artist", "limit": 10}
    )
    expected_name = normalized_artist_name(search_name)
    candidates = []
    for artist in response["artists"]["items"]:
        if normalized_artist_name(artist["name"]) == expected_name:
            candidates.append(
                {
                    "id": artist["id"],
                    "name": artist["name"],
                    "spotify_url": artist.get("external_urls", {}).get("spotify"),
                }
            )
    return candidates


def load_verified_artists(path: Path = ARTISTS_FILE) -> list[dict]:
    """Load manually verified identities without making a Spotify search request."""
    with path.open(encoding="utf-8") as config_file:
        artist_groups = json.load(config_file)
    artists = []
    for group_name, group_artists in artist_groups.items():
        for artist in group_artists:
            if {"id", "name"} - artist.keys():
                raise ValueError(f"Invalid artist entry for {group_name!r}: {artist!r}")
            if normalized_artist_name(artist["name"]) != normalized_artist_name(group_name):
                raise ValueError(
                    f"Configured artist {artist['id']} is named {artist['name']!r}, "
                    f"not the {group_name!r} group."
                )
            artists.append({"id": artist["id"], "name": artist["name"]})
    return list({artist["id"]: artist for artist in artists}.values())


def get_artist_tracks(artist_id: str, artist_name: str) -> pd.DataFrame:
    """Fetch one small album page for one artist ID; no deep pagination."""
    album_page = spotify_get(
        f"{API_BASE_URL}/artists/{artist_id}/albums",
        {"include_groups": "album,single", "limit": ALBUM_LIMIT},
    )
    unique_albums = {album["id"]: album for album in album_page["items"]}
    print(f"  Albums sampled: {len(unique_albums)}")
    tracks = []
    for album in unique_albums.values():
        track_page = spotify_get(
            f"{API_BASE_URL}/albums/{album['id']}/tracks", {"limit": ALBUM_TRACK_LIMIT}
        )
        for track in track_page["items"]:
            if not any(item["id"] == artist_id for item in track.get("artists", [])):
                continue
            credited_artists = track.get("artists", [])
            tracks.append(
                {
                    "track_id": track["id"],
                    "track_name": track["name"],
                    "artist_id": artist_id,
                    "artist_name": artist_name,
                    "album_id": album["id"],
                    "album_name": album["name"],
                    "duration_ms": track.get("duration_ms"),
                    "isrc": None,
                    "explicit": track.get("explicit"),
                    "track_number": track.get("track_number"),
                    "disc_number": track.get("disc_number"),
                    "album_type": album.get("album_type"),
                    "album_release_date": album.get("release_date"),
                    "album_total_tracks": album.get("total_tracks"),
                    "credited_artist_ids": "|".join(
                        item["id"] for item in credited_artists if item.get("id")
                    ),
                    "credited_artist_names": "|".join(
                        item["name"] for item in credited_artists if item.get("name")
                    ),
                }
            )
    return pd.DataFrame(tracks, columns=TRACK_COLUMNS)


def enrich_isrcs(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """Populate the isrc column from full track objects (GET /v1/tracks/{id}).

    The album-tracks endpoint omits external_ids, so ISRCs are fetched in a
    second pass, one track per request, and mapped back onto rows by track_id.
    The batch endpoint (GET /v1/tracks?ids=...) returns 403 for this app, so
    single-track calls are used instead. Reuses spotify_get, so this pass
    inherits wait_for_request_slot pacing and 429 Retry-After/backoff retries.
    A track that fails to fetch keeps isrc=None rather than aborting the run.
    """
    if tracks_df.empty:
        return tracks_df
    track_ids = [tid for tid in tracks_df["track_id"].dropna().unique().tolist()]
    isrc_by_track_id: dict[str, str] = {}
    print(f"\nFetching ISRCs for {len(track_ids)} tracks (1 request each)...")
    for track_id in track_ids:
        try:
            track = spotify_get(f"{API_BASE_URL}/tracks/{track_id}")
        except RuntimeError as error:
            print(f"  ISRC lookup failed for {track_id}: {error}")
            continue
        isrc = (track or {}).get("external_ids", {}).get("isrc")
        if isrc:
            isrc_by_track_id[track_id] = isrc
    if isrc_by_track_id:
        tracks_df["isrc"] = tracks_df["track_id"].map(isrc_by_track_id).fillna(tracks_df["isrc"])
    return tracks_df


def ingest_verified_artists(artists: list[dict]) -> pd.DataFrame:
    if not artists:
        raise RuntimeError(f"No verified artists configured in {ARTISTS_FILE}.")
    artist_frames = []
    for artist in artists:
        print(f"\nCollecting: {artist['name']} ({artist['id']})")
        artist_df = get_artist_tracks(artist["id"], artist["name"])
        print(f"  Tracks found: {len(artist_df)}")
        artist_frames.append(artist_df)
    tracks_df = pd.concat(artist_frames, ignore_index=True).drop_duplicates(subset="track_id")
    return enrich_isrcs(tracks_df)


def output_file_for_artists(artists: list[dict]) -> Path:
    """Build ``spotify_tracks_{artist_name}.csv`` for one same-name group."""
    names = {normalized_artist_name(artist["name"]): artist["name"] for artist in artists}
    if len(names) != 1:
        raise ValueError(
            "Write one same-name group per run so the output file can use its artist name."
        )
    artist_name = next(iter(names.values()))
    safe_name = re.sub(r"[^\w.-]+", "_", artist_name, flags=re.UNICODE).strip("._")
    return PROJECT_ROOT / "data" / f"spotify_tracks_{safe_name or 'artist'}.csv"


def print_summary(tracks_df: pd.DataFrame) -> None:
    print("\n==============================\nDATASET SUMMARY\n==============================")
    print(tracks_df.groupby(["artist_name", "artist_id"]).size())
    print("Unique artist IDs:", tracks_df["artist_id"].nunique())
    print("Unique artist names:", tracks_df["artist_name"].nunique())
    print("Tracks:", len(tracks_df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Spotify metadata by verified artist ID.")
    parser.add_argument("--discover", metavar="NAME", help="one-shot exact-name candidate discovery")
    parser.add_argument(
        "--artist-name",
        help="search this exact display name and ingest every matching returned artist ID",
    )
    parser.add_argument(
        "--use-verified-config",
        action="store_true",
        help="ingest IDs from data/same_name_artists.json instead of searching",
    )
    args = parser.parse_args()
    if args.discover:
        candidates = search_exact_artists(args.discover)
        print(f"Exact-name candidates returned for {args.discover!r} (not exhaustive):")
        for artist in candidates:
            print(f"name={artist['name']!r} | id={artist['id']!r} | url={artist['spotify_url']!r}")
        return
    if args.use_verified_config:
        artists = load_verified_artists()
    else:
        search_name = args.artist_name or SEARCH_NAME
        artists = search_exact_artists(search_name)
        if not artists:
            raise RuntimeError(f"No exact-name artist candidates returned for {search_name!r}.")
        print(f"Found {len(artists)} exact-name artist ID(s) for {search_name!r}.")
    tracks_df = ingest_verified_artists(artists)
    print_summary(tracks_df)
    output_file = output_file_for_artists(artists)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tracks_df.to_csv(output_file, index=False)
    print(f"\nSaved dataset to: {output_file}")


if __name__ == "__main__":
    main()
