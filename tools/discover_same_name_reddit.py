"""Reddit discovery HINT for same-name artist confusion -- NEVER a label source.

Read this before using the output:

    This tool is a DISCOVERY HINT ONLY. It surfaces band/artist names that
    Reddit threads mention as same-name collisions, so you have somewhere to
    point Spotify ingestion. It is NOT ground truth and NOT a label source.
    Nothing it prints is a verdict about whether a track is misattributed.
    Labels come only from MusicBrainz facts (tools/musicbrainz_labels.py) or
    stay "unsure" -- never from Reddit, and never fabricated by an LLM.

It uses the official Reddit API (OAuth script app, client-credentials grant);
it does NO HTML scraping. Credentials load from the PROJECT's PARENT directory
.env, consistent with src/ingestion/ingestion.py:

    REDDIT_CLIENT_ID
    REDDIT_CLIENT_SECRET
    REDDIT_USER_AGENT

It searches a few music subreddits for same-name confusion threads, extracts
candidate names plus the source thread URL (provenance), prints them, and -- only
with --write -- appends them to data/reddit_name_candidates.txt (git-ignored).

Usage: python tools/discover_same_name_reddit.py [--limit N] [--write]
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Credentials load from the PROJECT's PARENT directory .env, consistent with
# src/ingestion/ingestion.py (ENV_FILE = PROJECT_ROOT.parent / ".env").
ENV_FILE = PROJECT_ROOT.parent / ".env"
load_dotenv(ENV_FILE)

CANDIDATES_FILE = PROJECT_ROOT / "data" / "reddit_name_candidates.txt"

REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_OAUTH_BASE = "https://oauth.reddit.com"
# Subreddits where same-name artist confusion gets discussed.
SUBREDDITS = ["musicbrainz", "Music", "ifyoulikeblank", "WeAreTheMusicMakers"]
# Queries aimed at same-name collision threads (discovery hints, not labels).
SEARCH_QUERIES = [
    "two bands same name",
    "artist name collision",
    "wrong artist on Spotify",
]
# MusicBrainz/Spotify-style politeness: at most ~1 request/second, like
# src/ingestion/ingestion.py.
MIN_REQUEST_INTERVAL_SECONDS = 1.0
MAX_RATE_LIMIT_RETRIES = 5
REQUEST_TIMEOUT = 20
SEARCH_LIMIT = 25

_last_request_at = 0.0

# Candidate-name extraction: quoted names and "two bands named X" patterns.
_QUOTED = re.compile(r"[\"\u201c\u201d\u2018\u2019']([^\"\u201c\u201d\u2018\u2019']{2,60})['\"\u201c\u201d\u2018\u2019]")
_NAMED = re.compile(
    r"\b(?:bands?|artists?|groups?)\s+(?:both\s+)?(?:called|named)\s+([A-Z][\w'&.\- ]{1,40})",
    re.IGNORECASE,
)


def missing_credentials():
    """Return the names of any required Reddit credentials that are not set."""
    required = ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"]
    return [name for name in required if not os.getenv(name)]


def print_setup_instructions(missing):
    """Explain how to configure Reddit API credentials (no traceback)."""
    print("Reddit API credentials are not configured.", file=sys.stderr)
    print(f"Missing: {', '.join(missing)}", file=sys.stderr)
    print("", file=sys.stderr)
    print("This tool uses the official Reddit API (OAuth), never HTML scraping.", file=sys.stderr)
    print("To enable it:", file=sys.stderr)
    print("  1. Create a Reddit 'script' app at https://www.reddit.com/prefs/apps", file=sys.stderr)
    print("     (choose type 'script'; a redirect URI like http://localhost is fine).", file=sys.stderr)
    print("  2. Note the client id (under the app name) and the secret.", file=sys.stderr)
    print(f"  3. Add these lines to the .env file at {ENV_FILE}:", file=sys.stderr)
    print("       REDDIT_CLIENT_ID=your_client_id", file=sys.stderr)
    print("       REDDIT_CLIENT_SECRET=your_client_secret", file=sys.stderr)
    print("       REDDIT_USER_AGENT=music-disambiguation-discovery/1.0 (by u/yourname)", file=sys.stderr)
    print("  4. Re-run this command.", file=sys.stderr)


def _wait_for_request_slot():
    """Pace requests so runs do not burst into a 429, like ingestion.py."""
    global _last_request_at
    remaining = MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
    if remaining > 0:
        time.sleep(remaining)
    _last_request_at = time.monotonic()


def _rate_limit_delay(response, attempt):
    """Honor Retry-After, falling back to capped exponential backoff."""
    try:
        return max(float(response.headers["Retry-After"]), 1.0)
    except (KeyError, ValueError):
        return min(2**attempt, 30) + 0.5


def get_access_token(client_id, client_secret, user_agent):
    """Obtain an app-only OAuth token via the client-credentials grant."""
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        _wait_for_request_slot()
        response = requests.post(
            REDDIT_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            headers={"User-Agent": user_agent},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
            delay = _rate_limit_delay(response, attempt)
            response.close()
            print(f"Reddit token endpoint rate-limited; waiting {delay:.1f}s...")
            time.sleep(delay)
            continue
        if not response.ok:
            raise RuntimeError(
                f"Reddit token request failed ({response.status_code}): {response.text}"
            )
        return response.json()["access_token"]
    raise RuntimeError("Reddit token request failed after retries.")


def reddit_get(path, token, user_agent, params=None):
    """GET Reddit OAuth JSON, retrying 429s with Retry-After/backoff."""
    headers = {"Authorization": f"bearer {token}", "User-Agent": user_agent}
    url = f"{REDDIT_OAUTH_BASE}{path}"
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        _wait_for_request_slot()
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
            delay = _rate_limit_delay(response, attempt)
            response.close()
            print(f"Reddit rate-limited on {path}; waiting {delay:.1f}s...")
            time.sleep(delay)
            continue
        if not response.ok:
            raise RuntimeError(f"Reddit GET {path} failed ({response.status_code}): {response.text}")
        return response.json()
    raise RuntimeError(f"Reddit GET {path} remained rate-limited after retries.")


def extract_candidate_names(text):
    """Pull candidate band/artist names out of a thread title or body (best-effort)."""
    if not text:
        return []
    names = []
    for match in _NAMED.finditer(text):
        names.append(match.group(1).strip(" .'\"-"))
    for match in _QUOTED.finditer(text):
        names.append(match.group(1).strip())
    # Dedupe, keep order, drop empties.
    seen = []
    for name in names:
        if name and name not in seen:
            seen.append(name)
    return seen


def search_candidates(token, user_agent, limit=SEARCH_LIMIT):
    """Search the music subreddits and return (name, source_thread_url) hints."""
    results = []
    seen = set()
    for subreddit in SUBREDDITS:
        for query in SEARCH_QUERIES:
            payload = reddit_get(
                f"/r/{subreddit}/search",
                token,
                user_agent,
                params={
                    "q": query,
                    "restrict_sr": "true",
                    "sort": "relevance",
                    "limit": limit,
                },
            )
            for child in payload.get("data", {}).get("children", []):
                post = child.get("data", {})
                permalink = post.get("permalink")
                thread_url = f"https://www.reddit.com{permalink}" if permalink else post.get("url")
                text = " ".join(filter(None, [post.get("title"), post.get("selftext")]))
                for name in extract_candidate_names(text):
                    key = (name.casefold(), thread_url)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append((name, thread_url))
    return results


def write_candidates(candidates, path=CANDIDATES_FILE):
    """Append candidates to the git-ignored candidates file (provenance kept)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for name, thread_url in candidates:
            handle.write(f"{name}\t{thread_url}\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Reddit discovery HINT for same-name artist collisions "
            "(NOT a label source; official Reddit API, no scraping)."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=SEARCH_LIMIT,
        help="max results per subreddit/query (default: %(default)s)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"also append candidates to {CANDIDATES_FILE} (git-ignored)",
    )
    args = parser.parse_args(argv)

    print("=" * 72)
    print("DISCOVERY HINT ONLY -- NOT ground truth, NOT a label source.")
    print("Reddit names are leads for Spotify ingestion; labels come from")
    print("MusicBrainz facts (or 'unsure'), never from Reddit or an LLM.")
    print("=" * 72)

    missing = missing_credentials()
    if missing:
        print_setup_instructions(missing)
        sys.exit(1)

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    token = get_access_token(client_id, client_secret, user_agent)
    candidates = search_candidates(token, user_agent, limit=args.limit)

    if not candidates:
        print("No candidate names surfaced from the searched threads.")
        return

    print(f"\n{len(candidates)} candidate name hint(s) (name <- source thread):")
    for name, thread_url in candidates:
        print(f"  {name}\t{thread_url}")

    if args.write:
        path = write_candidates(candidates)
        print(f"\nAppended {len(candidates)} hint(s) to {path}")
    else:
        print("\n(Re-run with --write to append these hints to "
              f"{CANDIDATES_FILE}.)")


if __name__ == "__main__":
    main()
