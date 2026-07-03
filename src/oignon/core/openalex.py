"""OpenAlex API client for fetching paper data."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyalex
from pyalex import Works

from oignon.core.formats import (
    extract_id,
    format_full_paper,
    format_slim_paper,
)
from oignon.core.graph import FullPaper, SlimPaper
from oignon.core.ratelimit import (
    OpenAlexRateLimitError,
    is_rate_limit,
    throttle,
)

# Configure pyalex: polite pool, NO automatic retries. With any nonzero
# retry count, urllib3 honors OpenAlex's Retry-After header on 429s with
# an uncapped sleep (even when 429 is not in the retry list), which can
# hang a tool call for minutes during a hard block. Our own throttle()
# prevents self-inflicted 429s; a real one fails fast and surfaces as
# OpenAlexRateLimitError so the agent knows to wait and retry.
pyalex.config.email = os.environ.get("OPENALEX_EMAIL", "")
pyalex.config.max_retries = 0

# API limits
OPENALEX_MAX_PER_PAGE = 200
OPENALEX_MAX_FILTER_IDS = 100
MAX_PARALLEL_REQUESTS = 10
DEFAULT_BRANCH_SEEDS_LIMIT = 200

# Fields for slim fetches (ranking only)
SLIM_FIELDS = ["id", "publication_year", "cited_by_count", "referenced_works"]

# Fields for full fetches (final papers)
FULL_FIELDS = [
    "id",
    "doi",
    "title",
    "authorships",
    "publication_year",
    "cited_by_count",
    "referenced_works",
    "type",
    "language",
    "open_access",
    "primary_location",
    "abstract_inverted_index",
    "fwci",
    "citation_normalized_percentile",
    "primary_topic",
    "sustainable_development_goals",
    "keywords",
]


def chunk(items: list, size: int) -> list[list]:
    """Split list into chunks of given size."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def _fetch_batch_slim(batch: list[str]) -> dict[str, SlimPaper]:
    """Fetch a batch of papers with slim fields."""
    papers = {}
    id_filter = "|".join(batch)

    throttle()
    try:
        results = (
            Works()
            .filter(openalex=id_filter)
            .select(SLIM_FIELDS)
            .get(per_page=OPENALEX_MAX_PER_PAGE)
        )
        for work in results:
            paper = format_slim_paper(work)
            papers[paper.id] = paper
    except Exception as e:
        if is_rate_limit(e):
            raise OpenAlexRateLimitError() from e

    return papers


def fetch_papers_slim(
    work_ids: list[str], parallel: bool = True
) -> dict[str, SlimPaper]:
    """Fetch multiple papers with slim fields for ranking."""
    if not work_ids:
        return {}

    batches = chunk(work_ids, OPENALEX_MAX_FILTER_IDS)
    papers = {}

    if parallel and len(batches) > 1:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS) as executor:
            futures = [executor.submit(_fetch_batch_slim, b) for b in batches]
            for future in as_completed(futures):
                papers.update(future.result())
    else:
        for batch in batches:
            papers.update(_fetch_batch_slim(batch))

    return papers


def _fetch_batch_full(batch: list[str]) -> dict[str, FullPaper]:
    """Fetch a batch of papers with full fields."""
    papers = {}
    id_filter = "|".join(batch)

    throttle()
    try:
        results = (
            Works()
            .filter(openalex=id_filter)
            .select(FULL_FIELDS)
            .get(per_page=OPENALEX_MAX_PER_PAGE)
        )
        for work in results:
            paper = format_full_paper(work)
            papers[paper.id] = paper
    except Exception as e:
        if is_rate_limit(e):
            raise OpenAlexRateLimitError() from e

    return papers


def fetch_papers_full(
    work_ids: list[str], parallel: bool = True
) -> dict[str, FullPaper]:
    """Fetch multiple papers with full fields for final graph."""
    if not work_ids:
        return {}

    batches = chunk(work_ids, OPENALEX_MAX_FILTER_IDS)
    papers = {}

    if parallel and len(batches) > 1:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS) as executor:
            futures = [executor.submit(_fetch_batch_full, b) for b in batches]
            for future in as_completed(futures):
                papers.update(future.result())
    else:
        for batch in batches:
            papers.update(_fetch_batch_full(batch))

    return papers


def fetch_paper(work_id: str) -> FullPaper | None:
    """Fetch a single paper with full fields."""
    work_id = extract_id(work_id)

    # Handle DOI input
    if work_id.startswith("10."):
        work_id = f"https://doi.org/{work_id}"

    throttle()
    try:
        work = Works()[work_id]
        return format_full_paper(work)
    except Exception as e:
        if is_rate_limit(e):
            raise OpenAlexRateLimitError() from e
        return None


def fetch_citing_papers(
    work_id: str, limit: int = DEFAULT_BRANCH_SEEDS_LIMIT
) -> list[str]:
    """Fetch IDs of papers that cite the given work."""
    throttle()
    try:
        results = Works().filter(cites=work_id).select(["id"]).get(per_page=limit)
        return [extract_id(w.get("id")) for w in results]
    except Exception as e:
        if is_rate_limit(e):
            raise OpenAlexRateLimitError() from e
        return []


def search_papers(query: str, limit: int = 10) -> list[FullPaper]:
    """Search OpenAlex for papers matching the query."""
    throttle()
    try:
        results = Works().search(query).get(per_page=limit)
        return [format_full_paper(w) for w in results]
    except Exception as e:
        if is_rate_limit(e):
            raise OpenAlexRateLimitError() from e
        return []
