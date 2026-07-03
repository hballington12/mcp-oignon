"""Multi-strategy paper search against OpenAlex.

OpenAlex's default relevance search is weak for title/author lookups:
full-text matching drowns out exact title hits, and author names in the
query string just add noise. So we run several narrow queries in parallel
and fuse the rankings (reciprocal rank fusion), then boost results whose
title closely matches the query or whose authors match the author hint.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

from pyalex import Works

from oignon.core.graph import FullPaper
from oignon.core.openalex import (
    FULL_FIELDS,
    fetch_paper,
    format_full_paper,
)

RESULTS_PER_STRATEGY = 25
RRF_K = 60  # standard reciprocal-rank-fusion constant
TITLE_SIM_WEIGHT = 0.03
AUTHOR_MATCH_WEIGHT = 0.02

OPENALEX_ID_RE = re.compile(r"[Ww]\d{4,}")
DOI_RE = re.compile(r"10\.\d{4,9}/\S+")


def detect_work_id(text: str) -> str | None:
    """If the query is (or contains) a DOI or OpenAlex ID, extract it."""
    t = text.strip()

    if OPENALEX_ID_RE.fullmatch(t):
        return t.upper()

    if "doi.org/" in t:
        return t.split("doi.org/")[-1].strip()

    m = DOI_RE.search(t)
    if m:
        return m.group(0).rstrip(".,;)")

    return None


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _title_similarity(query: str, title: str) -> float:
    """Fuzzy similarity between query and title, 0..1."""
    q = _normalize(query)
    t = _normalize(title)
    if not q or not t:
        return 0.0

    ratio = SequenceMatcher(None, q, t).ratio()

    # Token containment: all query words appearing in the title is a
    # strong signal even when the title has extra words.
    q_tokens = set(q.split())
    t_tokens = set(t.split())
    containment = len(q_tokens & t_tokens) / len(q_tokens) if q_tokens else 0.0

    return max(ratio, containment)


def _author_match(author: str, paper: FullPaper) -> float:
    """Fraction of author-name tokens found among the paper's authors."""
    tokens = [t for t in _normalize(author).split() if len(t) > 2]
    if not tokens:
        return 0.0

    names = _normalize(" ".join(a.name for a in paper.authors))
    matched = sum(1 for t in tokens if t in names.split())
    return matched / len(tokens)


def _apply_year(works: Works, year: str) -> Works:
    """Apply a year filter: '2015' (exact) or '2010-2020' (range)."""
    year = year.strip()
    if not year:
        return works

    range_match = re.fullmatch(r"(\d{4})\s*-\s*(\d{4})", year)
    if range_match:
        start, end = range_match.groups()
        return works.filter(
            from_publication_date=f"{start}-01-01",
            to_publication_date=f"{end}-12-31",
        )

    if re.fullmatch(r"\d{4}", year):
        return works.filter(publication_year=int(year))

    return works


def _run_strategy(build_query) -> list[FullPaper]:
    """Execute one search strategy, returning [] on any API error."""
    try:
        results = build_query().select(FULL_FIELDS).get(per_page=RESULTS_PER_STRATEGY)
        return [format_full_paper(w) for w in results]
    except Exception:
        return []


def _build_strategies(query: str, author: str, year: str) -> dict:
    """Assemble named strategy thunks based on which inputs are present."""
    strategies = {}

    if query:
        strategies["relevance"] = lambda: _apply_year(Works().search(query), year)
        strategies["title"] = lambda: _apply_year(
            Works().filter(title={"search": query}), year
        )

    if author:
        strategies["author"] = lambda: _apply_year(
            Works().filter(raw_author_name={"search": author}), year
        ).sort(cited_by_count="desc")

    if query and author:
        strategies["query+author"] = lambda: _apply_year(
            Works().search(query).filter(raw_author_name={"search": author}), year
        )

    return strategies


def find_papers(
    query: str = "",
    author: str = "",
    year: str = "",
    limit: int = 10,
) -> list[dict]:
    """Search OpenAlex with multiple strategies and fuse the rankings.

    Returns result dicts sorted by fused score, with enough metadata
    (venue, topic, DOI) for an agent to disambiguate similar papers.
    """
    query = query.strip()
    author = author.strip()

    # Direct lookup when the query is already an identifier
    work_id = detect_work_id(query) if query else None
    if work_id:
        paper = fetch_paper(work_id)
        return [_to_result(paper, 1.0, ["direct_id"])] if paper else []

    strategies = _build_strategies(query, author, year)
    if not strategies:
        return []

    with ThreadPoolExecutor(max_workers=len(strategies)) as executor:
        futures = {
            name: executor.submit(_run_strategy, thunk)
            for name, thunk in strategies.items()
        }
        ranked_lists = {name: f.result() for name, f in futures.items()}

    # Reciprocal rank fusion across strategies
    papers: dict[str, FullPaper] = {}
    scores: dict[str, float] = {}
    matched: dict[str, list[str]] = {}

    for name, results in ranked_lists.items():
        for rank, paper in enumerate(results):
            papers[paper.id] = paper
            scores[paper.id] = scores.get(paper.id, 0.0) + 1 / (RRF_K + rank + 1)
            matched.setdefault(paper.id, []).append(name)

    # Boosts: fuzzy title match and author-name match
    for pid, paper in papers.items():
        if query:
            scores[pid] += TITLE_SIM_WEIGHT * _title_similarity(query, paper.title)
        if author:
            scores[pid] += AUTHOR_MATCH_WEIGHT * _author_match(author, paper)

    top = sorted(scores, key=lambda pid: scores[pid], reverse=True)[:limit]
    return [_to_result(papers[pid], scores[pid], matched[pid]) for pid in top]


def _to_result(paper: FullPaper, score: float, strategies: list[str]) -> dict:
    """Format a paper as a search-result dict for the agent."""
    return {
        "id": paper.id,
        "title": paper.title,
        "authors": [a.name for a in paper.authors[:3]],
        "year": paper.year,
        "citations": paper.citation_count,
        "venue": paper.source_name,
        "doi": paper.doi,
        "topic": paper.primary_topic.name if paper.primary_topic else None,
        "score": round(score, 4),
        "matched_by": strategies,
    }
