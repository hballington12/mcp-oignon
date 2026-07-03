"""Convert raw OpenAlex work dicts into paper dataclasses."""

from oignon.core.graph import (
    Author,
    CitationPercentile,
    FullPaper,
    PrimaryTopic,
    SDG,
    SlimPaper,
)

MAX_AUTHORS_IN_PAPER = 5


def extract_id(openalex_url: str | None) -> str:
    """Extract work ID from OpenAlex URL."""
    if not openalex_url:
        return ""
    if openalex_url.startswith("https://openalex.org/"):
        return openalex_url.split("/")[-1]
    return openalex_url


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct abstract from OpenAlex inverted index."""
    if not inverted_index:
        return ""
    words = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((word, pos))
    words.sort(key=lambda x: x[1])
    return " ".join(w for w, _ in words)


def format_slim_paper(work: dict) -> SlimPaper:
    """Format OpenAlex work into slim paper for ranking."""
    return SlimPaper(
        id=extract_id(work.get("id")),
        year=work.get("publication_year") or 0,
        citation_count=work.get("cited_by_count") or 0,
        references=[extract_id(r) for r in work.get("referenced_works", [])],
    )


def format_full_paper(work: dict) -> FullPaper:
    """Format OpenAlex work into full paper for final graph."""
    refs = work.get("referenced_works", [])
    authorships = work.get("authorships", [])

    # Parse authors
    authors = []
    for auth in authorships[:MAX_AUTHORS_IN_PAPER]:
        author = Author(
            name=auth.get("author", {}).get("display_name", ""),
            orcid=auth.get("author", {}).get("orcid"),
        )
        institutions = auth.get("institutions", [])
        if institutions:
            author.affiliation = institutions[0].get("display_name")
            author.affiliation_country = institutions[0].get("country_code")
        authors.append(author)

    # Parse primary topic
    primary_topic = None
    pt = work.get("primary_topic")
    if pt and pt.get("display_name"):
        primary_topic = PrimaryTopic(
            id=pt.get("id", ""),
            name=pt.get("display_name", ""),
            subfield={
                "id": pt.get("subfield", {}).get("id", ""),
                "name": pt.get("subfield", {}).get("display_name", ""),
            }
            if pt.get("subfield")
            else None,
            field={
                "id": pt.get("field", {}).get("id", ""),
                "name": pt.get("field", {}).get("display_name", ""),
            }
            if pt.get("field")
            else None,
            domain={
                "id": pt.get("domain", {}).get("id", ""),
                "name": pt.get("domain", {}).get("display_name", ""),
            }
            if pt.get("domain")
            else None,
        )

    # Parse citation percentile
    citation_percentile = None
    cp = work.get("citation_normalized_percentile")
    if cp and cp.get("value") is not None:
        value = cp.get("value")
        citation_percentile = CitationPercentile(
            value=value,
            is_in_top_1_percent=cp.get("is_in_top_1_percent") or value >= 99,
            is_in_top_10_percent=cp.get("is_in_top_10_percent") or value >= 90,
        )

    # Parse SDGs
    sdgs = None
    raw_sdgs = work.get("sustainable_development_goals", [])
    if raw_sdgs:
        sdgs = [
            SDG(
                id=s.get("id", ""),
                name=s.get("display_name", ""),
                score=s.get("score", 0),
            )
            for s in raw_sdgs
            if s.get("display_name") and s.get("score") is not None
        ]

    # Parse keywords
    keywords = None
    raw_kw = work.get("keywords", [])
    if raw_kw:
        keywords = [k.get("keyword") for k in raw_kw if k.get("keyword")]

    # Safely extract nested location fields
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    open_access = work.get("open_access") or {}

    return FullPaper(
        id=extract_id(work.get("id")),
        doi=work.get("doi"),
        title=work.get("title", ""),
        authors=authors,
        year=work.get("publication_year") or 0,
        citation_count=work.get("cited_by_count") or 0,
        references_count=len(refs),
        references=[extract_id(r) for r in refs],
        openalex_url=work.get("id", ""),
        type=work.get("type"),
        source_type=source.get("type"),
        source_name=source.get("display_name"),
        open_access=open_access.get("is_oa"),
        language=work.get("language"),
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
        fwci=work.get("fwci"),
        citation_percentile=citation_percentile,
        primary_topic=primary_topic,
        sdgs=sdgs if sdgs else None,
        keywords=keywords if keywords else None,
    )
