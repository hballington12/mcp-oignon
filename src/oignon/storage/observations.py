"""Convert papers to searchable observation strings and index fields."""

import re

from oignon.core.graph import FullPaper


def build_observations(paper: FullPaper, role: str) -> list[str]:
    """Flatten paper metadata into human-readable observation strings."""
    observations = []

    # Core metadata
    observations.append(f"Title: {paper.title}")
    observations.append(f"Year: {paper.year}")

    if paper.authors:
        author_names = [a.name for a in paper.authors[:5]]
        if len(paper.authors) > 5:
            author_names.append("et al.")
        observations.append(f"Authors: {', '.join(author_names)}")

    observations.append(f"Citations: {paper.citation_count}")
    observations.append(f"References: {paper.references_count}")

    if paper.doi:
        observations.append(f"DOI: {paper.doi}")

    observations.append(f"Graph role: {role}")

    if paper.source_name:
        observations.append(f"Published in: {paper.source_name}")

    if paper.open_access is not None:
        observations.append(f"Open access: {'yes' if paper.open_access else 'no'}")

    if paper.fwci:
        observations.append(f"Field-weighted citation impact: {paper.fwci:.2f}")

    if paper.citation_percentile:
        cp = paper.citation_percentile
        if cp.is_in_top_1_percent:
            observations.append("Highly cited: top 1% in field")
        elif cp.is_in_top_10_percent:
            observations.append("Highly cited: top 10% in field")

    if paper.primary_topic:
        topic = paper.primary_topic
        observations.append(f"Topic: {topic.name}")
        if topic.field:
            observations.append(f"Field: {topic.field.get('name', '')}")
        if topic.domain:
            observations.append(f"Domain: {topic.domain.get('name', '')}")

    if paper.keywords:
        observations.append(f"Keywords: {', '.join(paper.keywords[:10])}")

    if paper.sdgs:
        sdg_names = [s.name for s in paper.sdgs[:3]]
        observations.append(f"SDGs: {', '.join(sdg_names)}")

    # Abstract sentences
    if paper.abstract:
        observations.extend(split_abstract(paper.abstract))

    return observations


def build_index_fields(paper: FullPaper, role: str) -> dict[str, str]:
    """Structured fields for the BM25 index (field-weighted)."""
    topic_parts = []
    if paper.primary_topic:
        topic = paper.primary_topic
        topic_parts.append(topic.name)
        for level in (topic.subfield, topic.field, topic.domain):
            if level:
                topic_parts.append(level.get("name", ""))

    return {
        "title": paper.title,
        "abstract": paper.abstract or "",
        "authors": " ".join(a.name for a in paper.authors),
        "keywords": " ".join(paper.keywords or []),
        "topic": " ".join(topic_parts),
        "venue": paper.source_name or "",
        "other": f"{paper.year} {role} {paper.type or ''}",
    }


def split_abstract(abstract: str, max_sentences: int = 10) -> list[str]:
    """Split abstract into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", abstract.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences[:max_sentences]
