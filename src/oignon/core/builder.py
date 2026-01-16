"""Graph construction and ranking algorithms.

Architecture:
- ROOTS: Historical lineage (what led to this paper)
  - root_seeds = subject's references
  - root_papers = references of root_seeds (deeper history)

- BRANCHES: Future influence (what came from this paper)
  - branch_seeds = papers citing the subject
  - branch_papers = references of branch_seeds (filtered by frequency)
  - Must be published >= subject year + 1
"""

import math
from datetime import datetime
from typing import Callable

from oignon.core.graph import (
    FullPaper,
    Graph,
    GraphMetadata,
    SlimPaper,
)
from oignon.core.openalex import (
    fetch_citing_papers,
    fetch_paper,
    fetch_papers_full,
    fetch_papers_slim,
)

# Configuration
DEFAULT_N_ROOTS = 25
DEFAULT_N_BRANCHES = 25
DEFAULT_BRANCH_SEEDS_LIMIT = 200
CITATION_HALF_LIFE = 4

# Frequency filter: only fetch papers referenced by >= this many branch seeds
MIN_BRANCH_REF_FREQUENCY = 2


def recency_weight(
    paper_year: int,
    current_year: int | None = None,
    half_life: int = CITATION_HALF_LIFE,
) -> float:
    """Weight factor for recent papers (higher = newer)."""
    if current_year is None:
        current_year = datetime.now().year
    years_since = max(1, current_year - paper_year)
    return 1 + math.log(1 + half_life / years_since)


def compute_root_ranks(
    root_seeds: dict[str, SlimPaper],
    root_papers: dict[str, SlimPaper],
) -> dict[str, dict]:
    """Compute rank scores for root papers (historical lineage)."""
    all_papers = {**root_seeds, **root_papers}
    seed_ids = set(root_seeds.keys())

    # citedCount: how many seeds reference this paper
    cited_counts: dict[str, int] = {}
    for seed in root_seeds.values():
        for ref_id in seed.references:
            if ref_id in root_papers:
                cited_counts[ref_id] = cited_counts.get(ref_id, 0) + 1

    # coCitedCount: how often cited alongside seeds
    cocited_counts: dict[str, int] = {}
    for paper in all_papers.values():
        refs = set(paper.references)
        seeds_in_refs = refs & seed_ids
        if seeds_in_refs:
            for ref_id in refs:
                if ref_id not in seed_ids and ref_id in all_papers:
                    cocited_counts[ref_id] = cocited_counts.get(ref_id, 0) + len(
                        seeds_in_refs
                    )

    # coCitingCount: refs shared with seeds
    seed_refs: set[str] = set()
    for seed in root_seeds.values():
        seed_refs.update(seed.references)

    cociting_counts: dict[str, int] = {}
    for paper_id, paper in all_papers.items():
        if paper_id in seed_ids:
            continue
        refs = set(paper.references)
        shared = len(refs & seed_refs)
        if shared > 0:
            cociting_counts[paper_id] = shared

    # Combine into ranks
    ranks: dict[str, dict] = {}
    for paper_id in root_papers:
        cited = cited_counts.get(paper_id, 0)
        cocited = cocited_counts.get(paper_id, 0)
        cociting = cociting_counts.get(paper_id, 0)
        ranks[paper_id] = {
            "rank": cited + cocited + cociting,
            "citedCount": cited,
            "coCitedCount": cocited,
            "coCitingCount": cociting,
        }

    return ranks


def compute_branch_ranks(
    source: FullPaper | SlimPaper,
    branch_seeds: dict[str, SlimPaper],
    branch_papers: dict[str, SlimPaper],
) -> dict[str, dict]:
    """Compute rank scores for branch papers (future influence)."""
    subject_refs = set(source.references)
    subject_id = source.id
    all_papers = {**branch_seeds, **branch_papers}
    branch_seed_ids = set(branch_seeds.keys())
    current_year = datetime.now().year

    # citingCount: how many branch_seeds this paper references
    citing_counts: dict[str, int] = {}
    for paper_id, paper in branch_papers.items():
        refs = set(paper.references)
        count = len(refs & branch_seed_ids)
        if count > 0:
            citing_counts[paper_id] = count

    # coCitingCount: refs shared with subject
    cociting_counts: dict[str, int] = {}
    for paper_id, paper in branch_papers.items():
        refs = set(paper.references)
        shared = len(refs & subject_refs)
        if shared > 0:
            cociting_counts[paper_id] = shared

    # coCitedCount: how often cited alongside subject (recency-weighted)
    cocited_counts: dict[str, float] = {}
    for paper in all_papers.values():
        refs = set(paper.references)
        ref_ids = {r.split("/")[-1] if "/" in r else r for r in refs}
        if subject_id in ref_ids:
            weight = recency_weight(paper.year, current_year)
            for ref_id in ref_ids:
                if ref_id != subject_id and ref_id in branch_papers:
                    cocited_counts[ref_id] = cocited_counts.get(ref_id, 0.0) + weight

    # Combine into ranks
    ranks: dict[str, dict] = {}
    for paper_id in branch_papers:
        citing = citing_counts.get(paper_id, 0)
        cociting = cociting_counts.get(paper_id, 0)
        cocited = cocited_counts.get(paper_id, 0.0)
        ranks[paper_id] = {
            "rank": citing + cociting + cocited,
            "citingCount": citing,
            "coCitingCount": cociting,
            "coCitedCount": round(cocited, 2),
        }

    return ranks


def get_top_ranked(ranks: dict[str, dict], n: int = 50) -> list[str]:
    """Get the top N papers by rank score."""
    sorted_papers = sorted(ranks.items(), key=lambda x: x[1]["rank"], reverse=True)
    return [paper_id for paper_id, _ in sorted_papers[:n]]


def build_edges(
    source: FullPaper,
    all_seeds: dict[str, FullPaper],
    top_papers: dict[str, FullPaper],
) -> list[dict]:
    """Build citation edges between papers."""
    edges = []
    all_ids = {source.id} | set(all_seeds.keys()) | set(top_papers.keys())

    # Source -> its refs
    for ref_id in source.references:
        if ref_id in all_ids:
            edges.append({"source": source.id, "target": ref_id, "type": "cites"})

    # Seeds -> their refs
    for seed_id, seed in all_seeds.items():
        for ref_id in seed.references:
            if ref_id in all_ids:
                edges.append({"source": seed_id, "target": ref_id, "type": "cites"})

    # Top papers -> their refs
    for paper_id, paper in top_papers.items():
        for ref_id in paper.references:
            if ref_id in all_ids:
                edges.append({"source": paper_id, "target": ref_id, "type": "cites"})

    return edges


def build_graph(
    source_id: str,
    n_roots: int = DEFAULT_N_ROOTS,
    n_branches: int = DEFAULT_N_BRANCHES,
    on_progress: Callable[[str], None] | None = None,
) -> Graph:
    """Build the full citation network graph.

    Args:
        source_id: OpenAlex work ID, DOI, or URL
        n_roots: Number of top-ranked roots to include
        n_branches: Number of top-ranked branches to include
        on_progress: Optional callback for progress updates

    Returns:
        Complete Graph object
    """
    start_time = datetime.now()
    api_calls = 0

    def progress(msg: str):
        if on_progress:
            on_progress(msg)

    # Step 1: Fetch source paper
    progress(f"Fetching source: {source_id}")
    source = fetch_paper(source_id)
    api_calls += 1
    if not source:
        raise ValueError(f"Could not fetch source paper: {source_id}")
    progress(f"Source: {source.title} ({source.year})")

    # Step 2: Build ROOTS (historical lineage)
    progress(f"Fetching {len(source.references)} root seeds...")
    root_seeds_slim = fetch_papers_slim(source.references)
    api_calls += 1
    progress(f"  Got {len(root_seeds_slim)} root seeds")

    # Expand roots: refs of refs
    all_root_ref_ids: set[str] = set()
    for seed in root_seeds_slim.values():
        all_root_ref_ids.update(seed.references)
    all_root_ref_ids -= set(root_seeds_slim.keys())

    progress(f"Expanding roots: {len(all_root_ref_ids)} papers...")
    root_papers_slim = fetch_papers_slim(list(all_root_ref_ids))
    api_calls += 1
    progress(f"  Got {len(root_papers_slim)} root papers")

    # Step 3: Rank and select top roots
    progress("Ranking roots...")
    root_ranks = compute_root_ranks(root_seeds_slim, root_papers_slim)
    top_root_ids = get_top_ranked(root_ranks, n_roots)
    progress(f"  Selected top {len(top_root_ids)} roots")

    # Step 4: Build BRANCHES (future influence)
    progress("Fetching citing papers...")
    citing_ids = fetch_citing_papers(source.id, DEFAULT_BRANCH_SEEDS_LIMIT)
    api_calls += 1
    progress(f"  Found {len(citing_ids)} citing papers")

    if citing_ids:
        branch_seeds_slim_raw = fetch_papers_slim(citing_ids)
        api_calls += 1

        # Filter: citations > 0, year >= source.year + 1
        min_year = source.year + 1
        branch_seeds_slim = {
            pid: paper
            for pid, paper in branch_seeds_slim_raw.items()
            if paper.citation_count > 0 and paper.year >= min_year
        }
        progress(
            f"  Filtered to {len(branch_seeds_slim)} branch seeds "
            f"(year >= {min_year}, citations > 0)"
        )

        # Count reference frequency across branch seeds
        branch_ref_freq: dict[str, int] = {}
        for seed in branch_seeds_slim.values():
            for ref in seed.references:
                branch_ref_freq[ref] = branch_ref_freq.get(ref, 0) + 1

        # Filter to papers referenced by >= MIN_BRANCH_REF_FREQUENCY seeds
        branch_seed_ids = set(branch_seeds_slim.keys())
        filtered_branch_refs = {
            ref_id
            for ref_id, count in branch_ref_freq.items()
            if count >= MIN_BRANCH_REF_FREQUENCY
            and ref_id not in branch_seed_ids
            and ref_id != source.id
        }

        total_refs = len(branch_ref_freq)
        progress(
            f"  Frequency filter: {total_refs} -> {len(filtered_branch_refs)} refs "
            f"(>= {MIN_BRANCH_REF_FREQUENCY} seeds)"
        )

        # Fetch filtered branch refs
        branch_papers_slim_raw = fetch_papers_slim(list(filtered_branch_refs))
        api_calls += 1

        # Filter by year and citations
        branch_papers_slim = {
            pid: paper
            for pid, paper in branch_papers_slim_raw.items()
            if paper.year >= min_year and paper.citation_count > 0
        }
        progress(f"  Got {len(branch_papers_slim)} branch papers")
    else:
        branch_seeds_slim = {}
        branch_papers_slim = {}
        progress("  No citing papers found - branches empty")

    # Step 5: Rank and select top branches
    progress("Ranking branches...")

    # Convert source to SlimPaper for ranking
    source_slim = SlimPaper(
        id=source.id,
        year=source.year,
        citation_count=source.citation_count,
        references=source.references,
    )

    branch_ranks = compute_branch_ranks(
        source_slim, branch_seeds_slim, branch_papers_slim
    )
    top_branch_ids = get_top_ranked(branch_ranks, n_branches)
    progress(f"  Selected top {len(top_branch_ids)} branches")

    # Step 6: Fetch full metadata for final papers
    all_seed_ids = list(root_seeds_slim.keys()) + list(branch_seeds_slim.keys())
    all_top_ids = top_root_ids + top_branch_ids
    ids_needing_full = list(set(all_seed_ids + all_top_ids))

    progress(f"Fetching full metadata for {len(ids_needing_full)} papers...")
    full_papers = fetch_papers_full(ids_needing_full)
    api_calls += 1
    progress(f"  Got {len(full_papers)} full papers")

    # Build final structures
    all_ranks = {**root_ranks, **branch_ranks}

    # Top papers with roles and ranks
    top_papers: dict[str, FullPaper] = {}
    for pid in top_root_ids:
        if pid in full_papers:
            paper = full_papers[pid]
            paper.role = "root"
            paper.rank = all_ranks.get(pid, {}).get("rank", 0)
            paper.rank_details = all_ranks.get(pid)
            top_papers[pid] = paper

    for pid in top_branch_ids:
        if pid in full_papers:
            paper = full_papers[pid]
            paper.role = "branch"
            paper.rank = all_ranks.get(pid, {}).get("rank", 0)
            paper.rank_details = all_ranks.get(pid)
            top_papers[pid] = paper

    # Seeds with full data
    root_seeds_full = {
        pid: full_papers[pid] for pid in root_seeds_slim if pid in full_papers
    }
    branch_seeds_full = {
        pid: full_papers[pid] for pid in branch_seeds_slim if pid in full_papers
    }

    # Build edges
    all_seeds = {**root_seeds_full, **branch_seeds_full}
    edges = build_edges(source, all_seeds, top_papers)

    # Build metadata
    elapsed = (datetime.now() - start_time).total_seconds()
    metadata = GraphMetadata(
        source_year=source.year,
        total_root_seeds=len(root_seeds_slim),
        total_root_papers=len(root_papers_slim),
        total_branch_seeds=len(branch_seeds_slim),
        total_branch_papers=len(branch_papers_slim),
        n_roots=len(top_root_ids),
        n_branches=len(top_branch_ids),
        papers_in_graph=len(top_papers),
        edges_in_graph=len(edges),
        build_time_seconds=round(elapsed, 2),
        api_calls=api_calls,
        timestamp=datetime.now().isoformat(),
    )

    progress(
        f"Complete: {len(top_root_ids)} roots + {len(top_branch_ids)} branches "
        f"= {len(top_papers)} papers, {len(edges)} edges"
    )

    # Order papers by rank (roots first, then branches)
    ordered_papers = [top_papers[pid] for pid in top_root_ids if pid in top_papers] + [
        top_papers[pid] for pid in top_branch_ids if pid in top_papers
    ]

    return Graph(
        source_paper=source,
        root_seeds=list(root_seeds_full.values()),
        branch_seeds=list(branch_seeds_full.values()),
        papers=ordered_papers,
        edges=edges,
        metadata=metadata,
    )
