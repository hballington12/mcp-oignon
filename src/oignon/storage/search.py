"""Ranked keyword search over a loaded citation graph.

Field-weighted BM25: matches in titles, keywords, and topics count more
than matches deep in an abstract, and multi-term queries rank papers
matching more (rarer) terms higher. This replaces naive substring
matching, which could only find literal phrases and returned results
in arbitrary order.
"""

import math
import re
from dataclasses import dataclass, field

BM25_K1 = 1.5
BM25_B = 0.75

FIELD_WEIGHTS = {
    "title": 3.0,
    "keywords": 2.5,
    "topic": 2.5,
    "authors": 2.0,
    "venue": 1.5,
    "abstract": 1.0,
    "other": 1.0,
}

# Minimal stopword list - enough to stop "the effect of" queries from
# matching every abstract, without pulling in a dependency.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "its", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "which", "with",
}


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, stopwords removed."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


@dataclass
class _Doc:
    """Indexed document: weighted term frequencies."""

    tf: dict[str, float] = field(default_factory=dict)
    length: float = 0.0


class SearchIndex:
    """In-memory field-weighted BM25 index."""

    def __init__(self):
        self._docs: dict[str, _Doc] = {}
        self._df: dict[str, int] = {}

    def clear(self) -> None:
        self._docs.clear()
        self._df.clear()

    def add(self, doc_id: str, fields: dict[str, str]) -> None:
        """Index a document. fields maps field name -> raw text."""
        doc = _Doc()

        for field_name, text in fields.items():
            if not text:
                continue
            weight = FIELD_WEIGHTS.get(field_name, 1.0)
            for token in tokenize(text):
                doc.tf[token] = doc.tf.get(token, 0.0) + weight
                doc.length += weight

        for token in doc.tf:
            self._df[token] = self._df.get(token, 0) + 1

        self._docs[doc_id] = doc

    def search(self, query: str, limit: int = 15) -> list[tuple[str, float]]:
        """BM25-ranked (doc_id, score) pairs for documents matching any term."""
        tokens = tokenize(query)
        if not tokens or not self._docs:
            return []

        n_docs = len(self._docs)
        avg_length = sum(d.length for d in self._docs.values()) / n_docs

        scores: dict[str, float] = {}
        for token in set(tokens):
            df = self._df.get(token)
            if not df:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

            for doc_id, doc in self._docs.items():
                tf = doc.tf.get(token)
                if not tf:
                    continue
                norm = 1 - BM25_B + BM25_B * doc.length / avg_length
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (
                    tf * (BM25_K1 + 1) / (tf + BM25_K1 * norm)
                )

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:limit]


def pick_snippets(
    observations: list[str], query: str, max_snippets: int = 2
) -> list[str]:
    """Pick the observations most relevant to the query, as evidence of
    why a paper matched."""
    tokens = set(tokenize(query))
    if not tokens:
        return []

    scored = []
    for obs in observations:
        if obs.startswith("Title: "):
            continue  # title is already shown in the result
        overlap = len(tokens & set(tokenize(obs)))
        if overlap > 0:
            scored.append((overlap, obs))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [obs for _, obs in scored[:max_snippets]]
