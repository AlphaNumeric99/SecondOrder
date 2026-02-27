"""URL selection and chain query handling for research pipeline."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from rank_bm25 import BM25Okapi

from app.services.embeddings_local import LocalEmbeddingService
from app.tools import web_utils


def _compute_bm25_scores(query: str, documents: list[str]) -> list[float]:
    """
    Compute BM25 scores using rank-bm25 library.

    Returns normalized scores (0-1 range).
    """
    if not documents:
        return []

    # Tokenize documents
    tokenized_docs = [doc.lower().split() for doc in documents]

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_docs)

    # Get scores for query - convert numpy array to list
    scores = bm25.get_scores(query.lower().split())
    scores = [float(s) for s in scores]  # Convert to Python floats

    # Normalize to 0-1 range
    max_score = max(scores) if scores else 1
    if max_score > 0:
        scores = [s / max_score for s in scores]

    return scores


def _clean_url_for_scoring(url: str) -> str:
    """Clean URL for scoring - extract domain and path, replace separators with spaces."""
    if not url:
        return ""
    try:
        from urllib.parse import urlparse
    except ImportError:
        return url

    parsed = urlparse(url)
    # Combine domain + path, remove params/fragments
    domain = parsed.netloc or ""
    path = parsed.path or ""
    # Replace / and _ and - with spaces, remove file extensions
    cleaned = f"{domain} {path}".lower()
    cleaned = re.sub(r'[/\-_.]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


async def _compute_embedding_scores(
    query: str, documents: list[str], urls: list[str]
) -> list[float]:
    """
    Compute embedding-based cosine similarity scores.

    Returns normalized scores (0-1 range).
    Embeddings are already normalized, so cosine similarity = dot product.
    """
    if not documents:
        return []

    # Add cleaned URLs to documents for richer context
    doc_texts = []
    for doc, url in zip(documents, urls):
        cleaned_url = _clean_url_for_scoring(url)
        combined = f"{doc} {cleaned_url}" if cleaned_url else doc
        doc_texts.append(combined)

    # Get embeddings
    embed_service = LocalEmbeddingService()
    query_embedding = await embed_service.embed_text(query)
    doc_embeddings = await embed_service.embed_texts(doc_texts)

    if not query_embedding or not doc_embeddings:
        return [0.0] * len(documents)

    # Compute cosine similarity (dot product since normalized)
    scores = []
    for doc_emb in doc_embeddings:
        similarity = sum(q * d for q, d in zip(query_embedding, doc_emb))
        scores.append(float(similarity))

    # Normalize to 0-1 range
    max_score = max(scores) if scores else 1
    if max_score > 0:
        scores = [s / max_score for s in scores]

    return scores


class URLSelector:
    """Handles URL selection and ranking for research pipeline."""

    # Domains to exclude from results
    NOISY_DOMAINS = {
        "reddit.com",
        "www.reddit.com",
        "tiktok.com",
        "www.tiktok.com",
        "youtube.com",
        "www.youtube.com",
        "facebook.com",
        "www.facebook.com",
        "tokchart.com",
        "quora.com",
        "www.quora.com",
        "ranker.com",
        "www.ranker.com",
        "thebash.com",
        "www.thebash.com",
        "singersroom.com",
        "www.singersroom.com",
    }

    def __init__(self):
        pass

    def select_top_urls(
        self,
        search_results: list[dict[str, Any]],
        query: str,
        max_urls: int = 8,
    ) -> list[str]:
        """Select the top URLs to scrape based on BM25 + embedding relevance scores."""
        # Extract year from query for special handling
        query_year = self._extract_year_from_query(query)

        # Prepare documents with cleaned URLs for BM25
        doc_texts = []
        urls = []
        for r in search_results:
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            urls.append(url)
            # Include cleaned URL in document text for better BM25 matching
            cleaned_url = _clean_url_for_scoring(url)
            doc_texts.append(f"{title} {content} {cleaned_url}")

        # Compute BM25 scores
        bm25_scores = _compute_bm25_scores(query, doc_texts)

        # Compute embedding scores (async)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        _compute_embedding_scores(query, doc_texts, urls)
                    )
                    embedding_scores = future.result(timeout=30)
            else:
                embedding_scores = asyncio.run(_compute_embedding_scores(query, doc_texts, urls))
        except Exception:
            embedding_scores = [0.0] * len(search_results)

        # Deduplicate by URL, keep highest rank score.
        ranked_by_url: dict[str, float] = {}
        for idx, r in enumerate(search_results):
            url = r.get("url", "")
            if url and web_utils.is_valid_url(url):
                domain = web_utils.extract_domain(url).lower()
                if domain in self.NOISY_DOMAINS:
                    continue

                title = str(r.get("title", "") or "").lower()
                snippet = str(r.get("content", "") or "").lower()
                lowered_url = url.lower()

                # Get BM25 and embedding scores
                bm25_score = bm25_scores[idx] if idx < len(bm25_scores) else 0.0
                embedding_score = embedding_scores[idx] if idx < len(embedding_scores) else 0.0

                # Base score: BM25 + embedding
                rank_score = (bm25_score * 0.50) + (embedding_score * 0.50)

                # Year-based boost
                if query_year:
                    url_year_match = f"/{query_year}/" in lowered_url or f"_{query_year}" in lowered_url
                    if url_year_match:
                        rank_score += 1.5
                    if query_year in title or query_year in snippet:
                        rank_score += 0.5

                # Wikipedia domain boost
                if "wikipedia.org" in domain:
                    rank_score += 0.4

                prev_score = ranked_by_url.get(url)
                if prev_score is None or rank_score > prev_score:
                    ranked_by_url[url] = rank_score

        # Sort by rank score descending
        sorted_urls = sorted(ranked_by_url.items(), key=lambda x: x[1], reverse=True)
        selected: list[str] = []
        domain_counts: dict[str, int] = {}
        for url, _ in sorted_urls:
            if len(selected) >= max_urls:
                break
            domain = web_utils.extract_domain(url).lower()
            cap = 3 if "wikipedia.org" in domain else 2
            if domain_counts.get(domain, 0) >= cap:
                continue
            selected.append(url)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Fill remaining slots if needed
        if len(selected) < max_urls:
            for url, _ in sorted_urls:
                if len(selected) >= max_urls:
                    break
                if url not in selected:
                    selected.append(url)

        # Ensure Wikipedia representation
        wiki_candidates = [
            url for url, _ in sorted_urls if "wikipedia.org" in web_utils.extract_domain(url)
        ]
        for wiki_url in wiki_candidates[:2]:
            if wiki_url in selected:
                continue
            if len(selected) < max_urls:
                selected.append(wiki_url)
                continue
            # Replace lowest-ranked non-wiki URL
            for idx in range(len(selected) - 1, -1, -1):
                if "wikipedia.org" not in web_utils.extract_domain(selected[idx]):
                    selected[idx] = wiki_url
                    break

        return selected

    @staticmethod
    def _query_terms(query: str) -> tuple[list[str], list[str]]:
        """Extract query terms and quoted phrases from query."""
        if not query:
            return [], []

        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        quoted_phrases = [q.lower().strip() for q in quoted]

        # Remove quoted phrases and clean up
        query_without_quotes = re.sub(r'"[^"]+"', "", query)
        raw_terms = query_without_quotes.split()
        terms = []
        for term in raw_terms:
            cleaned = "".join(c for c in term if c.isalnum() or c in "-_").strip().lower()
            if cleaned and len(cleaned) > 1:
                terms.append(cleaned)

        # Remove very short and very long terms
        filtered_terms = [t for t in terms if 2 <= len(t) <= 30]
        return filtered_terms, quoted_phrases

    def _derive_entity_hints(self, search_results: list[dict[str, Any]]) -> list[str]:
        """Derive entity hints from search results for chain queries."""
        counts: dict[str, int] = {}
        for result in search_results[:80]:
            title = str(result.get("title", "") or "").strip()
            url = str(result.get("url", "") or "").strip()
            domain = web_utils.extract_domain(url).lower()
            if not title:
                continue

            candidates = self._extract_title_case_phrases(title)
            if "wikipedia.org" in domain:
                wiki_title = re.sub(
                    r"\s*-\s*Wikipedia\s*$", "", title, flags=re.IGNORECASE
                ).strip()
                if wiki_title:
                    candidates.append(wiki_title)

            bonus = 2 if "wikipedia.org" in domain else 1
            for candidate in candidates:
                normalized = " ".join(candidate.split()).strip().lower()
                if not normalized:
                    continue
                if self._is_generic_entity_phrase(normalized):
                    continue
                counts[normalized] = counts.get(normalized, 0) + bonus

        ranked = sorted(counts.items(), key=lambda item: (item[1], len(item[0])), reverse=True)
        hints: list[str] = []
        for hint, score in ranked:
            if score < 2:
                continue
            hints.append(hint)
            if len(hints) >= 6:
                break
        return hints

    @staticmethod
    def _is_chain_like_query(query: str) -> bool:
        """Check if query is a chain/multi-hop query."""
        lowered = (query or "").lower()
        markers = (
            "who",
            "that",
            "which",
            "from 20",
            "from 19",
            "played",
            "won",
            "name the two",
        )
        return sum(1 for marker in markers if marker in lowered) >= 3

    def _extract_chain_components(self, query: str) -> dict[str, list[str]]:
        """Extract chain components from multi-hop queries."""
        lowered = query.lower()

        relation_patterns = [
            "played drums",
            "played",
            "won",
            "from 20",
            "from 19",
            "best of",
            "viral on",
            "drummer in",
            "band that",
            "artist who",
        ]

        relations = [r for r in relation_patterns if r in lowered]

        primary_candidates = []

        # Quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        primary_candidates.extend([q.lower() for q in quoted])

        # Extract band after "in a band that won"
        if "band that won" in lowered or "band who won" in lowered:
            match = re.search(r'in a (?:band that|band who) won ["\']?([^"\']+)["\']?', lowered)
            if match:
                primary_candidates.append(match.group(1).strip())

        return {
            "primary": primary_candidates[0] if primary_candidates else "",
            "relations": relations,
            "all_candidates": primary_candidates,
        }

    @staticmethod
    def _is_generic_entity_phrase(value: str) -> bool:
        """Check if phrase is too generic to be useful."""
        lowered = value.lower().strip()
        if not lowered:
            return True
        generic_fragments = (
            "best ",
            "award",
            "timeline",
            "wikipedia",
            "newsroom",
            "top ",
            "list of ",
            "people and places",
            "share to",
            "email facebook",
        )
        return any(frag in lowered for frag in generic_fragments)

    @staticmethod
    def _extract_year_from_query(query: str) -> str | None:
        """Extract a 4-digit year from the query if present."""
        if not query:
            return None
        # Look for years between 1990 and 2030
        matches = re.findall(r'\b(19[9]\d|20[0-2]\d)\b', query)
        if matches:
            # Return the first year found (most specific to the query)
            return matches[0]
        return None

    @staticmethod
    def _extract_title_case_phrases(text: str) -> list[str]:
        """Extract title-case phrases from text."""
        if not text:
            return []

        # Match sequences of words starting with capital letters
        # but allowing for parentheses, colons, etc.
        title_pattern = re.compile(
            r'(?:^|[\s\-/:(])([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)'
        )
        matches = title_pattern.findall(text)

        # Filter out common non-entity patterns
        phrases = []
        for match in matches:
            # Skip if too short or matches common patterns
            if len(match) < 3:
                continue
            # Skip if it's a date pattern
            if re.match(r'^\d{4}$', match):
                continue
            # Skip if it's just a number
            if match.isdigit():
                continue
            phrases.append(match.strip())

        return phrases
