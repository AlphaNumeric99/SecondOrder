"""URL selection and chain query handling for research pipeline."""

from __future__ import annotations

import re
from typing import Any

from app.tools import web_utils


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
        """Select the top URLs to scrape based on relevance scores."""
        query_terms, quoted_phrases = self._query_terms(query)
        is_chain = self._is_chain_like_query(query)
        entity_hints = self._derive_entity_hints(search_results) if is_chain else []

        # Relation-aware retrieval for chain queries
        chain_components = self._extract_chain_components(query) if is_chain else {}
        primary_entity = chain_components.get("primary", "")

        # Deduplicate by URL, keep highest rank score.
        ranked_by_url: dict[str, float] = {}
        overlap_by_url: dict[str, int] = {}
        for r in search_results:
            url = r.get("url", "")
            raw_score = r.get("score", 0.0)
            try:
                base_score = float(raw_score)
            except (TypeError, ValueError):
                base_score = 0.0
            if url and web_utils.is_valid_url(url):
                domain = web_utils.extract_domain(url).lower()
                if domain in self.NOISY_DOMAINS:
                    continue

                title = str(r.get("title", "") or "").lower()
                snippet = str(r.get("content", "") or "").lower()
                url_lower = url.lower()

                title_overlap = 0
                url_overlap = 0
                snippet_overlap = 0
                if query_terms:
                    title_overlap = sum(1 for term in query_terms if term in title)
                    url_overlap = sum(1 for term in query_terms if term in url_lower)
                    snippet_overlap = sum(1 for term in query_terms if term in snippet)

                token_overlap = title_overlap + url_overlap + (0.25 * snippet_overlap)

                phrase_overlap = 0.0
                if quoted_phrases:
                    phrase_overlap = float(
                        sum(1 for phrase in quoted_phrases if phrase in title or phrase in url_lower)
                    ) + (
                        0.25 * float(sum(1 for phrase in quoted_phrases if phrase in snippet))
                    )
                entity_overlap = 0
                if entity_hints:
                    entity_overlap = sum(
                        1 for hint in entity_hints if hint in title or hint in url_lower or hint in snippet
                    )

                rank_score = (
                    base_score
                    + (token_overlap * 0.24)
                    + (phrase_overlap * 0.75)
                    + (entity_overlap * 0.35)
                )

                # Generic noise suppression
                lowered_url = url_lower
                if "/list_of_" in lowered_url and token_overlap <= 2:
                    rank_score -= 1.0
                if "/list_of_people_" in lowered_url:
                    rank_score -= 1.0
                if "how-to-get-a-wikipedia-page" in lowered_url and token_overlap <= 2:
                    rank_score -= 1.0
                if domain in {"bandzoogle.com", "diymusician.cdbaby.com"} and token_overlap <= 2:
                    rank_score -= 0.8

                # Relation-aware retrieval: penalize off-branch clusters for chain queries
                if chain_components and primary_entity:
                    all_candidates = chain_components.get("all_candidates", [])
                    matches_candidate = any(
                        cand in title or cand in url_lower
                        for cand in all_candidates if cand
                    )
                    matches_primary = primary_entity in title or primary_entity in url_lower

                    if all_candidates:
                        other_candidates = [c for c in all_candidates if c != primary_entity]
                        matches_others = any(
                            cand in title or cand in url_lower
                            for cand in other_candidates if cand
                        )
                        if matches_others and not matches_primary and token_overlap < 2:
                            rank_score -= 0.6
                            rank_score -= 0.5

                # Promote canonical pages
                if "wikipedia.org" in domain and token_overlap >= 2:
                    rank_score += 0.4

                if "wikipedia.org/wiki/" in lowered_url:
                    if any(
                        marker in lowered_url
                        for marker in ("_(musician)", "_(band)", "_(artist)", "_(drummer)")
                    ):
                        rank_score += 0.45
                    if "/list_of_" in lowered_url:
                        rank_score -= 0.8

                prev_score = ranked_by_url.get(url)
                if prev_score is None or rank_score > prev_score:
                    ranked_by_url[url] = rank_score
                    overlap_by_url[url] = int(token_overlap)

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
        priority_wiki = [
            url
            for url, _ in sorted_urls
            if "wikipedia.org/wiki/" in url.lower()
            and any(
                marker in url.lower()
                for marker in ("_(musician)", "_(band)", "_(artist)", "_(drummer)")
            )
        ]
        wiki_candidates = [
            url for url, _ in sorted_urls if "wikipedia.org" in web_utils.extract_domain(url)
        ]
        for wiki_url in (priority_wiki + wiki_candidates)[:2]:
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

        # Guarantee high-overlap URLs when query terms available
        if query_terms:
            overlap_sorted = sorted(
                overlap_by_url.items(), key=lambda x: x[1], reverse=True
            )
            for url, overlap_count in overlap_sorted:
                if overlap_count >= 2 and url not in selected:
                    if len(selected) < max_urls:
                        selected.append(url)

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
