"""
Module 3 Exercise: Automated Literature-Evidence Triage

Multi-stage workflow: Search → Deduplicate → Score → Extract

Learning Objective: Visualize an agentic workflow model by representing agent
tasks, interactions, and data flows in a diagram.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from difflib import SequenceMatcher

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from ls_action_space.action_space import query_pubmed, query_scholar, query_arxiv


def generate_workflow_diagram() -> str:
    """
    Generate Mermaid.js flowchart for the literature triage workflow.

    The diagram should show:
    1. Query input
    2. Three parallel search agents (PubMed, Scholar, ArXiv)
    3. Merge/Deduplication step
    4. Relevance scoring
    5. Evidence extraction
    6. Final output

    Returns:
        Valid Mermaid.js flowchart code (use 'flowchart TD' for top-down)
    """
    # TODO: Create a flowchart showing:
    # - Query node at top
    # - Fan-out to 3 search agents (parallel)
    # - Merge point
    # - Sequential: Dedup → Score → Extract → Output
    # Hint: Use --> for arrows, subgraph for grouping
    pass


class LiteratureTriageWorkflow:
    """Multi-stage literature triage: Search → Deduplicate → Score → Extract"""

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client
        self.trace: List[str] = []

    def _log(self, msg: str) -> None:
        self.trace.append(msg)
        print(f"  {msg}")

    # =========================================================================
    # STAGE 1: PARALLEL SEARCH (Provided)
    # =========================================================================

    def search_all_sources(self, query: str) -> List[Dict]:
        """Search PubMed, Scholar, and ArXiv in parallel."""
        all_results = []

        # PubMed
        start = time.time()
        pubmed = query_pubmed(query, max_results=10)
        for r in pubmed:
            r["source"] = "PubMed"
        all_results.extend(pubmed)
        self._log(f"PubMed: {len(pubmed)} articles ({time.time()-start:.1f}s)")

        # Scholar (may fail if dependencies missing)
        try:
            start = time.time()
            scholar_raw = query_scholar(query, max_results=5)
            for r in scholar_raw:
                if "error" not in r:
                    all_results.append({
                        "title": r.get("title", ""),
                        "abstract": r.get("snippet", ""),
                        "source": "Scholar"
                    })
            self._log(f"Scholar: {len(scholar_raw)} articles ({time.time()-start:.1f}s)")
        except ImportError:
            self._log("Scholar: skipped (missing beautifulsoup4)")

        # ArXiv (may fail if dependencies missing)
        try:
            start = time.time()
            arxiv_raw = query_arxiv(query, max_results=5)
            for r in arxiv_raw:
                all_results.append({
                    "title": r.get("title", ""),
                    "abstract": r.get("summary", ""),
                    "authors": r.get("authors", []),
                    "source": "ArXiv"
                })
            self._log(f"ArXiv: {len(arxiv_raw)} articles ({time.time()-start:.1f}s)")
        except ImportError:
            self._log("ArXiv: skipped (missing arxiv package)")

        return all_results

    # =========================================================================
    # STAGE 2: DEDUPLICATION (Provided)
    # =========================================================================

    def deduplicate(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicates by title similarity (>85% match)."""
        unique = []
        for article in articles:
            title = article.get("title", "").lower()
            is_dup = any(
                SequenceMatcher(None, title, u.get("title", "").lower()).ratio() > 0.85
                for u in unique
            )
            if not is_dup:
                unique.append(article)
        self._log(f"Deduplication: {len(articles)} → {len(unique)} unique")
        return unique

    # =========================================================================
    # STAGE 3: RELEVANCE SCORING (TODO - Implement this)
    # =========================================================================

    def score_relevance(self, articles: List[Dict], query: str) -> List[Dict]:
        """
        Use LLM to score each article's relevance (0-100).

        For each article:
        1. Build prompt with title, abstract, and query
        2. Ask LLM to score relevance 0-100 with brief reasoning
        3. Parse JSON response and add 'relevance_score' to article

        Returns: articles with 'relevance_score' field added
        """
        # TODO: Loop through articles
        # TODO: For each, prompt LLM: "Score this article's relevance to '{query}'"
        # TODO: Request JSON response: {"score": int, "reasoning": str}
        # TODO: Parse response, add score to article (default 50 on error)
        # TODO: Return scored articles
        pass

    # =========================================================================
    # STAGE 4: EVIDENCE EXTRACTION (TODO - Implement this)
    # =========================================================================

    def extract_evidence(self, articles: List[Dict]) -> List[Dict]:
        """
        Use LLM to extract structured evidence from each article.

        For each article, extract:
        - key_findings: List of 2-3 main conclusions
        - study_type: RCT, Review, Meta-Analysis, Case Study, etc.

        Returns: articles with 'key_findings' and 'study_type' fields added
        """
        # TODO: Loop through articles
        # TODO: Prompt LLM to extract key_findings and study_type from abstract
        # TODO: Request JSON response: {"key_findings": [...], "study_type": str}
        # TODO: Parse response, add fields to article
        # TODO: Return articles with evidence fields
        pass

    # =========================================================================
    # MAIN WORKFLOW (TODO - Implement this)
    # =========================================================================

    def run(self, query: str, threshold: int = 60) -> Dict:
        """
        Execute the complete triage workflow.

        Steps:
        1. Search all sources (provided)
        2. Deduplicate (provided)
        3. Score relevance (your implementation)
        4. Filter by threshold
        5. Extract evidence (your implementation)
        6. Sort by score descending

        Returns: {query, articles, stats, trace}
        """
        print(f"\n{'='*50}")
        print(f"LITERATURE TRIAGE: {query}")
        print(f"{'='*50}")
        self.trace = []

        # TODO: Step 1 - Call search_all_sources()
        # TODO: Step 2 - Call deduplicate()
        # TODO: Step 3 - Call score_relevance()
        # TODO: Step 4 - Filter articles where relevance_score >= threshold
        # TODO: Step 5 - Call extract_evidence() on filtered articles
        # TODO: Step 6 - Sort by relevance_score descending
        # TODO: Return results dict
        pass


def main():
    # Show workflow diagram
    print("=" * 50)
    print("WORKFLOW DIAGRAM")
    print("=" * 50)
    diagram = generate_workflow_diagram()
    if diagram:
        print("```mermaid")
        print(diagram)
        print("```")
        print("\nPaste into https://mermaid.live to visualize\n")
    else:
        print("TODO: Implement generate_workflow_diagram()\n")

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")
    workflow = LiteratureTriageWorkflow(client)

    results = workflow.run("CRISPR gene therapy safety", threshold=60)

    if results is None:
        print("\nImplement the TODO methods to see results.")
        return

    print(f"\n{'='*50}")
    print(f"RESULTS: {len(results.get('articles', []))} relevant articles")
    print(f"{'='*50}")
    for i, a in enumerate(results.get("articles", [])[:5], 1):
        print(f"\n{i}. [{a.get('relevance_score', 0)}] {a.get('title', '')[:60]}")
        print(f"   Type: {a.get('study_type', 'Unknown')} | Source: {a.get('source', 'Unknown')}")
        for f in a.get("key_findings", [])[:2]:
            print(f"   - {f[:70]}")


if __name__ == "__main__":
    main()
