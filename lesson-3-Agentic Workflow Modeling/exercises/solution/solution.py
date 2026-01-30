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
    """Generate Mermaid.js flowchart for the literature triage workflow."""
    return """flowchart TD
    Query[Search Query] --> PubMed[PubMed Agent]
    Query --> Scholar[Scholar Agent]
    Query --> ArXiv[ArXiv Agent]

    PubMed --> Merge[Merge Results]
    Scholar --> Merge
    ArXiv --> Merge

    Merge --> Dedup[Deduplicate]
    Dedup --> Score[Relevance Scorer]
    Score --> Filter{Score >= Threshold?}
    Filter -->|Yes| Extract[Evidence Extractor]
    Filter -->|No| Discard[Discard]
    Extract --> Output[Ranked Results]"""


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
    # STAGE 3: RELEVANCE SCORING
    # =========================================================================

    def score_relevance(self, articles: List[Dict], query: str) -> List[Dict]:
        """Use LLM to score each article's relevance (0-100)."""
        self._log(f"Scoring {len(articles)} articles...")

        for article in articles:
            prompt = f"""Score this article's relevance to the query: "{query}"

Title: {article.get('title', 'N/A')}
Abstract: {article.get('abstract', 'N/A')[:500]}

Respond with JSON only: {{"score": <0-100>, "reasoning": "<brief explanation>"}}"""

            try:
                response = self.llm.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=150
                )
                result = json.loads(response.choices[0].message.content)
                article["relevance_score"] = result.get("score", 50)
            except Exception:
                article["relevance_score"] = 50

        return articles

    # =========================================================================
    # STAGE 4: EVIDENCE EXTRACTION
    # =========================================================================

    def extract_evidence(self, articles: List[Dict]) -> List[Dict]:
        """Use LLM to extract structured evidence from each article."""
        self._log(f"Extracting evidence from {len(articles)} articles...")

        for article in articles:
            prompt = f"""Extract evidence from this article:

Title: {article.get('title', 'N/A')}
Abstract: {article.get('abstract', 'N/A')}

Respond with JSON only:
{{"key_findings": ["finding1", "finding2"], "study_type": "RCT|Review|Meta-Analysis|Case Study|Other"}}"""

            try:
                response = self.llm.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=200
                )
                result = json.loads(response.choices[0].message.content)
                article["key_findings"] = result.get("key_findings", [])
                article["study_type"] = result.get("study_type", "Unknown")
            except Exception:
                article["key_findings"] = ["Extraction failed"]
                article["study_type"] = "Unknown"

        return articles

    # =========================================================================
    # MAIN WORKFLOW
    # =========================================================================

    def run(self, query: str, threshold: int = 60) -> Dict:
        """Execute the complete triage workflow."""
        print(f"\n{'='*50}")
        print(f"LITERATURE TRIAGE: {query}")
        print(f"{'='*50}")
        self.trace = []

        # Step 1: Search
        articles = self.search_all_sources(query)

        # Step 2: Deduplicate
        articles = self.deduplicate(articles)

        # Step 3: Score
        articles = self.score_relevance(articles, query)

        # Step 4: Filter by threshold
        relevant = [a for a in articles if a.get("relevance_score", 0) >= threshold]
        self._log(f"Above threshold ({threshold}): {len(relevant)} articles")

        # Step 5: Extract evidence
        relevant = self.extract_evidence(relevant)

        # Step 6: Sort by score
        relevant.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        return {
            "query": query,
            "articles": relevant,
            "stats": {"searched": len(articles), "relevant": len(relevant)},
            "trace": self.trace
        }


def main():
    # Show workflow diagram
    print("=" * 50)
    print("WORKFLOW DIAGRAM")
    print("=" * 50)
    print("```mermaid")
    print(generate_workflow_diagram())
    print("```")
    print("\nPaste into https://mermaid.live to visualize\n")

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")
    workflow = LiteratureTriageWorkflow(client)

    results = workflow.run("CRISPR gene therapy safety", threshold=60)

    print(f"\n{'='*50}")
    print(f"RESULTS: {len(results['articles'])} relevant articles")
    print(f"{'='*50}")
    for i, a in enumerate(results["articles"][:5], 1):
        print(f"\n{i}. [{a.get('relevance_score', 0)}] {a.get('title', '')[:60]}")
        print(f"   Type: {a.get('study_type', 'Unknown')} | Source: {a.get('source', 'Unknown')}")
        for f in a.get("key_findings", [])[:2]:
            print(f"   - {f[:70]}")


if __name__ == "__main__":
    main()
