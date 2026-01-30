#!/usr/bin/env python3
"""
Module 1 Exercise: Literature Evidence Summarizer (Keyword Filter vs Agent)

This module demonstrates the difference between deterministic keyword-based filtering
and LLM-backed agentic analysis for pharmacovigilance literature review.

Learning Objective:
    Distinguish between deterministic and agentic systems and identify the
    core components of an AI agent.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from ls_action_space.action_space import query_pubmed


def keyword_adverse_event_filter(
    drug_name: str,
    adverse_event_term: str,
    max_results: int = 15
) -> List[Dict[str, str]]:
    """
    Simple keyword-based filtering for adverse event literature.

    This represents the deterministic, rule-based approach that cannot
    understand context or distinguish between mentions and actual reports.

    Args:
        drug_name: Name of the drug (e.g., "metformin")
        adverse_event_term: Adverse event to search for (e.g., "lactic acidosis")
        max_results: Maximum PubMed results to fetch

    Returns:
        List of articles where abstract contains BOTH drug_name AND adverse_event_term.
        Each dict contains: pmid, title, abstract
    """
    # Step 1: Construct PubMed query
    query = f"{drug_name} {adverse_event_term}"

    # Step 2: Call query_pubmed
    articles = query_pubmed(query, max_results=max_results)

    # Step 3: Filter to articles containing both terms (case-insensitive)
    drug_lower = drug_name.lower()
    ae_lower = adverse_event_term.lower()

    filtered_articles = []
    for article in articles:
        abstract_lower = article.get("abstract", "").lower()
        if drug_lower in abstract_lower and ae_lower in abstract_lower:
            filtered_articles.append({
                "pmid": article["pmid"],
                "title": article["title"],
                "abstract": article["abstract"]
            })

    return filtered_articles


def agent_adverse_event_analyzer(
    drug_name: str,
    adverse_event_term: str,
    llm_client: OpenAI,
    max_results: int = 15
) -> List[Dict[str, Any]]:
    """
    LLM-backed analysis of adverse event literature.

    This represents the agentic approach that can understand context,
    classify evidence quality, and provide actionable summaries.

    Args:
        drug_name: Name of the drug
        adverse_event_term: Adverse event to search for
        llm_client: OpenAI client instance
        max_results: Maximum PubMed results to analyze

    Returns:
        List of analyzed articles with relevance assessment, summaries,
        evidence level classification, and confidence scores.
    """
    # Step 1: Construct intelligent PubMed query
    query = f"{drug_name} {adverse_event_term}"
    articles = query_pubmed(query, max_results=max_results)

    results = []

    # Step 2: Analyze each article with non-empty abstract
    for article in articles:
        abstract = article.get("abstract", "")

        if not abstract.strip():
            # Skip articles without abstracts
            results.append({
                "pmid": article["pmid"],
                "title": article["title"],
                "relevant": False,
                "summary": "No abstract available for analysis",
                "evidence_level": "Unknown",
                "confidence": "Low"
            })
            continue

        # Step 3: Use LLM to analyze the article
        analysis = _analyze_article_with_llm(
            llm_client=llm_client,
            title=article["title"],
            abstract=abstract,
            drug_name=drug_name,
            adverse_event_term=adverse_event_term
        )

        results.append({
            "pmid": article["pmid"],
            "title": article["title"],
            **analysis
        })

    return results


def _analyze_article_with_llm(
    llm_client: OpenAI,
    title: str,
    abstract: str,
    drug_name: str,
    adverse_event_term: str
) -> Dict[str, Any]:
    """
    Use LLM to analyze a single article for adverse event relevance.

    Args:
        llm_client: OpenAI client instance
        title: Article title
        abstract: Article abstract
        drug_name: Drug being investigated
        adverse_event_term: Adverse event being searched

    Returns:
        Dictionary with: relevant, summary, evidence_level, confidence
    """
    system_prompt = """You are a pharmacovigilance expert analyzing scientific literature for adverse event signals.

Your task is to determine if an article ACTUALLY REPORTS or INVESTIGATES a specific adverse event for a drug,
versus merely MENTIONING it in background, exclusion criteria, or unrelated context.

You must be conservative: if the relevance is unclear, mark confidence as "Low".

Respond with a JSON object containing exactly these fields:
- "relevant": boolean - Does this article report new data, cases, or findings about the adverse event?
- "summary": string - One sentence summarizing the key finding related to the drug-adverse event relationship
- "evidence_level": string - One of: "Case Report", "Case Series", "Cohort Study", "RCT", "Meta-Analysis", "Review", "In Vitro/Animal"
- "confidence": string - "High", "Medium", or "Low" - your confidence in the relevance assessment

Examples of RELEVANT articles:
- Reports actual cases of the adverse event
- Provides incidence/prevalence data
- Investigates mechanisms or risk factors for the adverse event

Examples of NOT RELEVANT articles:
- Mentions adverse event only in exclusion criteria
- Discusses the adverse event in background/introduction only
- Reviews the topic without new data
- Studies a different primary outcome"""

    user_prompt = f"""Analyze this article for relevance to {drug_name}-associated {adverse_event_term}:

TITLE: {title}

ABSTRACT: {abstract}

Provide your analysis as JSON."""

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # Validate and normalize the response
        return {
            "relevant": bool(result.get("relevant", False)),
            "summary": str(result.get("summary", "Analysis unavailable")),
            "evidence_level": _validate_evidence_level(result.get("evidence_level", "Unknown")),
            "confidence": _validate_confidence(result.get("confidence", "Low"))
        }

    except Exception as e:
        # Handle API errors gracefully
        return {
            "relevant": False,
            "summary": f"Error during analysis: {str(e)}",
            "evidence_level": "Unknown",
            "confidence": "Low"
        }


def _validate_evidence_level(level: str) -> str:
    """Validate and normalize evidence level classification."""
    valid_levels = {
        "case report", "case series", "cohort study", "rct",
        "meta-analysis", "review", "in vitro/animal", "unknown"
    }
    level_lower = level.lower()

    # Map variations to standard terms
    if "case report" in level_lower:
        return "Case Report"
    elif "case series" in level_lower:
        return "Case Series"
    elif "cohort" in level_lower:
        return "Cohort Study"
    elif "rct" in level_lower or "randomized" in level_lower:
        return "RCT"
    elif "meta" in level_lower:
        return "Meta-Analysis"
    elif "review" in level_lower:
        return "Review"
    elif "vitro" in level_lower or "animal" in level_lower:
        return "In Vitro/Animal"
    else:
        return "Unknown"


def _validate_confidence(confidence: str) -> str:
    """Validate and normalize confidence level."""
    confidence_lower = confidence.lower()
    if "high" in confidence_lower:
        return "High"
    elif "medium" in confidence_lower or "moderate" in confidence_lower:
        return "Medium"
    else:
        return "Low"


def compare_approaches(
    drug_name: str,
    adverse_event_term: str,
    llm_client: OpenAI
) -> None:
    """
    Run both approaches and display comparative results.

    Args:
        drug_name: Drug to analyze
        adverse_event_term: Adverse event to search
        llm_client: OpenAI client instance
    """
    # Header
    print("=" * 80)
    print("ADVERSE EVENT LITERATURE REVIEW")
    print(f"Drug: {drug_name} | Adverse Event: {adverse_event_term}")
    print("=" * 80)
    print()

    # Run keyword filter
    print("Running keyword filter...")
    keyword_results = keyword_adverse_event_filter(drug_name, adverse_event_term)

    # Run agent analysis
    print("Running agent analysis...")
    agent_results = agent_adverse_event_analyzer(drug_name, adverse_event_term, llm_client)

    print()

    # Display keyword results
    print("-" * 80)
    print("KEYWORD FILTER RESULTS:")
    print(f"Found {len(keyword_results)} articles containing both keywords")
    print("-" * 80)
    print()

    if keyword_results:
        # Table header
        print(f"{'PMID':<12} | {'Title (truncated to 50 chars)':<52}")
        print(f"{'-'*12} | {'-'*52}")

        for article in keyword_results:
            title = article["title"][:50] + "..." if len(article["title"]) > 50 else article["title"]
            print(f"{article['pmid']:<12} | {title:<52}")
    else:
        print("No articles found matching keyword criteria.")

    print()
    print("=" * 80)
    print()

    # Display agent results
    relevant_count = sum(1 for r in agent_results if r["relevant"])
    print("-" * 80)
    print("AGENT ANALYSIS RESULTS:")
    print(f"Analyzed {len(agent_results)} articles, {relevant_count} determined relevant")
    print("-" * 80)
    print()

    if agent_results:
        # Table header
        print(f"{'PMID':<12} | {'Summary':<40} | {'Evidence':<14} | {'Relevant':<10}")
        print(f"{'-'*12} | {'-'*40} | {'-'*14} | {'-'*10}")

        for article in agent_results:
            summary = article["summary"][:38] + ".." if len(article["summary"]) > 40 else article["summary"]
            relevant_str = "✓ Yes" if article["relevant"] else "✗ No"
            print(f"{article['pmid']:<12} | {summary:<40} | {article['evidence_level']:<14} | {relevant_str:<10}")

    print()
    print("=" * 80)
    print()

    # Key differences analysis
    print("-" * 80)
    print("KEY DIFFERENCES:")
    print("-" * 80)
    print()

    # Find false positives (in keyword but not relevant per agent)
    keyword_pmids = {r["pmid"] for r in keyword_results}
    agent_lookup = {r["pmid"]: r for r in agent_results}

    false_positives = []
    for pmid in keyword_pmids:
        if pmid in agent_lookup and not agent_lookup[pmid]["relevant"]:
            false_positives.append(agent_lookup[pmid])

    print("False Positives from Keywords (mentioned but not relevant):")
    if false_positives:
        for fp in false_positives[:5]:  # Limit to 5
            print(f"  - PMID {fp['pmid']}: \"{fp['summary'][:60]}...\"")
    else:
        print("  (None identified)")

    print()

    # Find true relevant findings
    true_findings = [r for r in agent_results if r["relevant"]]

    print("True Findings Identified by Agent:")
    if true_findings:
        for tf in true_findings[:5]:  # Limit to 5
            print(f"  - PMID {tf['pmid']}: {tf['summary'][:60]}...")
    else:
        print("  (None identified)")

    print()

    # Count high-value evidence
    high_value_evidence = [r for r in agent_results
                          if r["relevant"] and r["evidence_level"] in ["RCT", "Meta-Analysis", "Cohort Study"]]

    # Summary statistics
    print("-" * 80)
    print("Summary:")
    print(f"  - Keyword filter: {len(keyword_results)} articles (includes {len(false_positives)} false positives)")
    print(f"  - Agent filter: {relevant_count} relevant articles (all verified)")
    if high_value_evidence:
        evidence_types = ", ".join(set(r["evidence_level"] for r in high_value_evidence))
        print(f"  - Agent identified {len(high_value_evidence)} high-quality evidence sources ({evidence_types})")
    print("=" * 80)


def main():
    """
    Main function that runs the comparison with test cases.
    """
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key to run the agent analysis.")
        return

    llm_client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

    # Test Case 1: Metformin and Lactic Acidosis
    print("\n" + "=" * 80)
    print("TEST CASE 1: METFORMIN AND LACTIC ACIDOSIS")
    print("=" * 80 + "\n")

    compare_approaches(
        drug_name="metformin",
        adverse_event_term="lactic acidosis",
        llm_client=llm_client
    )

    # Bonus: Test Case 2: Warfarin and Bleeding
    print("\n" + "=" * 80)
    print("TEST CASE 2: WARFARIN AND BLEEDING (BONUS)")
    print("=" * 80 + "\n")

    compare_approaches(
        drug_name="warfarin",
        adverse_event_term="bleeding",
        llm_client=llm_client
    )


if __name__ == "__main__":
    main()
