"""
Module 1 Demo: The Limitations of Deterministic Functions

This module demonstrates the difference between deterministic (regex-based)
and agentic (LLM-based) approaches to finding gene mentions in scientific text.

Part of: Agentic Workflows for Life Sciences Research
"""

import os
import re
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ls_action_space.action_space import query_pubmed


# =============================================================================
# Part 1: Deterministic Function
# =============================================================================

def find_gene_mentions_regex(abstract: str, gene_symbol: str) -> dict:
    """
    Find exact matches of a gene symbol in text using regex.

    Args:
        abstract: Text to search (e.g., PubMed abstract)
        gene_symbol: Gene to find (e.g., "TP53")

    Returns:
        {
            "gene": str,                    # The gene searched for
            "found": bool,                  # Whether any matches found
            "match_count": int,             # Number of exact matches
            "context_snippets": List[str]   # 50 chars before/after each match
        }

    Implementation notes:
    - Uses word boundaries to avoid partial matches
    - Case-sensitive matching (gene symbols are typically uppercase)
    - Extracts context snippets showing surrounding text
    """
    # Build regex pattern with word boundaries for exact matching
    pattern = r'\b' + re.escape(gene_symbol) + r'\b'

    matches = list(re.finditer(pattern, abstract))

    context_snippets = []
    for match in matches:
        start = max(0, match.start() - 50)
        end = min(len(abstract), match.end() + 50)

        snippet = abstract[start:end]
        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(abstract):
            snippet = snippet + "..."

        context_snippets.append(snippet)

    return {
        "gene": gene_symbol,
        "found": len(matches) > 0,
        "match_count": len(matches),
        "context_snippets": context_snippets
    }


# =============================================================================
# Part 2: Agentic Function
# =============================================================================

def find_gene_mentions_llm(abstract: str, gene_symbol: str, llm_client: OpenAI) -> dict:
    """
    Use an LLM to find gene mentions and interpret their functional context.

    Args:
        abstract: Text to analyze
        gene_symbol: Primary gene of interest (e.g., "TP53")
        llm_client: OpenAI client instance

    Returns:
        {
            "gene": str,                    # Primary gene
            "mentions_found": List[str],    # All forms found
            "functional_summary": str,      # 2-3 sentence summary
            "key_findings": List[str]       # Bullet points of main findings
        }
    """
    system_prompt = """You are a biomedical text analysis expert. Your task is to:
1. Find all mentions of a specified gene, including aliases, synonyms, and indirect references
2. Summarize the functional/biological context of the gene in the text
3. Extract key research findings about the gene

Respond in JSON format with the following structure:
{
    "mentions_found": ["list", "of", "all", "name", "variants", "found"],
    "functional_summary": "2-3 sentence summary of the gene's role in this text",
    "key_findings": ["bullet point 1", "bullet point 2", "..."]
}

For TP53, recognize aliases like: p53, tumor protein p53, tumor protein 53, TP53 gene, p53 protein, tumor suppressor p53, etc.
For BRCA1, recognize: breast cancer gene 1, BRCA1 protein, etc.
For EGFR, recognize: epidermal growth factor receptor, ErbB1, HER1, etc.

Be thorough in finding all forms of the gene mentioned, even indirect references."""

    user_prompt = f"""Analyze the following scientific abstract for mentions of the gene {gene_symbol}.

Abstract:
{abstract}

Find all mentions (including aliases and synonyms), provide a functional summary, and list key findings."""

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

        return {
            "gene": gene_symbol,
            "mentions_found": result.get("mentions_found", []),
            "functional_summary": result.get("functional_summary", "No summary available"),
            "key_findings": result.get("key_findings", [])
        }

    except Exception as e:
        return {
            "gene": gene_symbol,
            "mentions_found": [],
            "functional_summary": f"Error analyzing text: {str(e)}",
            "key_findings": []
        }


# =============================================================================
# Part 3: Comparison Driver
# =============================================================================

def run_comparison_demo():
    """
    Demonstrate the difference between deterministic and agentic approaches.
    """
    print("=" * 70)
    print("MODULE 1 DEMO: Deterministic vs Agentic Gene Finding")
    print("=" * 70)
    print()

    # Initialize LLM client
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        return

    llm_client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

    # Test the client with a simple request
    print("Testing OpenAI API connection...")
    llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=5
    )
    print("OpenAI API connected successfully")
    print()

    # Get abstracts from PubMed
    print("Fetching abstracts from PubMed...")
    abstracts = query_pubmed("TP53 cancer mutation", max_results=3)
    if not abstracts:
        print("No results returned from PubMed. Please try again later.")
        return
    print(f"Retrieved {len(abstracts)} abstracts from PubMed")
    print()

    gene_symbol = "TP53"

    # Process each abstract
    for i, article in enumerate(abstracts, 1):
        title = article.get("title", "No title")
        pmid = article.get("pmid", "Unknown")
        abstract = article.get("abstract", "")

        if not abstract:
            print(f"Skipping article {pmid}: No abstract available")
            continue

        # Truncate title for display
        display_title = title[:60] + "..." if len(title) > 60 else title

        print("=" * 70)
        print(f"ABSTRACT {i}: {display_title}")
        print(f"PMID: {pmid}")
        print("=" * 70)
        print()

        # Run deterministic approach
        regex_result = find_gene_mentions_regex(abstract, gene_symbol)

        # Run agentic approach
        llm_result = find_gene_mentions_llm(abstract, gene_symbol, llm_client)

        # Display results
        print("DETERMINISTIC (Regex):")
        print(f"  Gene: {regex_result['gene']}")
        print(f"  Matches found: {regex_result['match_count']}")
        if regex_result['context_snippets']:
            print("  Context snippets:")
            for snippet in regex_result['context_snippets'][:3]:  # Limit to 3 snippets
                # Clean up whitespace for display
                clean_snippet = ' '.join(snippet.split())
                print(f"    - \"{clean_snippet}\"")
        else:
            print("  Context snippets: None")
        print()

        print("AGENTIC (LLM):")
        print(f"  Mentions found: {llm_result['mentions_found']}")
        print(f"  Functional summary: {llm_result['functional_summary']}")
        print("  Key findings:")
        for finding in llm_result['key_findings']:
            print(f"    - {finding}")
        print()

        # Comparison analysis
        print("COMPARISON:")

        # Find what regex missed
        regex_found = {gene_symbol}  # Regex only finds exact matches
        llm_found = set(llm_result['mentions_found'])

        # Identify aliases that regex missed
        missed_aliases = []
        for mention in llm_result['mentions_found']:
            if mention.lower() != gene_symbol.lower():
                # Check if this alias appears in the abstract
                if mention.lower() in abstract.lower():
                    missed_aliases.append(mention)

        if missed_aliases:
            # Count occurrences of missed terms
            missed_with_counts = []
            for alias in missed_aliases:
                # Case-insensitive count
                pattern = r'\b' + re.escape(alias) + r'\b'
                count = len(re.findall(pattern, abstract, re.IGNORECASE))
                if count > 0:
                    missed_with_counts.append(f'"{alias}" ({count} occurrence{"s" if count > 1 else ""})')

            if missed_with_counts:
                print(f"  Regex missed: {', '.join(missed_with_counts)}")
        else:
            print("  Regex found all explicit mentions")

        print("  LLM captured: Functional interpretation + biological context")

        print()
        print("=" * 70)
        print()

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY: Key Differences")
    print("=" * 70)
    print()
    print("DETERMINISTIC (Regex) Approach:")
    print("  + Fast and predictable")
    print("  + Easy to test and debug")
    print("  + Consistent output format")
    print("  - Misses synonyms and aliases")
    print("  - No understanding of context or meaning")
    print("  - Cannot interpret functional significance")
    print()
    print("AGENTIC (LLM) Approach:")
    print("  + Recognizes gene aliases and synonyms")
    print("  + Understands biological context")
    print("  + Can synthesize and summarize findings")
    print("  + Captures implicit references")
    print("  - Higher computational cost")
    print("  - Less predictable output")
    print("  - Requires API access")
    print()
    print("WHEN TO USE EACH:")
    print("  Regex: High-throughput screening, exact matching, audit trails")
    print("  LLM: Interpretation, synthesis, handling natural language variation")
    print()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    run_comparison_demo()
