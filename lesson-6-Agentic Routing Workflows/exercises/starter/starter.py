"""
Module 6 Exercise: Oncology Combination-Therapy Planner

A routing workflow that classifies oncology queries and directs them to:
- StandardOfCareAgent: Current guidelines and approved regimens
- TrialSearchAgent: Relevant clinical trials

Learning Objective:
    Implement a routing workflow that classifies queries and directs them to
    appropriate specialist agents.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from ls_action_space.action_space import query_pubmed, query_clinicaltrials, query_fda_labels


ONCOLOGY_QUERIES = [
    "What is the first-line treatment for EGFR-mutant NSCLC?",
    "What's the recommended regimen for HER2-positive metastatic breast cancer?",
    "Are there any trials combining CDK4/6 inhibitors with immunotherapy in breast cancer?",
    "Find recruiting trials for KRAS G12C inhibitor combinations in NSCLC",
    "I have a patient with BRAF-mutant colorectal cancer who progressed - what are my options?",
]


class StandardOfCareAgent:
    """Provides guideline-based treatment recommendations using FDA labels and PubMed."""

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def process(self, query: str, context: Dict) -> Dict[str, Any]:
        """
        Provide standard of care information.

        Returns: {cancer_type, biomarker_status, recommended_regimens: [{line, regimen,
        drugs, evidence_level, key_trial}], special_considerations, guideline_source,
        caveats, sources}
        """
        cancer_type = context.get("cancer_type", "")
        biomarkers = context.get("biomarkers", [])
        drugs_mentioned = context.get("drugs_mentioned", [])

        # TODO: Step 1 - Query FDA labels for each drug mentioned (limit to first 3)

        # TODO: Step 2 - Search PubMed for recent treatment guidelines
        #       Query: "{cancer_type} treatment guidelines NCCN"

        # TODO: Step 3 - Build evidence context string from FDA info and PubMed

        # TODO: Step 4 - Create system prompt for oncology clinical decision support

        # TODO: Step 5 - Create user prompt with query, cancer type, biomarkers,
        #       drugs mentioned, and evidence context; request JSON response

        # TODO: Step 6 - Call LLM with temperature=0.3, response_format=json_object

        # TODO: Step 7 - Parse JSON response and add sources information

        # TODO: Step 8 - Handle exceptions with error fallback
        pass

    def _build_evidence_context(self, fda_info: Dict, pubmed_results: List) -> str:
        """Build context string from gathered evidence."""
        context_parts = []
        if fda_info:
            context_parts.append("FDA Label Information:")
            for drug, info in fda_info.items():
                context_parts.append(f"\n{drug}:")
                if info.get("indications_and_usage"):
                    context_parts.append(f"  Indications: {info['indications_and_usage'][:500]}")
        if pubmed_results:
            context_parts.append("\n\nRecent PubMed Articles:")
            for article in pubmed_results[:3]:
                context_parts.append(f"- {article['title']} (PMID: {article['pmid']})")
        return "\n".join(context_parts) if context_parts else "No additional evidence gathered."


class TrialSearchAgent:
    """Searches for relevant clinical trials."""

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def process(self, query: str, context: Dict) -> Dict[str, Any]:
        """
        Find relevant clinical trials.

        Returns: {search_parameters, trials_found, top_trials: [{NCTId, title, phase,
        status, interventions, relevance_score, relevance_explanation}],
        search_suggestions, notes}
        """
        cancer_type = context.get("cancer_type", "")
        biomarkers = context.get("biomarkers", [])
        drugs_mentioned = context.get("drugs_mentioned", [])

        # TODO: Step 1 - Build search expression for clinical trials
        #       Example: "NSCLC AND RECRUITING AND pembrolizumab"

        # TODO: Step 2 - Query clinical trials using query_clinicaltrials()

        # TODO: Step 3 - If few results (<5), try broader search without intervention filter

        # TODO: Step 4 - Use LLM to analyze and rank trial relevance via _analyze_trials()

        # TODO: Step 5 - Return structured result dict
        pass

    def _analyze_trials(self, query: str, context: Dict, trials: List[Dict]) -> Dict[str, Any]:
        """Use LLM to analyze and rank trial relevance."""
        cancer_type = context.get("cancer_type", "")
        drugs_mentioned = context.get("drugs_mentioned", [])

        if not trials:
            return {
                "search_parameters": {"condition": cancer_type, "interventions": drugs_mentioned,
                                      "phase_filter": None, "status_filter": "RECRUITING"},
                "trials_found": 0, "top_trials": [],
                "search_suggestions": ["Try broader terms", "Include completed trials"],
                "notes": "No recruiting trials found."
            }

        # TODO: Step 1 - Prepare trial summaries for LLM (limit to 15)
        #       Extract: NCTId, BriefTitle, Phase, OverallStatus, InterventionName

        # TODO: Step 2 - Create system prompt for clinical trials specialist

        # TODO: Step 3 - Create user prompt with query, context, and trial summaries
        #       Request ranked_trials with relevance_score, explanation

        # TODO: Step 4 - Call LLM and parse JSON response

        # TODO: Step 5 - Build final result with full trial information

        # TODO: Step 6 - Handle exceptions with basic unranked results
        pass


class OncologyQueryRouter:
    """Routes oncology queries to appropriate specialist agents."""

    ROUTES = {
        "standard_of_care": "Questions about approved/recommended treatments and guidelines",
        "trial_search": "Questions about finding clinical trials"
    }

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client
        self.specialists = {
            "standard_of_care": StandardOfCareAgent(llm_client),
            "trial_search": TrialSearchAgent(llm_client)
        }

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify oncology query into route category.

        Returns: {primary_route, secondary_route, confidence, extracted_context:
        {cancer_type, biomarkers, drugs_mentioned, query_intent}, reasoning}
        """
        # TODO: Step 1 - Create system prompt for oncology query classifier
        #       Define routes with keywords:
        #       - standard_of_care: "first-line", "standard", "recommended", "approved"
        #       - trial_search: "trial", "recruiting", "experimental", "find trials"

        # TODO: Step 2 - Create user prompt asking LLM to classify the query
        #       Request JSON with: primary_route, secondary_route, confidence,
        #       extracted_context, reasoning

        # TODO: Step 3 - Call LLM with temperature=0.2, response_format=json_object

        # TODO: Step 4 - Parse JSON response

        # TODO: Step 5 - Validate: primary_route valid, secondary_route valid or None

        # TODO: Step 6 - Handle exceptions with default classification
        pass

    def route(self, query: str) -> Dict[str, Any]:
        """Classify and dispatch to appropriate specialist(s)."""
        classification = self.classify(query)
        if classification is None:
            classification = {
                "primary_route": "standard_of_care", "secondary_route": None,
                "confidence": 0.5,
                "extracted_context": {"cancer_type": "", "biomarkers": [],
                                      "drugs_mentioned": [], "query_intent": query},
                "reasoning": "TODO: Implement classify()"
            }

        # Primary response
        primary_route = classification["primary_route"]
        primary_response = self.specialists[primary_route].process(
            query, classification["extracted_context"]
        )
        if primary_response is None:
            primary_response = {"error": "TODO: Implement process()"}

        result = {"query": query, "routing": classification,
                  "primary_response": primary_response, "secondary_response": None}

        # Secondary response for complex queries
        if classification.get("secondary_route"):
            secondary_route = classification["secondary_route"]
            secondary_response = self.specialists[secondary_route].process(
                query, classification["extracted_context"]
            )
            result["secondary_response"] = secondary_response if secondary_response else {"error": "TODO"}

        return result


def display_result(result: Dict) -> None:
    """Display formatted result."""
    routing = result["routing"]
    ctx = routing.get("extracted_context", {})
    print(f"  Route: {routing['primary_route']} (conf: {routing.get('confidence', 'N/A')})")
    print(f"  Cancer: {ctx.get('cancer_type', 'N/A')} | Biomarkers: {', '.join(ctx.get('biomarkers', [])) or 'None'}")

    primary = result["primary_response"]
    if primary.get("error"):
        print(f"  Response: {primary['error']}")
    elif routing["primary_route"] == "standard_of_care":
        regimens = primary.get("recommended_regimens", [])[:2]
        for r in regimens:
            print(f"    - {r.get('line', 'N/A')}: {r.get('regimen', 'N/A')} ({r.get('evidence_level', '')})")
    else:
        trials = primary.get("top_trials", [])[:3]
        for t in trials:
            print(f"    - {t.get('NCTId', 'N/A')}: {t.get('title', 'N/A')[:50]}...")


def main():
    """Main entry point."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")
    router = OncologyQueryRouter(client)

    print("Module 6: Oncology Combination-Therapy Planner")
    print("Routes: standard_of_care, trial_search\n")

    demo_queries = [ONCOLOGY_QUERIES[0], ONCOLOGY_QUERIES[2], ONCOLOGY_QUERIES[4]]
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*60}\nQuery {i}: {query[:60]}...")
        result = router.route(query)
        display_result(result)


if __name__ == "__main__":
    main()
