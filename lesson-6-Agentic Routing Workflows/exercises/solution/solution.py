"""
Module 6 Exercise: Oncology Combination-Therapy Planner

A routing workflow that classifies oncology queries and directs them to appropriate
specialist agents for handling combination therapy planning.

Specialist Agents:
- StandardOfCareAgent: Current guidelines and approved regimens
- TrialSearchAgent: Relevant clinical trials
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ls_action_space.action_space import (
    query_pubmed,
    query_clinicaltrials,
    query_fda_labels
)


# Sample oncology queries for testing
ONCOLOGY_QUERIES = [
    # Standard of Care queries
    "What is the first-line treatment for EGFR-mutant NSCLC?",
    "What's the recommended regimen for HER2-positive metastatic breast cancer?",
    "Current standard of care for BRAF V600E melanoma?",

    # Trial Search queries
    "Are there any trials combining CDK4/6 inhibitors with immunotherapy in breast cancer?",
    "Find recruiting trials for KRAS G12C inhibitor combinations in NSCLC",
    "What trials are testing triplet therapy for microsatellite-stable colorectal cancer?",

    # Ambiguous / Multi-route queries
    "I have a patient with BRAF-mutant colorectal cancer who progressed on standard therapy - what are my options?",
    "Tell me everything about nivolumab plus ipilimumab in melanoma"
]


class StandardOfCareAgent:
    """
    Provides guideline-based treatment recommendations.
    Uses FDA labels and PubMed for evidence-based recommendations.
    """

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def process(self, query: str, context: Dict) -> Dict[str, Any]:
        """
        Provide standard of care information.

        Uses:
        - query_fda_labels for approved indications
        - query_pubmed for recent guideline publications
        - LLM knowledge for NCCN/ESMO guidelines

        Returns structured treatment recommendations.
        """
        cancer_type = context.get("cancer_type", "")
        biomarkers = context.get("biomarkers", [])
        drugs_mentioned = context.get("drugs_mentioned", [])

        # Gather evidence from external sources
        fda_info = {}
        for drug in drugs_mentioned[:3]:  # Limit API calls
            fda_result = query_fda_labels(drug)
            if not fda_result.get("error"):
                fda_info[drug] = fda_result

        # Search PubMed for recent guidelines
        pubmed_query = f"{cancer_type} treatment guidelines NCCN"
        if biomarkers:
            pubmed_query += f" {' '.join(biomarkers)}"
        pubmed_results = query_pubmed(pubmed_query, max_results=5)

        # Build context for LLM
        evidence_context = self._build_evidence_context(fda_info, pubmed_results)

        system_prompt = """You are an oncology clinical decision support specialist focusing on
standard of care recommendations based on NCCN, ESMO, and FDA-approved treatments.

Provide evidence-based treatment recommendations in a structured format.
Always include:
1. The specific cancer type and relevant biomarkers
2. Recommended regimens by line of therapy (first-line, second-line, etc.)
3. Evidence level (Category 1, 2A, 2B)
4. Key pivotal trials supporting the recommendations
5. Important considerations and caveats

IMPORTANT: Always include a caveat that recommendations should be verified against
current guidelines and individualized based on patient factors."""

        user_prompt = f"""Query: {query}

Cancer Type: {cancer_type}
Biomarkers: {', '.join(biomarkers) if biomarkers else 'Not specified'}
Drugs Mentioned: {', '.join(drugs_mentioned) if drugs_mentioned else 'None'}

{evidence_context}

Provide a comprehensive standard of care response. Return your response as JSON with this structure:
{{
    "cancer_type": "string",
    "biomarker_status": "string",
    "recommended_regimens": [
        {{
            "line": "First-line",
            "regimen": "Regimen name",
            "drugs": ["drug1", "drug2"],
            "evidence_level": "Category 1",
            "key_trial": "Trial name"
        }}
    ],
    "special_considerations": ["consideration1", "consideration2"],
    "guideline_source": "NCCN v1.2024",
    "last_updated": "2024",
    "caveats": "Important limitations and notes"
}}"""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Add source information
            result["sources"] = {
                "fda_labels": list(fda_info.keys()),
                "pubmed_articles": [{"pmid": a["pmid"], "title": a["title"]} for a in pubmed_results[:3]]
            }

            return result

        except Exception as e:
            return {
                "error": str(e),
                "cancer_type": cancer_type,
                "biomarker_status": ", ".join(biomarkers),
                "recommended_regimens": [],
                "caveats": "Error generating recommendations. Please consult current guidelines directly."
            }

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
    """
    Searches for relevant clinical trials.
    """

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def process(self, query: str, context: Dict) -> Dict[str, Any]:
        """
        Find relevant clinical trials.

        Uses:
        - query_clinicaltrials for trial search
        - LLM to summarize and rank relevance
        """
        cancer_type = context.get("cancer_type", "")
        biomarkers = context.get("biomarkers", [])
        drugs_mentioned = context.get("drugs_mentioned", [])

        # Build search query
        condition = cancer_type
        intervention = " ".join(drugs_mentioned) if drugs_mentioned else None

        # Query clinical trials - build search expression
        search_terms = [condition, "RECRUITING"]
        if intervention:
            search_terms.append(intervention)
        search_expr = " AND ".join(search_terms)

        # Query clinical trials
        trials_result = query_clinicaltrials(search_expr, max_results=10)
        all_trials = trials_result.get("studies", [])

        # If few results, try without intervention filter
        if len(all_trials) < 5 and intervention:
            search_expr_broad = f"{condition} AND RECRUITING"
            additional_result = query_clinicaltrials(search_expr_broad, max_results=10)
            additional_trials = additional_result.get("studies", [])
            # Add unique trials
            existing_ncts = {t.get("NCTId") for t in all_trials}
            for trial in additional_trials:
                if trial.get("NCTId") not in existing_ncts:
                    all_trials.append(trial)

        # Use LLM to score relevance and summarize
        result = self._analyze_trials(query, context, all_trials)

        return result

    def _analyze_trials(self, query: str, context: Dict, trials: List[Dict]) -> Dict[str, Any]:
        """Use LLM to analyze and rank trial relevance."""

        cancer_type = context.get("cancer_type", "")
        biomarkers = context.get("biomarkers", [])
        drugs_mentioned = context.get("drugs_mentioned", [])

        if not trials:
            return {
                "search_parameters": {
                    "condition": cancer_type,
                    "interventions": drugs_mentioned,
                    "phase_filter": None,
                    "status_filter": "RECRUITING"
                },
                "trials_found": 0,
                "top_trials": [],
                "search_suggestions": [
                    "Try broader search terms",
                    "Include completed trials",
                    "Search for related cancer types"
                ],
                "notes": "No recruiting trials found matching the criteria."
            }

        # Prepare trial summaries for LLM
        trial_summaries = []
        for i, trial in enumerate(trials[:15]):  # Limit to 15 for context
            interventions = trial.get("InterventionName") or []
            conditions = trial.get("Condition") or []
            summary = {
                "index": i,
                "NCTId": trial.get("NCTId"),
                "title": trial.get("BriefTitle"),
                "phase": trial.get("Phase"),
                "status": trial.get("OverallStatus"),
                "interventions": interventions[:5] if isinstance(interventions, list) else [interventions],
                "conditions": conditions[:3] if isinstance(conditions, list) else [conditions],
                "enrollment": trial.get("EnrollmentCount")
            }
            trial_summaries.append(summary)

        system_prompt = """You are a clinical trials specialist helping oncologists find relevant trials
for their patients. Analyze the provided trials and rank them by relevance to the query.

Consider:
1. Match between trial conditions and patient's cancer type
2. Match between trial interventions and drugs of interest
3. Biomarker relevance
4. Trial phase (Phase 3 often more relevant for treatment decisions)
5. Enrollment status and size"""

        user_prompt = f"""Query: {query}

Patient Context:
- Cancer Type: {cancer_type}
- Biomarkers: {', '.join(biomarkers) if biomarkers else 'Not specified'}
- Drugs of Interest: {', '.join(drugs_mentioned) if drugs_mentioned else 'Not specified'}

Available Trials:
{json.dumps(trial_summaries, indent=2)}

Analyze these trials and return JSON with this structure:
{{
    "ranked_trials": [
        {{
            "index": 0,
            "relevance_score": 0.95,
            "relevance_explanation": "Why this trial is relevant",
            "key_eligibility": ["Important inclusion criteria"]
        }}
    ],
    "search_suggestions": ["Alternative searches to try"],
    "notes": "Overall observations about the search results"
}}

Rank the top 5 most relevant trials by relevance_score (0-1)."""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            llm_result = json.loads(response.choices[0].message.content)

            # Build final result with full trial information
            top_trials = []
            for ranked in llm_result.get("ranked_trials", [])[:5]:
                idx = ranked.get("index", 0)
                if idx < len(trials):
                    trial = trials[idx]
                    interventions = trial.get("InterventionName") or []
                    locations = trial.get("LocationCountry") or []
                    top_trials.append({
                        "NCTId": trial.get("NCTId"),
                        "title": trial.get("BriefTitle"),
                        "phase": trial.get("Phase"),
                        "status": trial.get("OverallStatus"),
                        "interventions": interventions if isinstance(interventions, list) else [interventions],
                        "enrollment": trial.get("EnrollmentCount"),
                        "relevance_score": ranked.get("relevance_score", 0.5),
                        "relevance_explanation": ranked.get("relevance_explanation", ""),
                        "key_eligibility": ranked.get("key_eligibility", []),
                        "locations_summary": ", ".join(locations[:3]) if locations else "See trial for locations"
                    })

            return {
                "search_parameters": {
                    "condition": cancer_type,
                    "interventions": drugs_mentioned,
                    "phase_filter": None,
                    "status_filter": "RECRUITING"
                },
                "trials_found": len(trials),
                "top_trials": top_trials,
                "search_suggestions": llm_result.get("search_suggestions", []),
                "notes": llm_result.get("notes", "")
            }

        except Exception as e:
            # Return basic results without LLM ranking
            def format_trial(t):
                interventions = t.get("InterventionName") or []
                locations = t.get("LocationCountry") or []
                return {
                    "NCTId": t.get("NCTId"),
                    "title": t.get("BriefTitle"),
                    "phase": t.get("Phase"),
                    "status": t.get("OverallStatus"),
                    "interventions": interventions if isinstance(interventions, list) else [interventions],
                    "enrollment": t.get("EnrollmentCount"),
                    "relevance_score": 0.5,
                    "relevance_explanation": "Relevance scoring unavailable",
                    "key_eligibility": [],
                    "locations_summary": ", ".join(locations[:3]) if locations else "N/A"
                }

            return {
                "search_parameters": {
                    "condition": cancer_type,
                    "interventions": drugs_mentioned,
                    "phase_filter": None,
                    "status_filter": "RECRUITING"
                },
                "trials_found": len(trials),
                "top_trials": [format_trial(t) for t in trials[:5]],
                "search_suggestions": [],
                "notes": f"Error during relevance analysis: {str(e)}"
            }


class OncologyQueryRouter:
    """
    Routes oncology queries to appropriate specialist agents.
    """

    ROUTES = {
        "standard_of_care": """Guidelines-based questions about approved treatments,
            first-line/second-line therapy, NCCN recommendations, established regimens""",
        "trial_search": """Finding clinical trials, experimental combinations,
            recruiting studies, novel agents in development"""
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

        Returns classification with primary route, optional secondary route,
        confidence score, and extracted context.
        """
        system_prompt = """You are an oncology query classifier for a clinical decision support system.

Classify queries into one of these categories:
1. standard_of_care: Questions about approved/recommended treatments, guidelines,
   established regimens. Keywords: "first-line", "standard", "recommended", "approved",
   "guidelines", "NCCN", "what is the treatment"

2. trial_search: Questions about finding clinical trials, experimental therapies,
   recruiting studies. Keywords: "trial", "recruiting", "experimental", "novel",
   "find trials", "clinical trial", "testing"

For complex queries that span categories, identify both primary and secondary routes.
Extract cancer type, biomarkers, and drug names mentioned.

Return valid JSON with confidence scores between 0 and 1."""

        user_prompt = f"""Classify this oncology query:

"{query}"

Return JSON with this exact structure:
{{
    "primary_route": "standard_of_care|trial_search",
    "secondary_route": null or "standard_of_care|trial_search",
    "confidence": 0.0 to 1.0,
    "extracted_context": {{
        "cancer_type": "specific cancer type or empty string",
        "biomarkers": ["list of biomarkers mentioned"],
        "drugs_mentioned": ["list of drug names"],
        "query_intent": "brief description of what user wants"
    }},
    "reasoning": "explanation of classification"
}}"""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Validate and normalize the response
            valid_routes = ["standard_of_care", "trial_search"]

            if result.get("primary_route") not in valid_routes:
                result["primary_route"] = "standard_of_care"

            if result.get("secondary_route") and result["secondary_route"] not in valid_routes:
                result["secondary_route"] = None

            if result.get("secondary_route") == result.get("primary_route"):
                result["secondary_route"] = None

            # Ensure extracted_context exists
            if "extracted_context" not in result:
                result["extracted_context"] = {
                    "cancer_type": "",
                    "biomarkers": [],
                    "drugs_mentioned": [],
                    "query_intent": ""
                }

            return result

        except Exception as e:
            # Return default classification on error
            return {
                "primary_route": "standard_of_care",
                "secondary_route": None,
                "confidence": 0.5,
                "extracted_context": {
                    "cancer_type": "",
                    "biomarkers": [],
                    "drugs_mentioned": [],
                    "query_intent": query
                },
                "reasoning": f"Default classification due to error: {str(e)}"
            }

    def route(self, query: str) -> Dict[str, Any]:
        """
        Classify and dispatch to appropriate specialist(s).

        If secondary_route is identified, also get response from second specialist.
        """
        classification = self.classify(query)

        # Get primary response
        primary_route = classification["primary_route"]
        primary_specialist = self.specialists[primary_route]
        primary_response = primary_specialist.process(
            query,
            classification["extracted_context"]
        )

        result = {
            "query": query,
            "routing": classification,
            "primary_response": primary_response,
            "secondary_response": None
        }

        # If complex query, also get secondary response
        if classification.get("secondary_route"):
            secondary_route = classification["secondary_route"]
            secondary_specialist = self.specialists[secondary_route]
            result["secondary_response"] = secondary_specialist.process(
                query,
                classification["extracted_context"]
            )

        return result


def format_routing_info(routing: Dict) -> str:
    """Format routing classification for display."""
    lines = [
        "ROUTING:",
        f"├── Primary Route: {routing['primary_route']} (confidence: {routing.get('confidence', 'N/A')})",
        f"├── Secondary Route: {routing.get('secondary_route', 'None')}",
    ]

    context = routing.get("extracted_context", {})
    lines.extend([
        f"├── Cancer Type: {context.get('cancer_type', 'Not specified')}",
        f"├── Biomarkers: {', '.join(context.get('biomarkers', [])) or 'None'}",
        f"├── Drugs: {', '.join(context.get('drugs_mentioned', [])) or 'None'}",
        f"└── Intent: {context.get('query_intent', 'Not specified')}"
    ])

    return "\n".join(lines)


def format_standard_of_care_response(response: Dict) -> str:
    """Format StandardOfCareAgent response for display."""
    if response.get("error"):
        return f"Error: {response['error']}"

    lines = [
        "┌" + "─" * 68 + "┐",
        f"│ {response.get('cancer_type', 'Cancer Type Not Specified'):<66} │",
        f"│ Biomarkers: {response.get('biomarker_status', 'N/A'):<54} │",
        f"│ Guideline: {response.get('guideline_source', 'N/A'):<55} │",
        "├" + "─" * 68 + "┤",
    ]

    regimens = response.get("recommended_regimens", [])
    if regimens:
        for regimen in regimens[:4]:  # Limit display
            line = regimen.get("line", "")
            name = regimen.get("regimen", "")
            evidence = regimen.get("evidence_level", "")
            trial = regimen.get("key_trial", "")

            lines.append(f"│ {line} ({evidence}):".ljust(68) + " │")
            lines.append(f"│   • {name[:60]}".ljust(68) + " │")

            drugs = regimen.get("drugs", [])
            if drugs:
                drugs_str = ", ".join(drugs)[:58]
                lines.append(f"│     Drugs: {drugs_str}".ljust(68) + " │")

            if trial:
                lines.append(f"│     Trial: {trial[:55]}".ljust(68) + " │")
            lines.append("│" + " " * 68 + "│")
    else:
        lines.append("│ No specific regimens identified.".ljust(68) + " │")

    # Special considerations
    considerations = response.get("special_considerations", [])
    if considerations:
        lines.append("├" + "─" * 68 + "┤")
        lines.append("│ Special Considerations:".ljust(68) + " │")
        for consideration in considerations[:3]:
            lines.append(f"│   • {consideration[:62]}".ljust(68) + " │")

    # Caveats
    caveats = response.get("caveats", "")
    if caveats:
        lines.append("├" + "─" * 68 + "┤")
        lines.append("│ CAVEATS:".ljust(68) + " │")
        # Word wrap caveats
        words = caveats.split()
        current_line = "│   "
        for word in words:
            if len(current_line) + len(word) + 1 < 68:
                current_line += word + " "
            else:
                lines.append(current_line.ljust(68) + " │")
                current_line = "│   " + word + " "
        if current_line.strip() != "│":
            lines.append(current_line.ljust(68) + " │")

    lines.append("└" + "─" * 68 + "┘")
    return "\n".join(lines)


def format_trial_search_response(response: Dict) -> str:
    """Format TrialSearchAgent response for display."""
    lines = [
        "┌" + "─" * 68 + "┐",
        f"│ Clinical Trials Search Results".ljust(68) + " │",
    ]

    params = response.get("search_parameters", {})
    condition = params.get("condition", "N/A")
    interventions = ", ".join(params.get("interventions", [])) or "Any"
    lines.extend([
        f"│ Condition: {condition[:55]}".ljust(68) + " │",
        f"│ Interventions: {interventions[:51]}".ljust(68) + " │",
        f"│ Trials Found: {response.get('trials_found', 0)}".ljust(68) + " │",
        "├" + "─" * 68 + "┤",
    ])

    top_trials = response.get("top_trials", [])
    if top_trials:
        for i, trial in enumerate(top_trials[:5], 1):
            nct = trial.get("NCTId", "")
            title = trial.get("title", "")[:50]
            phase = trial.get("phase", "N/A")
            score = trial.get("relevance_score", 0)

            lines.append(f"│ {i}. {nct} (Relevance: {score:.2f})".ljust(68) + " │")
            lines.append(f"│    {title}...".ljust(68) + " │")
            lines.append(f"│    Phase: {phase}".ljust(68) + " │")

            interventions = trial.get("interventions", [])[:2]
            if interventions:
                int_str = ", ".join(interventions)[:50]
                lines.append(f"│    Interventions: {int_str}".ljust(68) + " │")

            explanation = trial.get("relevance_explanation", "")[:55]
            if explanation:
                lines.append(f"│    Why: {explanation}".ljust(68) + " │")

            lines.append("│" + " " * 68 + "│")
    else:
        lines.append("│ No trials found matching criteria.".ljust(68) + " │")

    # Search suggestions
    suggestions = response.get("search_suggestions", [])
    if suggestions:
        lines.append("├" + "─" * 68 + "┤")
        lines.append("│ Search Suggestions:".ljust(68) + " │")
        for suggestion in suggestions[:3]:
            lines.append(f"│   • {suggestion[:62]}".ljust(68) + " │")

    lines.append("└" + "─" * 68 + "┘")
    return "\n".join(lines)


def format_response(route_type: str, response: Dict) -> str:
    """Format response based on route type."""
    if route_type == "standard_of_care":
        return format_standard_of_care_response(response)
    elif route_type == "trial_search":
        return format_trial_search_response(response)
    else:
        return json.dumps(response, indent=2)


def process_query(router: OncologyQueryRouter, query: str, query_num: int) -> None:
    """Process a single query and display formatted results."""
    print(f"\nQUERY {query_num}:")
    print(f'"{query}"')
    print()

    result = router.route(query)

    # Display routing information
    print(format_routing_info(result["routing"]))
    print()

    # Display primary response
    primary_route = result["routing"]["primary_route"]
    route_label = primary_route.replace("_", " ").upper()
    print(f"PRIMARY RESPONSE ({route_label}):")
    print(format_response(primary_route, result["primary_response"]))

    # Display secondary response if present
    if result.get("secondary_response") and result["routing"].get("secondary_route"):
        secondary_route = result["routing"]["secondary_route"]
        route_label = secondary_route.replace("_", " ").upper()
        print()
        print(f"SECONDARY RESPONSE ({route_label}):")
        print(format_response(secondary_route, result["secondary_response"]))


def main():
    """Main entry point for the oncology query router."""
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

    # Initialize router
    router = OncologyQueryRouter(client)

    # Print header
    print("=" * 70)
    print("ONCOLOGY COMBINATION-THERAPY PLANNER")
    print("=" * 70)

    # Process sample queries
    # Select a subset for demonstration (to avoid excessive API calls)
    demo_queries = [
        ONCOLOGY_QUERIES[0],  # Standard of care: EGFR NSCLC
        ONCOLOGY_QUERIES[3],  # Trial search: CDK4/6 + immunotherapy
        ONCOLOGY_QUERIES[6],  # Complex: BRAF CRC progression
    ]

    for i, query in enumerate(demo_queries, 1):
        process_query(router, query, i)
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
