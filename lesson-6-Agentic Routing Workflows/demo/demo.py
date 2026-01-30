"""
Module 6 Demo: Omics Core Request Router

A routing workflow that classifies omics core facility requests and dispatches
them to appropriate specialist agents (DNA Sequencing, Proteomics, Metabolomics).

Learning Objective: Implement a routing workflow that classifies a user query and directs it to the appropriate specialist agent, orchestrating sub-tasks as needed
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent / ".env")


# Sample requests for testing
SAMPLE_REQUESTS = [
    "We need whole exome sequencing for 48 tumor-normal pairs with 100x coverage",
    "Can you run TMT-labeled quantitative proteomics on our cell lysates?",
    "We're studying metabolic reprogramming and need untargeted metabolomics of polar metabolites",
    "I need RNA-seq for 24 samples with differential expression analysis",
]


# =============================================================================
# BASE SPECIALIST CLASS
# =============================================================================

class OmicsSpecialist:
    """
    Base class for omics specialists. Each specialist handles a specific type
    of omics request and provides service recommendations.
    """

    # Subclasses override these
    SPECIALTY_NAME = "Generic"
    RESPONSE_SCHEMA = {}
    DOMAIN_KNOWLEDGE = ""

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def process(self, request: str, entities: Dict) -> Dict[str, Any]:
        """Process a request and return service recommendations."""
        system_prompt = f"""You are a {self.SPECIALTY_NAME} specialist at an omics core facility.

Based on the user's request and extracted entities, provide a detailed service recommendation.

{self.DOMAIN_KNOWLEDGE}

Return valid JSON with this structure:
{json.dumps(self.RESPONSE_SCHEMA, indent=2)}

Provide realistic, professional recommendations based on current industry standards."""

        user_prompt = f"""Request: {request}

Extracted entities: {json.dumps(entities, indent=2)}

Please provide a detailed service recommendation."""

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)


class DNASeqSpecialist(OmicsSpecialist):
    """Handles DNA/RNA sequencing requests."""

    SPECIALTY_NAME = "DNA/RNA sequencing"
    RESPONSE_SCHEMA = {
        "service_type": "e.g., Whole Exome Sequencing, RNA-seq",
        "recommended_parameters": {
            "platform": "e.g., Illumina NovaSeq 6000",
            "read_length": "e.g., 2x150bp",
            "coverage": "e.g., 100x target coverage",
            "library_prep": "e.g., Twist Exome 2.0"
        },
        "sample_requirements": {
            "input_amount": "e.g., 200ng DNA per sample",
            "quality_requirements": "e.g., DIN score ≥4",
            "format": "e.g., DNA in TE buffer"
        },
        "deliverables": ["list of what client will receive"],
        "estimated_turnaround": "e.g., 4-6 weeks",
        "estimated_cost_range": "e.g., $350-450 per sample",
        "notes": "additional recommendations"
    }


class ProteomicsSpecialist(OmicsSpecialist):
    """Handles proteomics requests."""

    SPECIALTY_NAME = "proteomics"
    RESPONSE_SCHEMA = {
        "service_type": "e.g., TMT-16plex Quantitative Proteomics, IP-MS",
        "recommended_workflow": {
            "sample_prep": "e.g., In-solution digestion",
            "fractionation": "e.g., High-pH reverse phase",
            "ms_method": "e.g., DDA on Orbitrap Eclipse",
            "quantification": "e.g., TMT reporter ion quantification"
        },
        "sample_requirements": {
            "input_amount": "e.g., 100ug protein per sample",
            "format": "e.g., Cell pellet or lysate",
            "special_requirements": "any special handling notes"
        },
        "analysis_included": ["list of bioinformatics deliverables"],
        "estimated_turnaround": "e.g., 3-4 weeks",
        "estimated_cost_range": "e.g., $500-800 per sample",
        "notes": "additional recommendations"
    }


class MetabolomicsSpecialist(OmicsSpecialist):
    """Handles metabolomics/lipidomics requests."""

    SPECIALTY_NAME = "metabolomics/lipidomics"
    RESPONSE_SCHEMA = {
        "service_type": "e.g., Untargeted Polar Metabolomics, Lipidomics",
        "recommended_workflow": {
            "extraction": "e.g., 80% methanol extraction",
            "platform": "e.g., HILIC-MS/MS",
            "polarity": "e.g., Positive and negative mode",
            "coverage": "e.g., ~500 annotated metabolites"
        },
        "sample_requirements": {
            "sample_type": "e.g., Cells, tissue, serum",
            "amount": "e.g., 1M cells or 20mg tissue",
            "collection_notes": "e.g., Flash freeze immediately"
        },
        "analysis_included": ["list of bioinformatics deliverables"],
        "estimated_turnaround": "e.g., 3-4 weeks",
        "estimated_cost_range": "e.g., $200-400 per sample",
        "notes": "additional recommendations"
    }


# =============================================================================
# ROUTER
# =============================================================================

class OmicsRequestRouter:
    """
    Router that classifies requests and dispatches to specialist agents.

    This is the core of the routing pattern:
    1. Classify incoming request into a category
    2. Extract relevant entities
    3. Dispatch to appropriate specialist
    """

    ROUTES = {
        "dna_sequencing": "DNA/RNA sequencing requests (WES, WGS, RNA-seq, targeted panels)",
        "proteomics": "Protein analysis requests (MS, TMT, IP-MS, phosphoproteomics)",
        "metabolomics": "Small molecule analysis requests (metabolomics, lipidomics)"
    }

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client
        self.specialists = {
            "dna_sequencing": DNASeqSpecialist(llm_client),
            "proteomics": ProteomicsSpecialist(llm_client),
            "metabolomics": MetabolomicsSpecialist(llm_client)
        }

    def classify(self, request: str) -> Dict[str, Any]:
        """
        Classify the request into one of the three categories.
        Returns classification result with route, confidence, entities, and reasoning.
        """
        system_prompt = """You are a request classifier for an omics core facility.

Classify each request into exactly ONE category:
- dna_sequencing: Any DNA or RNA sequencing (WES, WGS, RNA-seq, targeted panels)
- proteomics: Any protein analysis (mass spectrometry, TMT/iTRAQ, IP-MS, phosphoproteomics)
- metabolomics: Any small molecule analysis (metabolomics, lipidomics, targeted metabolites)

Return valid JSON with this structure:
{
    "route": "dna_sequencing" | "proteomics" | "metabolomics",
    "confidence": 0.0-1.0,
    "extracted_entities": {
        "sample_type": "string or null",
        "sample_count": "integer or null",
        "technique_requested": "string",
        "analysis_goals": ["list of what they want to learn"]
    },
    "reasoning": "Brief explanation of classification decision"
}"""

        user_prompt = f'Classify this omics core facility request:\n\n"{request}"'

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    def route(self, request: str) -> Dict[str, Any]:
        """
        Classify request and dispatch to appropriate specialist.
        """
        # 1. Classify the request
        classification = self.classify(request)

        # 2. Check confidence and handle low-confidence cases
        if classification.get("confidence", 0) < 0.6:
            return {
                "original_request": request,
                "routing": classification,
                "specialist_response": None,
                "clarification_needed": True,
                "clarification_message": (
                    f"This request is ambiguous (confidence: {classification.get('confidence', 0):.2f}). "
                    "Please clarify the type of analysis needed."
                )
            }

        # 3. Dispatch to specialist
        route = classification["route"]
        specialist = self.specialists[route]
        entities = classification.get("extracted_entities", {})
        specialist_response = specialist.process(request, entities)

        return {
            "original_request": request,
            "routing": classification,
            "specialist_response": specialist_response,
            "clarification_needed": False
        }


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_result(result: Dict[str, Any]) -> str:
    """Format a routing result for display."""
    lines = []
    routing = result["routing"]

    # Routing decision
    lines.append("ROUTING DECISION:")
    lines.append(f"  Route: {routing.get('route', 'unknown')}")
    lines.append(f"  Confidence: {routing.get('confidence', 0):.2f}")
    lines.append(f"  Reasoning: {routing.get('reasoning', 'N/A')}")

    entities = routing.get("extracted_entities", {})
    if entities.get("technique_requested"):
        lines.append(f"  Technique: {entities['technique_requested']}")
    if entities.get("sample_count"):
        lines.append(f"  Samples: {entities['sample_count']}")

    # Specialist response
    if result.get("clarification_needed"):
        lines.append(f"\n  CLARIFICATION NEEDED: {result['clarification_message']}")
    elif result.get("specialist_response"):
        resp = result["specialist_response"]
        lines.append(f"\nSPECIALIST RESPONSE:")
        lines.append(f"  Service: {resp.get('service_type', 'N/A')}")
        lines.append(f"  Turnaround: {resp.get('estimated_turnaround', 'N/A')}")
        lines.append(f"  Cost: {resp.get('estimated_cost_range', 'N/A')}")
        if resp.get("notes"):
            lines.append(f"  Notes: {resp['notes'][:100]}...")

    return "\n".join(lines)


def run_routing_demo(requests: List[str], llm_client: OpenAI) -> List[Dict[str, Any]]:
    """Process all sample requests through the router."""
    print("=" * 70)
    print("OMICS CORE REQUEST ROUTER DEMO")
    print("=" * 70)
    print("\nThis demo shows how a routing workflow:")
    print("  1. Classifies incoming requests")
    print("  2. Extracts relevant entities")
    print("  3. Dispatches to specialized handlers")
    print()

    router = OmicsRequestRouter(llm_client)
    results = []

    for i, request in enumerate(requests, 1):
        print(f"\n{'─' * 70}")
        print(f"REQUEST {i}: \"{request}\"")
        print("─" * 70)

        result = router.route(request)
        results.append(result)
        print(format_result(result))

    # Summary table
    print(f"\n{'=' * 70}")
    print("ROUTING SUMMARY")
    print("=" * 70)
    print(f"{'Request (truncated)':<40} {'Route':<18} {'Confidence':>10}")
    print("-" * 70)

    for result in results:
        request = result["original_request"][:37]
        if len(result["original_request"]) > 37:
            request += "..."
        route = result["routing"].get("route", "unknown")
        conf = result["routing"].get("confidence", 0)
        print(f"{request:<40} {route:<18} {conf:>10.2f}")

    return results


def main():
    """Main entry point for the demo."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")
    run_routing_demo(SAMPLE_REQUESTS, client)


if __name__ == "__main__":
    main()
