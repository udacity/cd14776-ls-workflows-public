"""
Module 4 Demo: Cell Line Authentication Pipeline

A sequential agentic workflow demonstrating three agents working in sequence:
1. STR-Check Agent - Validates cell line identity via STR profile matching
2. Contamination-Audit Agent - Assesses contamination risk
3. Summary Agent - Generates authentication report

Learning Objective: Understand sequential agent workflows where each agent's
output becomes input for the next agent.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict
from datetime import datetime, timedelta

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent / ".env")


# =============================================================================
# BASE AGENT
# =============================================================================

class BaseAgent(ABC):
    """Base class for all agents in the workflow."""

    def __init__(self, name: str, llm_client: OpenAI):
        self.name = name
        self.llm = llm_client

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's task."""
        pass

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Helper method to call the LLM."""
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content


# =============================================================================
# STR CHECK AGENT
# =============================================================================

class STRCheckAgent(BaseAgent):
    """Agent 1: Validates cell line identity via STR profile matching."""

    # Reference STR profiles for common cell lines
    REFERENCE_DATABASE = {
        "HeLa": {"D5S818": "11,12", "D13S317": "12,13.3", "D7S820": "8,12", "vWA": "16,18", "TH01": "7", "TPOX": "8,12"},
        "MCF-7": {"D5S818": "11,12", "D13S317": "11", "D7S820": "8,9", "vWA": "14,15", "TH01": "6", "TPOX": "9,12"},
        "A549": {"D5S818": "11", "D13S317": "11", "D7S820": "8,11", "vWA": "14", "TH01": "8,9.3", "TPOX": "8,11"},
    }

    def __init__(self, llm_client: OpenAI):
        super().__init__("STR-Check Agent", llm_client)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare submitted STR profile against reference database."""
        cell_line = input_data.get("cell_line_name", "Unknown")
        submitted = input_data.get("submitted_str_profile", {})

        # Find best match across all references
        best_match, best_pct = None, 0
        for name, ref in self.REFERENCE_DATABASE.items():
            matches = sum(1 for m, v in ref.items() if submitted.get(m) == v)
            pct = (matches / len(ref)) * 100 if ref else 0
            if pct > best_pct:
                best_match, best_pct = name, pct

        # Check against claimed identity
        claimed_ref = self.REFERENCE_DATABASE.get(cell_line, {})
        if claimed_ref:
            matches = sum(1 for m, v in claimed_ref.items() if submitted.get(m) == v)
            match_pct = (matches / len(claimed_ref)) * 100
        else:
            match_pct = best_pct

        # Determine status
        if match_pct == 100:
            status, confidence = "Match", "High"
        elif match_pct >= 80:
            status, confidence = "Partial Match", "Medium"
        else:
            status, confidence = "No Match", "Low"

        # LLM interpretation
        interpretation = self._call_llm(
            "You are a cell line authentication expert. Provide a 2-sentence interpretation.",
            f"Cell line: {cell_line}, Match: {match_pct}%, Best match: {best_match} ({best_pct}%)"
        )

        return {
            "cell_line_name": cell_line,
            "str_match_status": status,
            "match_percentage": match_pct,
            "best_reference_match": best_match,
            "confidence": confidence,
            "interpretation": interpretation
        }


# =============================================================================
# CONTAMINATION AUDIT AGENT
# =============================================================================

class ContaminationAuditAgent(BaseAgent):
    """Agent 2: Assesses contamination risk based on STR results and test data."""

    def __init__(self, llm_client: OpenAI):
        super().__init__("Contamination-Audit Agent", llm_client)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess contamination risk from multiple data sources."""
        str_check = input_data.get("str_check", {})
        mycoplasma = input_data.get("mycoplasma_test", {}).get("result", "Unknown")
        passage = input_data.get("passage_number", 0)
        morphology = input_data.get("morphology_notes", "")

        risks = []
        recommendations = []

        # Mycoplasma assessment
        if mycoplasma == "Positive":
            risks.append("Mycoplasma contamination detected")
            recommendations.append("Quarantine and discard or treat cells")
        elif mycoplasma == "Indeterminate":
            risks.append("Mycoplasma test inconclusive")
            recommendations.append("Repeat mycoplasma testing with PCR")

        # STR assessment
        match_pct = str_check.get("match_percentage", 0)
        if match_pct < 80:
            risks.append(f"STR profile only {match_pct}% match")
            best = str_check.get("best_reference_match")
            if best != str_check.get("cell_line_name"):
                risks.append(f"Possible misidentification - matches {best}")
                recommendations.append(f"Verify identity - may be {best}")

        # Passage assessment
        if passage > 30:
            risks.append(f"High passage ({passage}) - drift risk")
            recommendations.append("Thaw fresh low-passage stock")

        # Morphology assessment
        if any(term in morphology.lower() for term in ["mixed", "floating", "unusual"]):
            risks.append(f"Morphology concern: {morphology}")

        # Determine overall risk
        if mycoplasma == "Positive":
            risk_level, can_proceed = "Critical", False
        elif match_pct < 80 or mycoplasma == "Indeterminate":
            risk_level, can_proceed = "High", False
        elif match_pct < 95 or passage > 30:
            risk_level, can_proceed = "Medium", True
            recommendations.append("Proceed with caution")
        else:
            risk_level, can_proceed = "Low", True
            recommendations.append("Continue with scheduled experiments")

        # LLM reasoning
        reasoning = self._call_llm(
            "You are a cell culture contamination expert. Provide a 2-sentence risk assessment.",
            f"Mycoplasma: {mycoplasma}, STR match: {match_pct}%, Passage: {passage}, Risks: {risks}"
        )

        return {
            "contamination_risk_level": risk_level,
            "mycoplasma_status": mycoplasma,
            "specific_risks": risks,
            "recommendations": recommendations,
            "can_proceed": can_proceed,
            "reasoning": reasoning
        }


# =============================================================================
# SUMMARY AGENT
# =============================================================================

class SummaryAgent(BaseAgent):
    """Agent 3: Generates authentication report from all previous results."""

    def __init__(self, llm_client: OpenAI):
        super().__init__("Summary Agent", llm_client)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive authentication report."""
        cell_line = input_data.get("cell_line_name", "Unknown")
        str_check = input_data.get("str_check", {})
        contamination = input_data.get("contamination_audit", {})

        # Determine authentication status
        str_status = str_check.get("str_match_status", "Unknown")
        can_proceed = contamination.get("can_proceed", False)
        risk_level = contamination.get("contamination_risk_level", "Unknown")

        if str_status == "Match" and can_proceed and risk_level == "Low":
            auth_status = "AUTHENTICATED"
        elif can_proceed:
            auth_status = "CONDITIONAL"
        else:
            auth_status = "FAILED"

        # Generate executive summary via LLM
        summary = self._call_llm(
            "You are a cell line authentication report writer. Write a 2-sentence executive summary.",
            f"Cell line: {cell_line}, Status: {auth_status}, STR: {str_status} ({str_check.get('match_percentage')}%), Risk: {risk_level}"
        )

        # Build report
        report_date = datetime.now().strftime("%Y-%m-%d")
        valid_until = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")

        report = f"""
{'='*60}
CELL LINE AUTHENTICATION REPORT
{'='*60}

Cell Line: {cell_line}
Report Date: {report_date}
Status: {auth_status}

EXECUTIVE SUMMARY
{summary}

STR PROFILE ANALYSIS
  Match Status: {str_status}
  Match Percentage: {str_check.get('match_percentage', 0)}%
  Best Reference Match: {str_check.get('best_reference_match', 'Unknown')}
  Confidence: {str_check.get('confidence', 'Unknown')}
  Interpretation: {str_check.get('interpretation', 'N/A')}

CONTAMINATION ASSESSMENT
  Risk Level: {risk_level}
  Mycoplasma: {contamination.get('mycoplasma_status', 'Unknown')}
  Can Proceed: {'Yes' if can_proceed else 'No'}
  Assessment: {contamination.get('reasoning', 'N/A')}

IDENTIFIED RISKS
{chr(10).join(f'  - {r}' for r in contamination.get('specific_risks', ['None'])) or '  - None'}

RECOMMENDATIONS
{chr(10).join(f'  - {r}' for r in contamination.get('recommendations', [])) or '  - None'}

AUTHENTICATION DECISION
  Status: {auth_status}
  Valid Until: {valid_until}

{'='*60}
"""

        return {
            "report_text": report,
            "authentication_status": auth_status,
            "report_date": report_date
        }


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class CellLineAuthenticationPipeline:
    """Orchestrates sequential execution of all agents."""

    def __init__(self, llm_client: OpenAI):
        self.str_agent = STRCheckAgent(llm_client)
        self.contamination_agent = ContaminationAuditAgent(llm_client)
        self.summary_agent = SummaryAgent(llm_client)

    def run(self, cell_line_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full authentication pipeline."""
        import time
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"CELL LINE AUTHENTICATION PIPELINE")
        print(f"Cell Line: {cell_line_data['cell_line_name']}")
        print(f"{'='*60}")

        # Step 1: STR Check
        print(f"\n[Step 1/3] STR-Check Agent...")
        str_result = self.str_agent.run(cell_line_data)
        print(f"  -> {str_result['str_match_status']} ({str_result['match_percentage']}%)")

        # Step 2: Contamination Audit (receives STR result)
        print(f"\n[Step 2/3] Contamination-Audit Agent...")
        contamination_input = {**cell_line_data, "str_check": str_result}
        contamination_result = self.contamination_agent.run(contamination_input)
        print(f"  -> Risk: {contamination_result['contamination_risk_level']}, Proceed: {contamination_result['can_proceed']}")

        # Step 3: Summary Report (receives all results)
        print(f"\n[Step 3/3] Summary Agent...")
        summary_input = {
            "cell_line_name": cell_line_data["cell_line_name"],
            "str_check": str_result,
            "contamination_audit": contamination_result
        }
        summary_result = self.summary_agent.run(summary_input)

        elapsed_ms = int((time.time() - start_time) * 1000)

        print(summary_result["report_text"])
        print(f"Pipeline completed in {elapsed_ms}ms")

        return {
            "final_report": summary_result["report_text"],
            "authentication_status": summary_result["authentication_status"],
            "execution_time_ms": elapsed_ms
        }


# =============================================================================
# TEST DATA
# =============================================================================

CLEAN_SAMPLE = {
    "cell_line_name": "HeLa",
    "submitted_str_profile": {
        "D5S818": "11,12", "D13S317": "12,13.3", "D7S820": "8,12",
        "vWA": "16,18", "TH01": "7", "TPOX": "8,12"
    },
    "passage_number": 15,
    "mycoplasma_test": {"method": "PCR", "result": "Negative"},
    "morphology_notes": "Typical epithelial morphology, adherent"
}

PROBLEMATIC_SAMPLE = {
    "cell_line_name": "MCF-7",  # Claims MCF-7 but profile matches HeLa
    "submitted_str_profile": {
        "D5S818": "11,12", "D13S317": "12,13.3", "D7S820": "8,12",
        "vWA": "16,18", "TH01": "7", "TPOX": "8,12"
    },
    "passage_number": 45,
    "mycoplasma_test": {"method": "DAPI", "result": "Indeterminate"},
    "morphology_notes": "Mixed morphology, some floating cells"
}


def main():
    """Run the cell line authentication pipeline demo."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")
    pipeline = CellLineAuthenticationPipeline(client)

    print("="*60)
    print("SEQUENTIAL AGENTIC WORKFLOW DEMO")
    print("="*60)
    print("\nThis demo shows three agents working in sequence:")
    print("  1. STR-Check Agent -> validates identity")
    print("  2. Contamination-Audit Agent -> assesses risk (uses STR result)")
    print("  3. Summary Agent -> generates report (uses all results)")

    # Test Case 1: Clean sample
    print("\n\n" + "#"*60)
    print("TEST CASE 1: CLEAN SAMPLE (HeLa)")
    print("#"*60)
    result1 = pipeline.run(CLEAN_SAMPLE)
    print(f"\n>>> Status: {result1['authentication_status']}")

    # Test Case 2: Problematic sample
    print("\n\n" + "#"*60)
    print("TEST CASE 2: PROBLEMATIC SAMPLE (Claims MCF-7, Actually HeLa)")
    print("#"*60)
    result2 = pipeline.run(PROBLEMATIC_SAMPLE)
    print(f"\n>>> Status: {result2['authentication_status']}")

    # Summary
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print(f"Clean sample: {result1['authentication_status']}")
    print(f"Problematic sample: {result2['authentication_status']}")


if __name__ == "__main__":
    main()
