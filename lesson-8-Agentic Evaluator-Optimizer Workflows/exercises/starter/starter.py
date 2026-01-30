"""
Module 8 Exercise: Trial Eligibility Criteria Refiner

An iterative evaluator-optimizer workflow with three specialized evaluators:
1. GCP (Good Clinical Practice) Compliance
2. Enrollment Feasibility
3. Demographic Fairness

Learning Objective:
Implement an evaluator-optimizer workflow where an AI agent's output is 
iteratively critiqued and refined by another agent to meet specific criteria.

"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


STUDY_CONTEXT = {
    "indication": "Advanced Non-Small Cell Lung Cancer (NSCLC)",
    "study_phase": "Phase 2",
    "primary_endpoint": "Objective Response Rate (ORR)",
    "target_enrollment": 120,
    "enrollment_timeline": "18 months",
    "treatment": "Novel KRAS G12C inhibitor + pembrolizumab",
    "prior_therapy_requirement": "At least one prior systemic therapy",
    "biomarker_requirement": "KRAS G12C mutation confirmed",
    "known_drug_characteristics": {
        "hepatotoxicity_risk": "moderate",
        "cardiac_risk": "low (QT prolongation monitoring needed)",
        "drug_interactions": ["strong CYP3A4 inhibitors/inducers"],
        "food_interaction": "take with food"
    },
    "study_assessments": [
        "CT scans every 8 weeks",
        "Blood draws every 2 weeks",
        "ECG at baseline and week 4",
        "Optional tumor biopsy"
    ],
    "sites": "15 academic medical centers in US"
}


class EligibilityCriteriaGenerator:
    """Generates clinical trial eligibility criteria."""

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def generate(self, study_context: Dict, feedback: Optional[Dict] = None,
                 previous_criteria: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate or revise eligibility criteria.

        Returns: {inclusion_criteria: [], exclusion_criteria: [], notes: str}
        """
        # TODO: Step 1 - Check if revision (previous_criteria + feedback exist) vs initial

        # TODO: Step 2 - For REVISION: system prompt for expert revising based on
        #       multi-stakeholder feedback (GCP, feasibility, fairness)
        #       User prompt: study context, previous criteria, feedback to address

        # TODO: Step 3 - For INITIAL: system prompt for expert creating criteria that are:
        #       scientifically rigorous, regulatory compliant, feasible, fair
        #       INCLUSION: Age, Diagnosis, Stage, Prior therapy, Biomarkers, ECOG, Labs
        #       EXCLUSION: Other cancers, CNS mets, Comorbidities, Medications, Pregnancy

        # TODO: Step 4 - Call LLM with response_format={"type": "json_object"}

        # TODO: Step 5 - Parse and return dict
        pass


class GCPComplianceEvaluator:
    """Evaluates regulatory compliance. PROVIDED as example."""

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def evaluate(self, criteria: Dict, study_context: Dict) -> Dict[str, Any]:
        """Check criteria against GCP/regulatory requirements."""
        system_prompt = """You are a regulatory affairs expert. Return JSON with:
- approved: boolean (score >= 80)
- score: float 0-100
- issues: list of {criterion, issue, severity, suggestion}
- checklist: {informed_consent_requirement, age_specification, measurable_criteria, safety_protections}
- feedback_summary: string"""

        user_prompt = f"""Evaluate for GCP/regulatory compliance:

STUDY: {json.dumps(study_context, indent=2)}
INCLUSION: {json.dumps(criteria.get('inclusion_criteria', []), indent=2)}
EXCLUSION: {json.dumps(criteria.get('exclusion_criteria', []), indent=2)}"""

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"}, temperature=0.3
        )
        result = json.loads(response.choices[0].message.content)
        score = float(result.get("score", 70))
        return {"approved": result.get("approved", score >= 80), "score": score,
                "issues": result.get("issues", []), "checklist": result.get("checklist", {}),
                "feedback_summary": result.get("feedback_summary", "")}


class EnrollmentFeasibilityEvaluator:
    """Evaluates practical enrollment feasibility."""

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def evaluate(self, criteria: Dict, study_context: Dict) -> Dict[str, Any]:
        """
        Returns: {approved, score, estimated_screen_fail_rate, restrictive_criteria: [{
        criterion, estimated_impact, patient_pool_reduction, suggestion}],
        enrollment_projection, feedback_summary}
        """
        # TODO: Step 1 - System prompt for clinical operations expert
        #       Consider: each criterion reduces pool, biomarkers limit population,
        #       strict labs increase screen failures

        # TODO: Step 2 - User prompt with context, criteria, target enrollment, timeline

        # TODO: Step 3 - Call LLM with response_format={"type": "json_object"}

        # TODO: Step 4 - Parse and return dict
        pass


class DemographicFairnessEvaluator:
    """Evaluates demographic inclusivity."""

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def evaluate(self, criteria: Dict, study_context: Dict) -> Dict[str, Any]:
        """
        Returns: {approved, score, fairness_concerns: [{criterion, affected_group,
        concern, impact_severity, suggestion}], representation_analysis,
        fda_diversity_guidance_alignment, feedback_summary}
        """
        # TODO: Step 1 - System prompt for diversity expert (per FDA guidance):
        #       Upper age limits justified? Excludes minorities? Lab value variation?
        #       Access requirements? Pregnancy exclusions appropriate?

        # TODO: Step 2 - User prompt with context and criteria

        # TODO: Step 3 - Call LLM with response_format={"type": "json_object"}

        # TODO: Step 4 - Parse and return dict
        pass


class MultiAspectEvaluator:
    """Coordinates all three evaluators."""

    GCP_WEIGHT, FEASIBILITY_WEIGHT, FAIRNESS_WEIGHT = 0.40, 0.35, 0.25

    def __init__(self, llm_client: OpenAI):
        self.gcp_evaluator = GCPComplianceEvaluator(llm_client)
        self.feasibility_evaluator = EnrollmentFeasibilityEvaluator(llm_client)
        self.fairness_evaluator = DemographicFairnessEvaluator(llm_client)

    def evaluate(self, criteria: Dict, study_context: Dict) -> Dict[str, Any]:
        """Run all evaluators and aggregate results."""
        gcp = self.gcp_evaluator.evaluate(criteria, study_context)
        feas = self.feasibility_evaluator.evaluate(criteria, study_context)
        fair = self.fairness_evaluator.evaluate(criteria, study_context)

        # Handle None from unimplemented
        if gcp is None:
            gcp = {"approved": False, "score": 0, "issues": [], "checklist": {}, "feedback_summary": "NOT IMPLEMENTED"}
        if feas is None:
            feas = {"approved": False, "score": 0, "estimated_screen_fail_rate": 0,
                   "restrictive_criteria": [], "enrollment_projection": {}, "feedback_summary": "NOT IMPLEMENTED"}
        if fair is None:
            fair = {"approved": False, "score": 0, "fairness_concerns": [],
                   "representation_analysis": {}, "fda_diversity_guidance_alignment": "", "feedback_summary": "NOT IMPLEMENTED"}

        overall = gcp["score"]*self.GCP_WEIGHT + feas["score"]*self.FEASIBILITY_WEIGHT + fair["score"]*self.FAIRNESS_WEIGHT
        all_approved = gcp["approved"] and feas["approved"] and fair["approved"]

        priority_issues = []
        for issue in gcp.get("issues", []):
            if issue.get("severity") in ["Critical", "Major"]:
                priority_issues.append(f"GCP: {issue.get('issue', '')}")
        for item in feas.get("restrictive_criteria", []):
            if item.get("estimated_impact") in ["High", "Medium"]:
                priority_issues.append(f"Feasibility: {item.get('criterion', '')} - {item.get('suggestion', '')}")
        for c in fair.get("fairness_concerns", []):
            if c.get("impact_severity") in ["High", "Medium"]:
                priority_issues.append(f"Fairness: {c.get('concern', '')}")

        return {"all_approved": all_approved, "overall_score": round(overall, 1),
                "gcp_evaluation": gcp, "feasibility_evaluation": feas, "fairness_evaluation": fair,
                "combined_feedback": f"GCP: {gcp['feedback_summary']}\nFeasibility: {feas['feedback_summary']}\nFairness: {fair['feedback_summary']}",
                "priority_issues": priority_issues[:5]}


class EligibilityCriteriaWorkflow:
    """Orchestrates the multi-evaluator optimization loop."""

    MAX_ITERATIONS = 5

    def __init__(self, llm_client: OpenAI):
        self.generator = EligibilityCriteriaGenerator(llm_client)
        self.evaluator = MultiAspectEvaluator(llm_client)

    def run(self, study_context: Dict) -> Dict[str, Any]:
        """Run the optimization loop."""
        print(f"\n{'='*60}\nELIGIBILITY CRITERIA OPTIMIZER: {study_context['indication']}\n{'='*60}")

        current_criteria, feedback, history, evaluation = None, None, [], None

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            print(f"\n[Iteration {iteration}]")

            current_criteria = self.generator.generate(study_context, feedback, current_criteria)
            if current_criteria is None:
                current_criteria = {"inclusion_criteria": ["NOT IMPLEMENTED"],
                                   "exclusion_criteria": ["NOT IMPLEMENTED"], "notes": "TODO"}

            evaluation = self.evaluator.evaluate(current_criteria, study_context)
            history.append({"iteration": iteration, "overall_score": evaluation["overall_score"],
                           "gcp_score": evaluation["gcp_evaluation"]["score"],
                           "feasibility_score": evaluation["feasibility_evaluation"]["score"],
                           "fairness_score": evaluation["fairness_evaluation"]["score"],
                           "approved": evaluation["all_approved"]})

            print(f"  GCP: {evaluation['gcp_evaluation']['score']:.0f} | "
                  f"Feas: {evaluation['feasibility_evaluation']['score']:.0f} | "
                  f"Fair: {evaluation['fairness_evaluation']['score']:.0f} | "
                  f"Overall: {evaluation['overall_score']:.0f}")

            if evaluation["all_approved"]:
                print(f"  ALL APPROVED after {iteration} iteration(s)")
                break
            else:
                feedback = {"gcp": evaluation["gcp_evaluation"]["feedback_summary"],
                           "feasibility": evaluation["feasibility_evaluation"]["feedback_summary"],
                           "fairness": evaluation["fairness_evaluation"]["feedback_summary"]}

        return {"final_criteria": current_criteria, "approved": evaluation["all_approved"] if evaluation else False,
                "iterations": iteration, "history": history, "final_evaluation": evaluation}


def display_results(result: Dict) -> None:
    """Display workflow results."""
    print(f"\n{'='*60}\nFINAL CRITERIA\n{'='*60}")
    for i, c in enumerate(result["final_criteria"].get("inclusion_criteria", []), 1):
        print(f"  IN{i}. {c}")
    for i, c in enumerate(result["final_criteria"].get("exclusion_criteria", []), 1):
        print(f"  EX{i}. {c}")

    print(f"\n{'='*60}\nHISTORY\n{'='*60}")
    for h in result["history"]:
        status = "APPROVED" if h["approved"] else "..."
        print(f"  Iter {h['iteration']}: {h['overall_score']:.0f} ({status})")


def main():
    """Main execution function."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

    print("Module 8: Eligibility Criteria Refiner")
    print("Evaluator-Optimizer Loop: GCP, Feasibility, Fairness\n")

    workflow = EligibilityCriteriaWorkflow(client)
    result = workflow.run(STUDY_CONTEXT)
    display_results(result)


if __name__ == "__main__":
    main()
