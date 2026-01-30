"""
Module 8 Exercise: Trial Eligibility Criteria Refiner

An iterative evaluator-optimizer workflow for clinical trial eligibility criteria
that refines based on evaluation across three dimensions:
1. GCP (Good Clinical Practice) Compliance
2. Enrollment Feasibility
3. Demographic Fairness
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


# Study context for the exercise
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
    """
    Generates clinical trial eligibility criteria.
    """

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def generate(
        self,
        study_context: Dict[str, Any],
        feedback: Optional[Dict[str, str]] = None,
        previous_criteria: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate or revise eligibility criteria.

        Args:
            study_context: Study design information
            feedback: Multi-evaluator feedback (if revising)
            previous_criteria: Previous version (if revising)

        Returns:
            Dictionary with inclusion_criteria, exclusion_criteria, and notes
        """

        if previous_criteria and feedback:
            if previous_criteria and feedback:
                lowest_dim = "fairness" if feedback.get("fairness_score", 100) <= feedback.get("feasibility_score", 100) else "feasibility"
                
                system_prompt = f"""You are revising clinical trial eligibility criteria based on multi-stakeholder feedback.

            PRIORITY: The {lowest_dim.upper()} score is lowest and MUST improve this iteration.

            {"FAIRNESS PRIORITY: You MUST make at least 2 changes that improve demographic inclusivity (age limits, ECOG status, lab ranges that vary by race, socioeconomic barriers)." if lowest_dim == "fairness" else ""}

            Execution rules:
            1. For each item under "RESTRICTIVE CRITERIA TO FIX" and "FAIRNESS CONCERNS TO FIX", you MUST modify the specific criterion text.
            2. Do not only add notes - make direct edits to inclusion/exclusion lists.
            3. If fairness score < 80, you MUST: remove upper age limits OR expand ECOG to 0-2 OR add "controlled/stable" language to comorbidity exclusions.

            Return JSON with: inclusion_criteria (list), exclusion_criteria (list), notes (string explaining changes made)."""


            user_prompt = f"""Please revise the following eligibility criteria based on the feedback provided.

STUDY CONTEXT:
{json.dumps(study_context, indent=2)}

PREVIOUS CRITERIA:
Inclusion:
{json.dumps(previous_criteria.get('inclusion_criteria', []), indent=2)}

Exclusion:
{json.dumps(previous_criteria.get('exclusion_criteria', []), indent=2)}

PREVIOUS SCORES (must improve these to >= 80):
- Feasibility: {feedback.get('feasibility_score', 'N/A')}/100
- Fairness: {feedback.get('fairness_score', 'N/A')}/100

FEEDBACK TO ADDRESS:

GCP Compliance Feedback:
{feedback.get('gcp', 'No specific feedback')}

Enrollment Feasibility Feedback:
{feedback.get('feasibility', 'No specific feedback')}

Demographic Fairness Feedback:
{feedback.get('fairness', 'No specific feedback')}

Please provide revised criteria that address the feedback while maintaining scientific rigor.
Return as JSON with inclusion_criteria (list), exclusion_criteria (list), and notes (string)."""

        else:
            system_prompt = """You are an expert in clinical trial design,
creating eligibility criteria that are:
- Scientifically rigorous
- Regulatory compliant (ICH-GCP, FDA guidance)
- Practically feasible for enrollment
- Fair across demographic groups

Write criteria that are specific, measurable, and unambiguous.

Return your response as a JSON object with these keys:
- inclusion_criteria: List of strings (each criterion as a string)
- exclusion_criteria: List of strings (each criterion as a string)
- notes: String explaining key decisions

Standard criteria categories to consider:

INCLUSION:
- Age
- Diagnosis/histology
- Disease stage
- Prior therapy
- Biomarker status
- Performance status
- Organ function (labs)
- Life expectancy
- Consent capability

EXCLUSION:
- Other malignancies
- CNS metastases (often excluded)
- Specific comorbidities
- Concomitant medications (drug interactions)
- Prior specific therapies
- Pregnancy/lactation
- Organ dysfunction
- Active infections
- Recent surgery"""

            user_prompt = f"""Please generate eligibility criteria for the following clinical trial:

STUDY CONTEXT:
{json.dumps(study_context, indent=2)}

Generate comprehensive inclusion and exclusion criteria appropriate for this study.
Return as JSON with inclusion_criteria (list), exclusion_criteria (list), and notes (string)."""

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        result = json.loads(response.choices[0].message.content)

        # Ensure proper structure
        return {
            "inclusion_criteria": result.get("inclusion_criteria", []),
            "exclusion_criteria": result.get("exclusion_criteria", []),
            "notes": result.get("notes", "")
        }


class GCPComplianceEvaluator:
    """
    Evaluates criteria for regulatory compliance.
    """

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def evaluate(self, criteria: Dict, study_context: Dict) -> Dict[str, Any]:
        """
        Check criteria against GCP/regulatory requirements.
        """
        system_prompt = """You are a regulatory affairs expert evaluating clinical trial eligibility criteria
for GCP (Good Clinical Practice) and FDA compliance.

Evaluate the criteria and return a JSON object with:
- approved: boolean (true if criteria meet regulatory standards, score >= 80)
- score: float 0-100 (regulatory compliance score)
- issues: list of objects with keys: criterion, issue, severity ("Critical"/"Major"/"Minor"), suggestion
- checklist: object with boolean values for:
  - informed_consent_requirement
  - age_specification
  - measurable_criteria
  - safety_protections
  - withdrawal_criteria_implied
- feedback_summary: string summarizing key regulatory concerns to address

Key checks:
- Age must be specified (FDA requirement)
- Criteria must be objectively measurable
- Safety-related exclusions adequate for known drug risks
- Vulnerable populations protected
- Criteria consistent with drug label/IB"""

        user_prompt = f"""Evaluate these eligibility criteria for GCP/regulatory compliance:

STUDY CONTEXT:
{json.dumps(study_context, indent=2)}

INCLUSION CRITERIA:
{json.dumps(criteria.get('inclusion_criteria', []), indent=2)}

EXCLUSION CRITERIA:
{json.dumps(criteria.get('exclusion_criteria', []), indent=2)}

Evaluate for regulatory compliance and return JSON with approved, score, issues, checklist, and feedback_summary."""

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )

        result = json.loads(response.choices[0].message.content)

        # Ensure proper structure and types
        score = float(result.get("score", 70))
        return {
            "approved": result.get("approved", score >= 80),
            "score": score,
            "issues": result.get("issues", []),
            "checklist": result.get("checklist", {}),
            "feedback_summary": result.get("feedback_summary", "")
        }


class EnrollmentFeasibilityEvaluator:
    """
    Evaluates criteria for practical enrollment.
    """

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def evaluate(self, criteria: Dict, study_context: Dict) -> Dict[str, Any]:
        """
        Assess whether criteria allow feasible enrollment.
        """
        system_prompt = """You are a clinical operations expert evaluating eligibility criteria
for enrollment feasibility.

Evaluate the criteria and return a JSON object with:
- approved: boolean (true if criteria allow feasible enrollment, score >= 80)
- score: float 0-100 (feasibility score)
- estimated_screen_fail_rate: float 0-1 (estimated proportion of screened patients who will fail)
- restrictive_criteria: list of objects with keys: criterion, estimated_impact ("High"/"Medium"/"Low"),
  patient_pool_reduction (e.g., "Reduces pool by ~40%"), suggestion
  IMPORTANT: Do NOT include the study_context biomarker_requirement as a restrictive criterion. 
  Only include criteria that are modifiable (labs, comorbidities, CNS mets, washouts, prior-therapy 
- enrollment_projection: object with:
  - estimated_eligible_patients: int
  - projected_enrollment_rate: string
  - timeline_feasibility: string ("On track"/"At risk"/"Unlikely")
- feedback_summary: string summarizing key feasibility concerns to address

CRITICAL SCORING RULES:
- If criteria have been RELAXED (wider lab ranges, shorter washouts, allowing stable CNS mets), 
  you MUST increase your score proportionally.
- You must cite SPECIFIC criterion text that justifies your score.
- Score 75 = overly restrictive. Score 80+ = reasonable for enrollment target.
- Relaxing ANY high-impact criterion should add 3-8 points.

IMPORTANT:
If the STUDY CONTEXT includes a biomarker_requirement, treat it as a fixed protocol constraint.
Do NOT recommend broadening/removing/changing the biomarker requirement.
Instead, improve feasibility by adjusting modifiable criteria (labs, comorbidities, CNS metastases,
prior-therapy exclusions, testing logistics/windows, visit burden).

Feasibility considerations:
- Each additional criterion reduces eligible pool
- Biomarker requirements limit population
- Strict lab requirements increase screen failures
- Geographic/access considerations
- Target enrollment: {target} patients in {timeline}

SCORING DISCIPLINE:
- Your score MUST be justified by citing 2–4 specific criteria verbatim (or close paraphrase).
- If the criteria become less restrictive in modifiable areas (labs/windows, CNS mets, comorbidities),
  your score MUST increase accordingly (do not repeat the same score unless nothing meaningful changed).
- feedback_summary MUST include "Top 3 changes" in numbered form, each referencing the exact criterion to change."""

        user_prompt = f"""Evaluate these eligibility criteria for enrollment feasibility:

STUDY CONTEXT:
{json.dumps(study_context, indent=2)}

INCLUSION CRITERIA:
{json.dumps(criteria.get('inclusion_criteria', []), indent=2)}

EXCLUSION CRITERIA:
{json.dumps(criteria.get('exclusion_criteria', []), indent=2)}

Target: {study_context.get('target_enrollment', 120)} patients in {study_context.get('enrollment_timeline', '18 months')}
Sites: {study_context.get('sites', 'Unknown')}

Evaluate for enrollment feasibility and return JSON with approved, score, estimated_screen_fail_rate,
restrictive_criteria, enrollment_projection, and feedback_summary."""

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )

        result = json.loads(response.choices[0].message.content)

        # Ensure proper structure and types
        score = float(result.get("score", 60))
        return {
            "approved": result.get("approved", score >= 80),
            "score": score,
            "estimated_screen_fail_rate": float(result.get("estimated_screen_fail_rate", 0.4)),
            "restrictive_criteria": result.get("restrictive_criteria", []),
            "enrollment_projection": result.get("enrollment_projection", {}),
            "feedback_summary": result.get("feedback_summary", "")
        }


class DemographicFairnessEvaluator:
    """
    Evaluates criteria for demographic inclusivity.
    """

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def evaluate(self, criteria: Dict, study_context: Dict) -> Dict[str, Any]:
        """
        Assess whether criteria promote diverse enrollment.
        """
        system_prompt = """You are a clinical trial diversity expert evaluating eligibility criteria
for demographic fairness and inclusivity.

CRITICAL SCORING RULES:
- Score 80 = baseline/default
- Each criterion that disproportionately excludes a group = -3 to -5 points
- Each improvement (e.g., removing upper age limit, adding "stable/controlled" language, 
relaxing socioeconomic barriers) = +3 to +5 points
- If criteria IMPROVED from previous version, score MUST increase.

Evaluate the criteria and return a JSON object with:
- approved: boolean (true if score >= 80)
- score: float 0-100 (START at 80, then adjust based on specific criteria)
- fairness_concerns: list of objects with keys: criterion (EXACT TEXT from criteria), 
affected_group, concern, impact_severity ("High"/"Medium"/"Low"), suggestion
- representation_analysis: object analyzing age, sex/gender, race/ethnicity, socioeconomic access
- fda_diversity_guidance_alignment: string
- feedback_summary: string with "TOP 3 CHANGES NEEDED:" numbered list

COMMON FAIRNESS ISSUES TO CHECK:
1. Upper age limits (exclude elderly) - remove or justify
2. ECOG 0-1 only (excludes minorities with less healthcare access) - consider ECOG 0-2
3. Strict lab values (normal ranges vary by race) - use inclusive ranges
4. Geographic/travel requirements (exclude rural patients)
5. Caregiver/transportation requirements (socioeconomic barrier)
6. English language requirements (exclude non-English speakers)
7. Broad comorbidity exclusions (elderly, minorities disproportionately affected)

You MUST cite the EXACT criterion text in fairness_concerns."""

        user_prompt = f"""Evaluate these eligibility criteria for demographic fairness:

STUDY CONTEXT:
{json.dumps(study_context, indent=2)}

INCLUSION CRITERIA:
{json.dumps(criteria.get('inclusion_criteria', []), indent=2)}

EXCLUSION CRITERIA:
{json.dumps(criteria.get('exclusion_criteria', []), indent=2)}

Evaluate for demographic fairness and return JSON with approved, score, fairness_concerns,
representation_analysis, fda_diversity_guidance_alignment, and feedback_summary."""

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )

        result = json.loads(response.choices[0].message.content)

        # Ensure proper structure and types
        score = float(result.get("score", 65))
        return {
            "approved": result.get("approved", score >= 80),
            "score": score,
            "fairness_concerns": result.get("fairness_concerns", []),
            "representation_analysis": result.get("representation_analysis", {}),
            "fda_diversity_guidance_alignment": result.get("fda_diversity_guidance_alignment", ""),
            "feedback_summary": result.get("feedback_summary", "")
        }


class MultiAspectEvaluator:
    """
    Coordinates all three evaluators.
    """

    # Weighting for overall score
    GCP_WEIGHT = 0.40
    FEASIBILITY_WEIGHT = 0.35
    FAIRNESS_WEIGHT = 0.25

    def __init__(self, llm_client: OpenAI):
        self.gcp_evaluator = GCPComplianceEvaluator(llm_client)
        self.feasibility_evaluator = EnrollmentFeasibilityEvaluator(llm_client)
        self.fairness_evaluator = DemographicFairnessEvaluator(llm_client)

    def evaluate(self, criteria: Dict, study_context: Dict) -> Dict[str, Any]:
        """
        Run all evaluators and aggregate results.
        """
        # Run all three evaluations
        gcp_eval = self.gcp_evaluator.evaluate(criteria, study_context)
        feasibility_eval = self.feasibility_evaluator.evaluate(criteria, study_context)
        fairness_eval = self.fairness_evaluator.evaluate(criteria, study_context)

        # Calculate weighted overall score
        overall_score = (
            gcp_eval["score"] * self.GCP_WEIGHT +
            feasibility_eval["score"] * self.FEASIBILITY_WEIGHT +
            fairness_eval["score"] * self.FAIRNESS_WEIGHT
        )

        # All must approve for overall approval
        all_approved = (
            gcp_eval["approved"] and
            feasibility_eval["approved"] and
            fairness_eval["approved"]
        )

        # Collect priority issues
        priority_issues = []

        # Add GCP issues
        for issue in gcp_eval.get("issues", []):
            if issue.get("severity") in ["Critical", "Major"]:
                priority_issues.append(f"GCP: {issue.get('issue', 'Unknown issue')}")

        # Add feasibility issues
        for item in feasibility_eval.get("restrictive_criteria", []):
            if item.get("estimated_impact") in ["High", "Medium"]:
                priority_issues.append(f"Feasibility: {item.get('criterion', 'Unknown')} - {item.get('suggestion', '')}")

        # Add fairness issues
        for concern in fairness_eval.get("fairness_concerns", []):
            if concern.get("impact_severity") in ["High", "Medium"]:
                priority_issues.append(f"Fairness: {concern.get('concern', 'Unknown concern')}")

        # Create combined feedback
        combined_feedback = f"""GCP COMPLIANCE:
{gcp_eval.get('feedback_summary', 'No feedback')}

ENROLLMENT FEASIBILITY:
{feasibility_eval.get('feedback_summary', 'No feedback')}

DEMOGRAPHIC FAIRNESS:
{fairness_eval.get('feedback_summary', 'No feedback')}"""

        return {
            "all_approved": all_approved,
            "overall_score": round(overall_score, 1),
            "gcp_evaluation": gcp_eval,
            "feasibility_evaluation": feasibility_eval,
            "fairness_evaluation": fairness_eval,
            "combined_feedback": combined_feedback,
            "priority_issues": priority_issues[:5]  # Top 5 issues
        }


class EligibilityCriteriaWorkflow:
    """
    Orchestrates the multi-evaluator optimization loop.
    """

    MAX_ITERATIONS = 5

    def __init__(self, llm_client: OpenAI):
        self.generator = EligibilityCriteriaGenerator(llm_client)
        self.evaluator = MultiAspectEvaluator(llm_client)

    def run(self, study_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the optimization loop.
        """
        print(f"\n{'='*70}")
        print("ELIGIBILITY CRITERIA OPTIMIZER")
        print(f"Study: {study_context['indication']} ({study_context['study_phase']})")
        print(f"{'='*70}")

        current_criteria = None
        feedback = None
        history = []
        evaluation = None
        iteration = 0

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            print(f"\n[Iteration {iteration}/{self.MAX_ITERATIONS}]")
            print("-" * 40)

            # Generate
            current_criteria = self.generator.generate(
                study_context,
                feedback=feedback,
                previous_criteria=current_criteria
            )

            print(f"Generated {len(current_criteria['inclusion_criteria'])} inclusion "
                  f"and {len(current_criteria['exclusion_criteria'])} exclusion criteria")

            # Evaluate with all three evaluators
            evaluation = self.evaluator.evaluate(current_criteria, study_context)

            # Log
            history.append({
                "iteration": iteration,
                "overall_score": evaluation["overall_score"],
                "gcp_score": evaluation["gcp_evaluation"]["score"],
                "feasibility_score": evaluation["feasibility_evaluation"]["score"],
                "fairness_score": evaluation["fairness_evaluation"]["score"],
                "approved": evaluation["all_approved"]
            })

            # Display
            print(f"\nEvaluation Scores:")
            print(f"  GCP Compliance:   {evaluation['gcp_evaluation']['score']:>5.1f}/100 "
                  f"{'✓' if evaluation['gcp_evaluation']['approved'] else '✗'}")
            print(f"  Feasibility:      {evaluation['feasibility_evaluation']['score']:>5.1f}/100 "
                  f"{'✓' if evaluation['feasibility_evaluation']['approved'] else '✗'}")
            print(f"  Fairness:         {evaluation['fairness_evaluation']['score']:>5.1f}/100 "
                  f"{'✓' if evaluation['fairness_evaluation']['approved'] else '✗'}")
            print(f"  Overall:          {evaluation['overall_score']:>5.1f}/100")

            if evaluation["all_approved"]:
                print(f"\n✓ ALL EVALUATORS APPROVED after {iteration} iteration(s)")
                break
            else:
                feas = evaluation["feasibility_evaluation"]
                fair = evaluation["fairness_evaluation"]

                feedback = {
                    "gcp": evaluation["gcp_evaluation"]["feedback_summary"],
                    "feasibility_score": feas["score"],
                    "fairness_score": fair["score"],
                    "feasibility": (
                        feas.get("feedback_summary", "") +
                        "\n\nRESTRICTIVE CRITERIA TO FIX (edit these directly):\n" +
                        "\n".join(
                            f"- {i.get('criterion','')} | impact={i.get('estimated_impact','')} | suggestion={i.get('suggestion','')}"
                            for i in feas.get("restrictive_criteria", [])[:5]
                        )
                    ),
                    "fairness": (
                        fair.get("feedback_summary", "") +
                        "\n\nFAIRNESS CONCERNS TO FIX (you MUST edit these specific criteria):\n" +
                        "\n".join(
                            f"- CRITERION: \"{c.get('criterion','')}\" | AFFECTED GROUP: {c.get('affected_group','')} | REQUIRED CHANGE: {c.get('suggestion','')}"
                            for c in fair.get("fairness_concerns", [])[:5]
                        )
                    )
                }
                print(f"\nPriority issues to address:")
                for issue in evaluation["priority_issues"][:3]:
                    print(f"  • {issue}")

        return {
            "final_criteria": current_criteria,
            "approved": evaluation["all_approved"] if evaluation else False,
            "iterations": iteration,
            "history": history,
            "final_evaluation": evaluation
        }


def print_iteration_history(history: List[Dict]) -> None:
    """Print the iteration history table."""
    print(f"\n{'='*70}")
    print("ITERATION HISTORY")
    print(f"{'='*70}\n")

    print("| Iter | Overall | GCP    | Feasibility | Fairness | Status    |")
    print("|------|---------|--------|-------------|----------|-----------|")

    for h in history:
        gcp_mark = "✓" if h["gcp_score"] >= 80 else "✗"
        feas_mark = "✓" if h["feasibility_score"] >= 80 else "✗"
        fair_mark = "✓" if h["fairness_score"] >= 80 else "✗"
        status = "APPROVED" if h["approved"] else "Revising"

        print(f"| {h['iteration']:<4} | {h['overall_score']:<7.1f} | {h['gcp_score']:<4.1f} {gcp_mark} | "
              f"{h['feasibility_score']:<9.1f} {feas_mark} | {h['fairness_score']:<6.1f} {fair_mark} | {status:<9} |")


def print_final_criteria(criteria: Dict) -> None:
    """Print the final eligibility criteria."""
    print(f"\n{'='*70}")
    print("FINAL ELIGIBILITY CRITERIA")
    print(f"{'='*70}")

    print("\nINCLUSION CRITERIA:")
    for i, criterion in enumerate(criteria.get("inclusion_criteria", []), 1):
        print(f"{i}. {criterion}")

    print("\nEXCLUSION CRITERIA:")
    for i, criterion in enumerate(criteria.get("exclusion_criteria", []), 1):
        print(f"{i}. {criterion}")


def print_evaluation_summary(evaluation: Dict) -> None:
    """Print the detailed evaluation summary."""
    print(f"\n{'='*70}")
    print("DETAILED EVALUATION SUMMARY")
    print(f"{'='*70}")

    # GCP Compliance
    gcp = evaluation["gcp_evaluation"]
    gcp_mark = "✓" if gcp["approved"] else "✗"
    print(f"\nGCP COMPLIANCE ({gcp['score']:.0f}/100) {gcp_mark}")

    checklist = gcp.get("checklist", {})
    for key, value in checklist.items():
        mark = "✓" if value else "✗"
        formatted_key = key.replace("_", " ").title()
        print(f"  {mark} {formatted_key}")

    # Enrollment Feasibility
    feas = evaluation["feasibility_evaluation"]
    feas_mark = "✓" if feas["approved"] else "✗"
    print(f"\nENROLLMENT FEASIBILITY ({feas['score']:.0f}/100) {feas_mark}")
    print(f"  Estimated Screen Fail Rate: {feas.get('estimated_screen_fail_rate', 0)*100:.0f}%")

    projection = feas.get("enrollment_projection", {})
    if projection:
        print(f"  Projected Enrollment: {projection.get('projected_enrollment_rate', 'Unknown')}")
        print(f"  Timeline Assessment: {projection.get('timeline_feasibility', 'Unknown')}")

    if feas.get("restrictive_criteria"):
        print("\n  Improvements needed:")
        for item in feas["restrictive_criteria"][:3]:
            print(f"    - {item.get('criterion', 'Unknown')}: {item.get('suggestion', '')}")

    # Demographic Fairness
    fair = evaluation["fairness_evaluation"]
    fair_mark = "✓" if fair["approved"] else "✗"
    print(f"\nDEMOGRAPHIC FAIRNESS ({fair['score']:.0f}/100) {fair_mark}")

    rep = fair.get("representation_analysis", {})
    if rep:
        for key, value in rep.items():
            formatted_key = key.replace("_", " ").title()
            print(f"  {formatted_key}: {value}")

    print(f"\n  FDA Diversity Guidance Alignment: {fair.get('fda_diversity_guidance_alignment', 'Unknown')}")


def main():
    """Main execution function."""
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

    # Create and run the workflow
    workflow = EligibilityCriteriaWorkflow(client)
    result = workflow.run(STUDY_CONTEXT)

    # Print iteration history
    print_iteration_history(result["history"])

    # Print final criteria
    print_final_criteria(result["final_criteria"])

    # Print evaluation summary
    print_evaluation_summary(result["final_evaluation"])

    print(f"\n{'='*70}")

    return result


if __name__ == "__main__":
    main()
