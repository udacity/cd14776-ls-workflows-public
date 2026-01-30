"""
Module 8 Demo: Evaluator-Optimizer Workflow

Patient Consent Form Generator with Ethics Auditor
An iterative workflow where a generator creates consent forms and an
ethics auditor evaluates them until IRB standards are met.

Generator Agent → Ethics Auditor Agent → [Loop until approved or max iterations]
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent / ".env")


# Study Information Input
STUDY_INFO = {
    "title": "A Phase 2 Study of XYZ-101 in Patients with Advanced Solid Tumors",
    "protocol_number": "XYZ-101-002",
    "sponsor": "Acme Therapeutics",
    "principal_investigator": "Dr. Jane Smith, MD",
    "institution": "University Medical Center",
    "study_type": "Interventional, open-label, single-arm",
    "study_drug": "XYZ-101",
    "drug_description": "An oral small molecule inhibitor targeting the ABC pathway",
    "target_population": "Adults with advanced solid tumors who have failed standard therapy",
    "sample_size": 60,
    "study_duration": "24 months",
    "visit_schedule": "Screening, Day 1, then every 2 weeks for 3 months, then monthly",
    "procedures": [
        "Physical examination",
        "Blood draws (approximately 4 tablespoons per visit)",
        "CT scans every 8 weeks",
        "Optional tumor biopsy at baseline and week 8",
        "Quality of life questionnaires"
    ],
    "known_risks": [
        "Nausea and vomiting (common, ~40%)",
        "Fatigue (common, ~35%)",
        "Liver enzyme elevation (common, ~25%, usually reversible)",
        "Rash (uncommon, ~10%)",
        "QT prolongation (rare, <5%, requires monitoring)",
        "Unknown risks from a new investigational drug"
    ],
    "potential_benefits": [
        "Possible tumor shrinkage or disease stabilization",
        "No guarantee of direct benefit",
        "Contribution to medical knowledge"
    ],
    "alternatives": [
        "Standard chemotherapy",
        "Other clinical trials",
        "Supportive/palliative care"
    ],
    "compensation": "Parking reimbursement, no other payment",
    "costs": "Study drug and study-related tests provided at no cost. Standard care costs may be billed to insurance."
}


@dataclass
class ConsentEvaluationCriteria:
    """
    IRB evaluation criteria for informed consent documents.
    Based on 45 CFR 46.116 and ICH-GCP E6(R2).
    """

    REQUIRED_ELEMENTS: List[str] = field(default_factory=lambda: [
        "study_purpose",           # Clear statement of research purpose
        "procedures",              # Description of procedures
        "duration",                # Expected duration of participation
        "risks",                   # Foreseeable risks/discomforts
        "benefits",                # Potential benefits
        "alternatives",            # Alternative treatments
        "confidentiality",         # How data will be protected
        "compensation",            # Compensation for injury
        "contact_info",            # Who to contact
        "voluntary_participation", # Participation is voluntary
        "withdrawal_rights"        # Right to withdraw without penalty
    ])

    READABILITY_TARGET: float = 8.0  # Target grade level (8th grade or lower)

    QUALITY_CHECKS: List[str] = field(default_factory=lambda: [
        "no_coercive_language",    # No language that could unduly influence
        "balanced_risk_benefit",   # Risks and benefits equally prominent
        "clear_experimental_nature", # Clear that this is research
        "no_exculpatory_language", # No waiver of legal rights
        "appropriate_length"       # Not overwhelming (target: 5-10 pages)
    ])


class ConsentFormGenerator:
    """
    Generates patient consent form drafts.
    """

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def generate(
        self,
        study_info: Dict[str, Any],
        feedback: Optional[str] = None,
        previous_draft: Optional[str] = None
    ) -> str:
        """
        Generate or revise a consent form.

        Args:
            study_info: Study details
            feedback: Evaluator feedback from previous iteration (if any)
            previous_draft: Previous version to revise (if any)

        Returns:
            Markdown-formatted consent form text
        """

        if previous_draft and feedback:
            # Revision mode
            system_prompt = """You are revising a patient consent form based on
ethics committee feedback. Address each concern while maintaining
readability and regulatory compliance.

Focus on the specific feedback provided. Do not introduce new issues.

Writing guidelines:
- Use 8th grade reading level or lower
- Use short sentences (15-20 words average)
- Avoid medical jargon; explain necessary terms in simple words
- Use "you" and "your" (second person)
- Use bullet points for lists
- Be honest and balanced about risks
- NEVER use exculpatory language (phrases that waive legal rights)
- NEVER use coercive language (phrases that pressure participation)"""

            user_prompt = f"""Please revise this consent form based on the feedback:

EVALUATOR FEEDBACK:
{feedback}

PREVIOUS DRAFT:
{previous_draft}

STUDY INFORMATION:
{json.dumps(study_info, indent=2)}

Generate the complete revised consent form addressing ALL feedback items.
Make sure to include all required sections:
1. STUDY TITLE
2. INTRODUCTION
3. PURPOSE OF THE STUDY
4. STUDY PROCEDURES
5. RISKS AND DISCOMFORTS
6. POTENTIAL BENEFITS
7. ALTERNATIVES
8. CONFIDENTIALITY
9. COSTS AND COMPENSATION
10. VOLUNTARY PARTICIPATION
11. WITHDRAWAL FROM THE STUDY
12. QUESTIONS AND CONTACT INFORMATION - MUST include a specific 24-hour emergency phone number (use 555-123-4567)
13. STATEMENT OF CONSENT (signature block)

IMPORTANT: Do NOT use placeholders like [Insert Phone Number]. Use actual values:
- PI contact: (555) 867-5309
- 24-hour emergency number: (555) 123-4567
- IRB contact: (555) 246-8101"""

        else:
            # Initial generation - intentionally less detailed to demonstrate iteration
            system_prompt = """You are writing a patient consent form for a clinical trial.
Write using medical terminology where appropriate. Focus on being thorough
and professional. Include standard legal protections."""

            user_prompt = f"""Generate a patient consent form for this study:

{json.dumps(study_info, indent=2)}

Include these sections:
1. STUDY TITLE
2. INTRODUCTION
3. PURPOSE OF THE STUDY
4. STUDY PROCEDURES
5. RISKS AND DISCOMFORTS
6. POTENTIAL BENEFITS
7. ALTERNATIVES
8. CONFIDENTIALITY
9. COSTS AND COMPENSATION
10. VOLUNTARY PARTICIPATION
11. WITHDRAWAL FROM THE STUDY
12. CONTACT INFORMATION (PI: 555-867-5309, IRB: 555-246-8101)
13. STATEMENT OF CONSENT (signature block)

Write in a professional medical/scientific style."""

        # Generate consent form
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )

        return response.choices[0].message.content


class EthicsAuditorAgent:
    """
    Evaluates consent forms against IRB criteria.
    """

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client
        self.criteria = ConsentEvaluationCriteria()

    def evaluate(self, consent_form: str, study_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate consent form against IRB criteria.

        Args:
            consent_form: Generated consent form text
            study_info: Study details for context

        Returns:
            Evaluation results dictionary
        """

        # Calculate readability
        readability_score = self.calculate_readability(consent_form)
        word_count = len(consent_form.split())

        system_prompt = """You are an IRB ethics reviewer evaluating patient consent forms.
You are STRICT but fair. Your job is to protect research participants.

Your evaluation must be thorough and specific:
1. Check all 11 required elements (45 CFR 46.116)
2. Assess readability - target is 8th grade or below. If above 8.0, this is a critical issue.
3. Flag ANY coercive or exculpatory language - these are automatic failures
4. Ensure balanced presentation of risks and benefits
5. Verify the experimental/investigational nature is CLEARLY stated
6. Contact information MUST include a 24-hour emergency number
7. Technical/medical terms MUST be explained in plain language

Be constructive: provide specific, actionable feedback.

A consent form is APPROVED only if:
- ALL required elements are present AND adequate
- ALL quality checks pass
- Readability score is ≤ 8.0 (8th grade)
- No critical issues remain
- 24-hour contact number is provided
- Medical terms are explained simply

If readability is above 8.0, add this to critical_issues.
If 24-hour emergency contact is missing, add this to critical_issues.
If medical jargon is not explained, add this to critical_issues.

Respond with a JSON object containing your evaluation."""

        user_prompt = f"""Evaluate this consent form against IRB criteria:

CONSENT FORM:
{consent_form}

STUDY INFORMATION (for reference):
{json.dumps(study_info, indent=2)}

CALCULATED METRICS:
- Readability Score (Flesch-Kincaid Grade Level): {readability_score}
- Word Count: {word_count}
- Readability Target: ≤ 8.0 (8th grade or below)

Evaluate and respond with a JSON object in this exact format:
{{
    "required_elements": {{
        "study_purpose": {{"present": true/false, "adequate": true/false, "feedback": "specific feedback"}},
        "procedures": {{"present": true/false, "adequate": true/false, "feedback": "specific feedback"}},
        "duration": {{"present": true/false, "adequate": true/false, "feedback": "specific feedback"}},
        "risks": {{"present": true/false, "adequate": true/false, "feedback": "specific feedback"}},
        "benefits": {{"present": true/false, "adequate": true/false, "feedback": "specific feedback"}},
        "alternatives": {{"present": true/false, "adequate": true/false, "feedback": "specific feedback"}},
        "confidentiality": {{"present": true/false, "adequate": true/false, "feedback": "specific feedback"}},
        "compensation": {{"present": true/false, "adequate": true/false, "feedback": "specific feedback"}},
        "contact_info": {{"present": true/false, "adequate": true/false, "feedback": "specific feedback"}},
        "voluntary_participation": {{"present": true/false, "adequate": true/false, "feedback": "specific feedback"}},
        "withdrawal_rights": {{"present": true/false, "adequate": true/false, "feedback": "specific feedback"}}
    }},
    "quality_checks": {{
        "no_coercive_language": {{"pass": true/false, "examples": ["any coercive phrases found"]}},
        "balanced_risk_benefit": {{"pass": true/false, "feedback": "assessment"}},
        "clear_experimental_nature": {{"pass": true/false, "feedback": "assessment"}},
        "no_exculpatory_language": {{"pass": true/false, "examples": ["any exculpatory phrases found"]}},
        "appropriate_length": {{"pass": true/false, "word_count": {word_count}}}
    }},
    "critical_issues": ["list of issues that MUST be fixed before approval"],
    "suggestions": ["optional improvements"]
}}

Be strict but fair. Mark something as inadequate only if it truly fails to meet requirements."""

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        # Parse the LLM response
        response_text = response.choices[0].message.content

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text

        try:
            evaluation = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback evaluation if JSON parsing fails
            evaluation = {
                "required_elements": {},
                "quality_checks": {},
                "critical_issues": ["Unable to parse evaluation - manual review required"],
                "suggestions": []
            }

        # Add calculated metrics
        evaluation["readability_score"] = readability_score
        evaluation["word_count"] = word_count

        # Calculate overall score
        evaluation["overall_score"] = self._calculate_overall_score(evaluation, readability_score)

        # Determine approval status
        evaluation["approved"] = self._check_approval(evaluation, readability_score)

        # Generate summary feedback
        evaluation["summary_feedback"] = self._generate_summary_feedback(evaluation, readability_score)

        return evaluation

    def calculate_readability(self, text: str) -> float:
        """
        Calculate Flesch-Kincaid grade level.

        Formula: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59

        Target: ≤ 8.0 (8th grade or lower)
        """
        # Count sentences (rough estimate)
        sentences = len(re.findall(r'[.!?]+', text))
        sentences = max(sentences, 1)

        # Count words
        words_list = re.findall(r'\b[a-zA-Z]+\b', text)
        words = len(words_list)
        words = max(words, 1)

        # Estimate syllables (simplified)
        def count_syllables(word: str) -> int:
            word = word.lower()
            vowels = "aeiou"
            count = 0
            prev_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            # Handle silent e
            if word.endswith('e') and count > 1:
                count -= 1
            return max(count, 1)

        syllables = sum(count_syllables(w) for w in words_list)

        # Flesch-Kincaid Grade Level
        grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
        return round(grade, 1)

    def _calculate_overall_score(self, evaluation: Dict[str, Any], readability_score: float) -> float:
        """Calculate overall score out of 100."""
        score = 100.0

        # Deduct for missing/inadequate required elements (5 points each)
        required = evaluation.get("required_elements", {})
        for element, status in required.items():
            if isinstance(status, dict):
                if not status.get("present", False):
                    score -= 5
                elif not status.get("adequate", False):
                    score -= 3

        # Deduct for failed quality checks (5 points each)
        quality = evaluation.get("quality_checks", {})
        for check, status in quality.items():
            if isinstance(status, dict) and not status.get("pass", True):
                score -= 5

        # Deduct for readability issues
        if readability_score > 8.0:
            score -= min(15, (readability_score - 8.0) * 5)

        # Deduct for critical issues (3 points each)
        critical_issues = evaluation.get("critical_issues", [])
        score -= len(critical_issues) * 3

        return max(0, round(score, 0))

    def _check_approval(self, evaluation: Dict[str, Any], readability_score: float) -> bool:
        """Determine if consent form is approved."""
        # Check all required elements are present and adequate
        required = evaluation.get("required_elements", {})
        for element, status in required.items():
            if isinstance(status, dict):
                if not status.get("present", False) or not status.get("adequate", False):
                    return False

        # Check all quality checks pass
        quality = evaluation.get("quality_checks", {})
        for check, status in quality.items():
            if isinstance(status, dict) and not status.get("pass", True):
                return False

        # Check readability
        if readability_score > 8.0:
            return False

        # Check no critical issues
        if evaluation.get("critical_issues", []):
            return False

        return True

    def _generate_summary_feedback(self, evaluation: Dict[str, Any], readability_score: float) -> str:
        """Generate consolidated feedback for the generator."""
        feedback_parts = []

        # Readability feedback
        if readability_score > 8.0:
            feedback_parts.append(
                f"({len(feedback_parts) + 1}) Readability is at grade level {readability_score}, "
                f"but target is 8.0 or below. Simplify language, use shorter sentences, "
                f"and explain medical terms in plain words."
            )

        # Required elements feedback
        required = evaluation.get("required_elements", {})
        for element, status in required.items():
            if isinstance(status, dict):
                if not status.get("present", False):
                    feedback_parts.append(
                        f"({len(feedback_parts) + 1}) Missing required element: {element.replace('_', ' ')}. "
                        f"{status.get('feedback', '')}"
                    )
                elif not status.get("adequate", False):
                    feedback_parts.append(
                        f"({len(feedback_parts) + 1}) {element.replace('_', ' ').title()} needs improvement: "
                        f"{status.get('feedback', '')}"
                    )

        # Quality checks feedback
        quality = evaluation.get("quality_checks", {})
        for check, status in quality.items():
            if isinstance(status, dict) and not status.get("pass", True):
                check_name = check.replace("_", " ")
                if "examples" in status and status["examples"]:
                    examples = "; ".join(status["examples"][:2])
                    feedback_parts.append(
                        f"({len(feedback_parts) + 1}) Quality check failed - {check_name}. "
                        f"Examples found: \"{examples}\""
                    )
                else:
                    feedback_parts.append(
                        f"({len(feedback_parts) + 1}) Quality check failed - {check_name}. "
                        f"{status.get('feedback', '')}"
                    )

        # Critical issues
        for issue in evaluation.get("critical_issues", []):
            if issue not in str(feedback_parts):  # Avoid duplicates
                feedback_parts.append(f"({len(feedback_parts) + 1}) {issue}")

        if not feedback_parts:
            return "All requirements met. Document approved."

        return " ".join(feedback_parts)


class ConsentFormWorkflow:
    """
    Orchestrates the generator-evaluator loop.
    """

    MAX_ITERATIONS = 5

    def __init__(self, llm_client: OpenAI):
        self.generator = ConsentFormGenerator(llm_client)
        self.evaluator = EthicsAuditorAgent(llm_client)

    def run(self, study_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the evaluator-optimizer loop until approved or max iterations.

        Returns:
            Dictionary with final consent form and evaluation results
        """
        print(f"\n{'='*70}")
        print(f"CONSENT FORM GENERATOR - EVALUATOR-OPTIMIZER WORKFLOW")
        print(f"Study: {study_info['title']}")
        print(f"{'='*70}")

        current_draft = None
        feedback = None
        history = []
        evaluation = None

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            print(f"\n[Iteration {iteration}/{self.MAX_ITERATIONS}]")
            print("-" * 40)

            # Generate (or revise)
            if iteration == 1:
                print("Generating initial draft...")
                current_draft = self.generator.generate(study_info)
            else:
                critical_count = len(evaluation.get('critical_issues', []))
                print(f"Revising based on {critical_count} critical issues...")
                current_draft = self.generator.generate(
                    study_info,
                    feedback=feedback,
                    previous_draft=current_draft
                )

            # Evaluate
            print("Evaluating draft...")
            evaluation = self.evaluator.evaluate(current_draft, study_info)

            # Log iteration
            history.append({
                "iteration": iteration,
                "score": evaluation["overall_score"],
                "readability": evaluation["readability_score"],
                "critical_issues": evaluation.get("critical_issues", []),
                "feedback_summary": evaluation["summary_feedback"]
            })

            # Display progress
            print(f"  Score: {evaluation['overall_score']}/100")
            print(f"  Readability: Grade {evaluation['readability_score']}")
            print(f"  Critical Issues: {len(evaluation.get('critical_issues', []))}")

            if evaluation["approved"]:
                print(f"\n✓ APPROVED after {iteration} iteration(s)")
                break
            else:
                feedback = evaluation["summary_feedback"]
                # Truncate feedback display
                display_feedback = feedback[:100] + "..." if len(feedback) > 100 else feedback
                print(f"\n  Feedback: {display_feedback}")

        if not evaluation["approved"]:
            print(f"\n✗ NOT APPROVED after {self.MAX_ITERATIONS} iterations")
            print("  Manual review required")

        return {
            "final_consent_form": current_draft,
            "approved": evaluation["approved"],
            "iterations": iteration,
            "iteration_history": history,
            "final_evaluation": evaluation
        }

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print formatted results."""

        # Iteration History Table
        print(f"\n{'='*70}")
        print("ITERATION HISTORY")
        print(f"{'='*70}")
        print()
        print("| Iter | Score | Readability | Critical Issues |")
        print("|------|-------|-------------|-----------------|")
        for h in results["iteration_history"]:
            approved_mark = " ✓" if h["iteration"] == results["iterations"] and results["approved"] else ""
            print(f"| {h['iteration']:4} | {h['score']:5.0f} | {h['readability']:11.1f} | {len(h['critical_issues']):15}{approved_mark} |")

        # Final Evaluation
        print(f"\n{'='*70}")
        print("FINAL EVALUATION")
        print(f"{'='*70}")

        eval_data = results["final_evaluation"]
        status = "APPROVED ✓" if results["approved"] else "NOT APPROVED ✗"
        readability_status = "✓" if eval_data["readability_score"] <= 8.0 else "✗"

        print(f"\nOverall Score: {eval_data['overall_score']}/100")
        print(f"Status: {status}")
        print(f"Readability: Grade {eval_data['readability_score']} (Target: ≤8.0) {readability_status}")

        # Required Elements
        print("\nRequired Elements:")
        required = eval_data.get("required_elements", {})
        for element, status in required.items():
            if isinstance(status, dict):
                present = status.get("present", False)
                adequate = status.get("adequate", False)
                feedback = status.get("feedback", "")

                if present and adequate:
                    mark = "✓"
                    detail = feedback if feedback else "Complete"
                elif present:
                    mark = "△"
                    detail = f"Needs improvement: {feedback}"
                else:
                    mark = "✗"
                    detail = f"Missing: {feedback}"

                element_name = element.replace("_", " ").title()
                print(f"  {mark} {element_name}: {detail}")

        # Quality Checks
        print("\nQuality Checks:")
        quality = eval_data.get("quality_checks", {})
        for check, status in quality.items():
            if isinstance(status, dict):
                passed = status.get("pass", True)
                mark = "✓" if passed else "✗"
                check_name = check.replace("_", " ").title()

                if not passed:
                    examples = status.get("examples", [])
                    feedback = status.get("feedback", "")
                    if examples:
                        detail = f"Issues: {', '.join(examples[:2])}"
                    elif feedback:
                        detail = feedback
                    else:
                        detail = "Failed"
                    print(f"  {mark} {check_name}: {detail}")
                else:
                    print(f"  {mark} {check_name}")

        # Suggestions
        suggestions = eval_data.get("suggestions", [])
        if suggestions:
            print("\nSuggestions for Enhancement (optional):")
            for suggestion in suggestions:
                print(f"  - {suggestion}")

        # Final Consent Form
        print(f"\n{'='*70}")
        print("FINAL CONSENT FORM")
        print(f"{'='*70}")
        print()
        print(results["final_consent_form"])
        print()
        print(f"{'='*70}")


def main():
    """Main entry point."""
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

    # Create workflow
    workflow = ConsentFormWorkflow(client)

    # Run the evaluator-optimizer loop
    results = workflow.run(STUDY_INFO)

    # Print formatted results
    workflow.print_results(results)

    return results


if __name__ == "__main__":
    main()
