"""
Module 5 Demo: Prompt Chaining Workflow for ICSR Processing

This module implements a three-step prompt chain for processing
Individual Case Safety Reports (ICSRs):
    Chain 1: ICSR Extractor - Extract structured data from narrative
    Chain 2: Risk Statement Writer - Generate regulatory-format risk statement
    Chain 3: Mitigation Suggestion Writer - Generate risk mitigation recommendations

Each chain's output is specifically formatted to be the ideal input for the next.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent / ".env")


# Sample ICSR data for demonstration
SAMPLE_ICSR = """
INDIVIDUAL CASE SAFETY REPORT (ICSR)
Report ID: US-2025-001234

PATIENT INFORMATION:
- Age: 72 years
- Sex: Female
- Weight: 65 kg
- Height: 162 cm

MEDICAL HISTORY:
- Type 2 diabetes mellitus (10 years)
- Hypertension (8 years)
- Mild chronic kidney disease (Stage 3a, eGFR 45 mL/min)
- History of UTI (resolved)

CURRENT MEDICATIONS:
- Metformin 1000mg twice daily (5 years)
- Lisinopril 10mg once daily (8 years)
- Empagliflozin 10mg once daily (started 3 weeks prior to event)

EVENT DESCRIPTION:
Three weeks after initiating empagliflozin therapy, the patient presented to
the emergency department with progressive confusion over 24 hours, rapid deep
breathing (Kussmaul respirations), and mild abdominal discomfort. Blood glucose
at presentation was 180 mg/dL (not significantly elevated). Arterial blood gas
revealed pH 7.15, pCO2 18 mmHg, bicarbonate 6 mEq/L. Serum lactate was mildly
elevated at 2.8 mmol/L. Serum ketones were strongly positive. Anion gap was 28.

Diagnosis: Euglycemic diabetic ketoacidosis (euDKA)

TREATMENT:
- Empagliflozin immediately discontinued
- IV fluid resuscitation (normal saline)
- IV insulin infusion
- Electrolyte replacement
- ICU admission for monitoring

OUTCOME:
Patient recovered after 4 days of ICU care. Acidosis resolved by day 2.
Mental status returned to baseline by day 3. Discharged on day 5 with
empagliflozin permanently discontinued. Follow-up planned with endocrinology.

CAUSALITY ASSESSMENT (Reporter):
Probable - temporal relationship, known class effect of SGLT2 inhibitors,
positive dechallenge (improvement after discontinuation)

REPORTER:
Dr. Jane Smith, Attending Physician
General Hospital Emergency Department
Date of Report: 2025-01-15
"""


def parse_json_response(response_text: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response, handling common formatting issues.

    Args:
        response_text: Raw text response from LLM

    Returns:
        Parsed JSON as dictionary

    Raises:
        ValueError: If JSON cannot be parsed
    """
    # Try direct parsing first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        if end > start:
            try:
                return json.loads(response_text[start:end].strip())
            except json.JSONDecodeError:
                pass

    # Try to extract JSON from generic code blocks
    if "```" in response_text:
        start = response_text.find("```") + 3
        # Skip language identifier if present
        newline = response_text.find("\n", start)
        if newline > start:
            start = newline + 1
        end = response_text.find("```", start)
        if end > start:
            try:
                return json.loads(response_text[start:end].strip())
            except json.JSONDecodeError:
                pass

    # Try to find JSON object in text
    brace_start = response_text.find("{")
    brace_end = response_text.rfind("}") + 1
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(response_text[brace_start:brace_end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {response_text[:200]}...")


def icsr_extractor_chain(icsr_text: str, llm_client: OpenAI) -> Dict[str, Any]:
    """
    Chain 1: Extract structured data from ICSR narrative.

    The output schema is specifically designed to feed Chain 2.

    Args:
        icsr_text: Raw ICSR narrative text
        llm_client: OpenAI client

    Returns:
        Dictionary with structured extraction following the defined schema
    """
    system_prompt = """You are a pharmacovigilance data extraction specialist.
Extract structured information from Individual Case Safety Reports (ICSRs).

Rules:
- Extract only explicitly stated information
- Use "Unknown" or "Not stated" for missing information
- Use MedDRA preferred terms for adverse events when possible
- List all seriousness criteria that apply (hospitalization, life-threatening, etc.)
- Be precise about temporal relationships

Output must be valid JSON matching the specified schema exactly."""

    user_prompt = f"""Extract structured data from this ICSR:

{icsr_text}

Output the following JSON structure exactly:
{{
    "patient": {{
        "age": <integer>,
        "sex": "<string>",
        "weight_kg": <float>,
        "relevant_history": ["<list of conditions relevant to the adverse event>"]
    }},
    "suspect_drug": {{
        "name": "<string>",
        "dose": "<string>",
        "duration_before_event": "<string>",
        "indication": "<string>"
    }},
    "concomitant_drugs": [
        {{
            "name": "<string>",
            "dose": "<string>",
            "potential_interaction": <boolean>
        }}
    ],
    "adverse_event": {{
        "description": "<brief clinical description>",
        "meddra_pt": "<MedDRA preferred term>",
        "onset_time": "<time from drug start to event>",
        "severity": "<Mild|Moderate|Severe|Life-threatening>",
        "serious_criteria": ["<list of seriousness criteria met>"],
        "outcome": "<Recovered|Recovering|Not recovered|Fatal|Unknown>"
    }},
    "causality_indicators": {{
        "temporal_relationship": "<description of timing>",
        "dechallenge": "<Positive|Negative|Not applicable|Unknown>",
        "rechallenge": "<Positive|Negative|Not done|Unknown>",
        "known_class_effect": <boolean>,
        "alternative_explanations": ["<list of possible alternative causes>"]
    }},
    "reporter_assessment": "<reporter's stated causality assessment>"
}}"""

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,  # Lower temperature for accurate extraction
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    response_text = response.choices[0].message.content
    return parse_json_response(response_text)


def risk_statement_writer_chain(extraction: Dict[str, Any], llm_client: OpenAI) -> str:
    """
    Chain 2: Generate a regulatory-format risk statement.

    Input: Structured extraction from Chain 1
    Output: Formatted risk statement for safety database entry

    The prompt explicitly references the Chain 1 output structure.

    Args:
        extraction: Output dict from icsr_extractor_chain
        llm_client: OpenAI client

    Returns:
        3-4 sentence risk statement in regulatory format
    """
    system_prompt = """You are a pharmacovigilance writer creating risk statements
for regulatory safety databases.

Write in formal regulatory style:
- Concise but complete
- Include key clinical parameters
- State causality assessment with supporting evidence
- Use standard medical terminology
- Do not editorialize or add recommendations (that comes later)

Format: 3-4 sentences covering patient context, event description, and causality.
Output ONLY the risk statement text, no additional formatting or explanation."""

    user_prompt = f"""Based on this extracted ICSR data, write a regulatory risk statement:

Patient: {json.dumps(extraction['patient'], indent=2)}

Suspect Drug: {json.dumps(extraction['suspect_drug'], indent=2)}

Adverse Event: {json.dumps(extraction['adverse_event'], indent=2)}

Causality Indicators: {json.dumps(extraction['causality_indicators'], indent=2)}

Reporter Assessment: {extraction['reporter_assessment']}

Write a 3-4 sentence risk statement following this format:
"A [age]-year-old [sex] with [relevant history] developed [adverse event] [time] after starting [drug] [dose] for [indication]. [Key clinical details]. [Causality statement based on indicators]."
"""

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,  # Slightly higher for natural writing
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content.strip()


def mitigation_suggestion_chain(
    extraction: Dict[str, Any],
    risk_statement: str,
    llm_client: OpenAI
) -> Dict[str, Any]:
    """
    Chain 3: Generate risk mitigation recommendations.

    Input: Both Chain 1 extraction AND Chain 2 risk statement
    Output: Structured mitigation recommendations

    Args:
        extraction: Output from Chain 1
        risk_statement: Output from Chain 2
        llm_client: OpenAI client

    Returns:
        Dictionary with structured mitigation recommendations
    """
    system_prompt = """You are a senior pharmacovigilance physician developing
risk mitigation strategies.

Consider:
- Immediate actions for this case type
- Broader prescriber education needs
- Patient risk factors that increase susceptibility
- Monitoring that could enable earlier detection
- Whether this represents a new signal or known risk

Be specific and actionable. Reference the drug class and mechanism where relevant.
Output must be valid JSON matching the specified schema exactly."""

    user_prompt = f"""Based on this case and risk assessment, provide mitigation recommendations:

RISK STATEMENT:
{risk_statement}

CASE DETAILS:
- Patient Risk Factors: {json.dumps(extraction['patient']['relevant_history'])}
- Drug: {extraction['suspect_drug']['name']} {extraction['suspect_drug']['dose']}
- Event: {extraction['adverse_event']['meddra_pt']}
- Severity: {extraction['adverse_event']['severity']}
- Outcome: {extraction['adverse_event']['outcome']}
- Causality: {extraction['reporter_assessment']}

Provide structured mitigation recommendations in this exact JSON format:
{{
    "immediate_actions": ["<list of actions for this specific case type>"],
    "prescriber_recommendations": ["<list of guidance for healthcare providers>"],
    "patient_counseling_points": ["<list of key patient education items>"],
    "monitoring_recommendations": ["<list of ongoing monitoring needs>"],
    "labeling_considerations": "<potential label update needs as single string>",
    "signal_priority": "<High|Medium|Low>",
    "signal_rationale": "<explanation for the priority level>",
    "follow_up_needed": ["<list of additional information to request>"]
}}"""

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    response_text = response.choices[0].message.content
    return parse_json_response(response_text)


def run_safety_signal_chain(icsr_text: str, llm_client: OpenAI) -> Dict[str, Any]:
    """
    Execute the full prompt chain.

    Shows the chain pattern: each step's output feeds the next step's input.

    Args:
        icsr_text: Raw ICSR narrative text
        llm_client: OpenAI client

    Returns:
        Dictionary containing outputs from all three chains
    """
    print("=" * 70)
    print("SAFETY SIGNAL PROMPT CHAIN")
    print("=" * 70)

    # Chain 1: Extract
    print("\n[CHAIN 1: ICSR Extraction]")
    print("-" * 40)
    try:
        extraction = icsr_extractor_chain(icsr_text, llm_client)
        print("Extracted structured data:")
        print(f"  - Patient: {extraction['patient']['age']}yo {extraction['patient']['sex']}")
        print(f"  - Suspect drug: {extraction['suspect_drug']['name']}")
        print(f"  - Adverse event: {extraction['adverse_event']['meddra_pt']}")
        print(f"  - Severity: {extraction['adverse_event']['severity']}")
    except Exception as e:
        print(f"ERROR in Chain 1: {e}")
        raise

    # Chain 2: Risk Statement (uses Chain 1 output)
    print("\n[CHAIN 2: Risk Statement Generation]")
    print("-" * 40)
    try:
        risk_statement = risk_statement_writer_chain(extraction, llm_client)
        print(f"Risk Statement:\n{risk_statement}")
    except Exception as e:
        print(f"ERROR in Chain 2: {e}")
        raise

    # Chain 3: Mitigation (uses Chain 1 + Chain 2 outputs)
    print("\n[CHAIN 3: Mitigation Recommendations]")
    print("-" * 40)
    try:
        mitigation = mitigation_suggestion_chain(extraction, risk_statement, llm_client)
        print(f"Signal Priority: {mitigation['signal_priority']}")
        print(f"\nImmediate Actions:")
        for action in mitigation['immediate_actions']:
            print(f"  • {action}")
        print(f"\nPrescriber Recommendations:")
        for rec in mitigation['prescriber_recommendations']:
            print(f"  • {rec}")
        print(f"\nPatient Counseling Points:")
        for point in mitigation['patient_counseling_points']:
            print(f"  • {point}")
        print(f"\nMonitoring Recommendations:")
        for rec in mitigation['monitoring_recommendations']:
            print(f"  • {rec}")
        print(f"\nLabeling Considerations:")
        print(f"  {mitigation['labeling_considerations']}")
        print(f"\nSignal Rationale:")
        print(f"  {mitigation['signal_rationale']}")
        print(f"\nFollow-up Needed:")
        for item in mitigation['follow_up_needed']:
            print(f"  • {item}")
    except Exception as e:
        print(f"ERROR in Chain 3: {e}")
        raise

    # Final output
    print("\n" + "=" * 70)
    print("CHAIN COMPLETE")
    print("=" * 70)

    return {
        "extraction": extraction,
        "risk_statement": risk_statement,
        "mitigation": mitigation
    }


def main():
    """Main entry point for the demo."""
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key.")
        return

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

    # Run the prompt chain
    result = run_safety_signal_chain(SAMPLE_ICSR, client)

    # Optionally save complete results to JSON
    output_file = "chain_output.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nComplete results saved to: {output_file}")


if __name__ == "__main__":
    main()
