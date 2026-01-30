#!/usr/bin/env python3
"""
Module 3 Demo: Visualizing Workflows

Part of the course: Agentic Workflows for Life Sciences Research
Nanodegree: Life Sciences Agentic AI Nanodegree (Udacity)

Learning Objective:
    Visualize an agentic workflow model by representing agent tasks,
    interactions, and data flows in a diagram.

Scenario:
    Clinical Trial Pre-screen: Deterministic Form Fill vs Agentic Eligibility Hub
    Compare a simple linear form-validation workflow against a parallel
    multi-agent eligibility assessment system using Mermaid.js diagrams.
"""

import base64
import urllib.request
import urllib.parse
from pathlib import Path


def generate_deterministic_workflow() -> str:
    """
    Generate Mermaid.js code for a simple linear trial pre-screening workflow.

    The deterministic workflow:
    1. Patient data arrives
    2. Form validation (are all fields filled?)
    3. Age check (is age within range?)
    4. Lab values check (are values within thresholds?)
    5. Diagnosis check (does diagnosis match?)
    6. Output: Eligible or Ineligible
    """
    return """flowchart TD
    A[Patient Data Input] --> B{Form Validator}
    B -->|Valid| C{Age Check}
    B -->|Invalid| X[INELIGIBLE: Missing Data]
    C -->|Pass| D{Lab Values Check}
    C -->|Fail| X1[INELIGIBLE: Age Out of Range]
    D -->|Pass| E{Diagnosis Check}
    D -->|Fail| X2[INELIGIBLE: Lab Values Outside Thresholds]
    E -->|Pass| Y[ELIGIBLE]
    E -->|Fail| X3[INELIGIBLE: Diagnosis Mismatch]

    style Y fill:#90EE90
    style X fill:#FFB6C1
    style X1 fill:#FFB6C1
    style X2 fill:#FFB6C1
    style X3 fill:#FFB6C1"""


def generate_agentic_workflow() -> str:
    """
    Generate Mermaid.js code for a parallel multi-agent eligibility assessment.

    The agentic workflow:
    1. Patient data arrives at OrchestratorAgent
    2. Orchestrator dispatches to parallel specialist agents
    3. ResultAggregator collects all assessments
    4. ReasoningAgent handles edge cases and conflicts
    5. Human-in-the-loop checkpoint for borderline cases
    6. Final eligibility decision with explanation
    """
    return """flowchart TD
    subgraph Input
        A[Patient Data]
    end

    subgraph Orchestration
        B[OrchestratorAgent]
    end

    subgraph "Parallel Assessment"
        C1[AgeEligibilityAgent]
        C2[BiomarkerAgent]
        C3[ComorbidityAgent]
        C4[MedicationConflictAgent]
    end

    subgraph Synthesis
        D[ResultAggregator]
        E[ReasoningAgent]
    end

    subgraph "Human Review"
        F{Borderline Case?}
        G[Human Reviewer]
    end

    subgraph Output
        H[Final Decision + Explanation]
    end

    A --> B
    B -->|age criteria| C1
    B -->|biomarker data| C2
    B -->|medical history| C3
    B -->|medication list| C4
    C1 -->|age assessment| D
    C2 -->|biomarker assessment| D
    C3 -->|comorbidity assessment| D
    C4 -->|medication assessment| D
    D -->|combined results| E
    E -->|reasoned decision| F
    F -->|Yes| G
    F -->|No| H
    G -->|final approval| H

    style B fill:#87CEEB
    style C1 fill:#DDA0DD
    style C2 fill:#DDA0DD
    style C3 fill:#DDA0DD
    style C4 fill:#DDA0DD
    style E fill:#98FB98
    style G fill:#FFD700
    style H fill:#90EE90"""


def render_mermaid_to_png(mermaid_code: str, output_path: str) -> bool:
    """
    Render Mermaid diagram to PNG using mermaid.ink API.

    Args:
        mermaid_code: Valid Mermaid.js diagram code
        output_path: Path to save the PNG file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Encode the mermaid code to base64
        encoded = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('utf-8')

        # Build the mermaid.ink URL
        url = f"https://mermaid.ink/img/{encoded}?type=png"

        # Download the PNG
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            png_data = response.read()

        # Save to file
        with open(output_path, 'wb') as f:
            f.write(png_data)

        return True

    except Exception as e:
        print(f"  Warning: Could not render PNG via mermaid.ink: {e}")
        return False


def generate_comparison_table() -> str:
    """Generate a formatted comparison table of the two approaches."""
    return """| Aspect              | Deterministic                | Agentic                          |
|---------------------|------------------------------|----------------------------------|
| Processing          | Sequential, blocking         | Parallel, concurrent             |
| Edge Cases          | Fails or requires manual     | Reasons through with context     |
| Speed               | Fast for simple cases        | Faster for complex (parallelism) |
| Adaptability        | Requires code changes        | Can incorporate new criteria     |
| Auditability        | Clear pass/fail trail        | Requires explanation logging     |
| Human Involvement   | Only for failures            | Strategic checkpoints            |
| Explanations        | "Failed check X"             | "Considered X, Y, Z because..."  |"""


def run_workflow_visualization_demo():
    """
    Run the complete demonstration with PNG generation.

    Generates Mermaid diagrams and renders them to PNG files.
    """
    separator = "═" * 70
    output_dir = Path(__file__).parent

    print(f"""
{separator}
WORKFLOW VISUALIZATION DEMO: Clinical Trial Pre-screening
{separator}

SCENARIO:
A clinical trial site needs to pre-screen patients for eligibility.
Compare two approaches: deterministic form validation vs. intelligent
multi-agent assessment.
""")

    # Generate and render deterministic workflow
    print(f"{separator}")
    print("DETERMINISTIC WORKFLOW")
    print(f"{separator}\n")

    deterministic_code = generate_deterministic_workflow()
    deterministic_png = output_dir / "deterministic_workflow.png"

    print("Characteristics:")
    print("  • Linear, sequential processing")
    print("  • Binary pass/fail decisions")
    print("  • No contextual reasoning")
    print("  • Simple to implement and audit\n")

    print(f"Rendering diagram to: {deterministic_png}")
    if render_mermaid_to_png(deterministic_code, str(deterministic_png)):
        print(f"  ✓ Saved: {deterministic_png}\n")
    else:
        print(f"  Mermaid code (copy to https://mermaid.live):\n")
        print(f"```mermaid\n{deterministic_code}\n```\n")

    # Generate and render agentic workflow
    print(f"{separator}")
    print("AGENTIC WORKFLOW")
    print(f"{separator}\n")

    agentic_code = generate_agentic_workflow()
    agentic_png = output_dir / "agentic_workflow.png"

    print("Characteristics:")
    print("  • Parallel agent execution")
    print("  • Domain-specific reasoning per agent")
    print("  • Aggregation and synthesis")
    print("  • Human-in-the-loop for edge cases")
    print("  • Explainable decisions\n")

    print(f"Rendering diagram to: {agentic_png}")
    if render_mermaid_to_png(agentic_code, str(agentic_png)):
        print(f"  ✓ Saved: {agentic_png}\n")
    else:
        print(f"  Mermaid code (copy to https://mermaid.live):\n")
        print(f"```mermaid\n{agentic_code}\n```\n")

    # Comparison table
    print(f"{separator}")
    print("COMPARISON")
    print(f"{separator}\n")
    print(generate_comparison_table())

    # Key concepts
    print(f"\n{separator}")
    print("KEY WORKFLOW MODELING CONCEPTS")
    print(f"{separator}")
    print("""
1. AGENT REPRESENTATION
   - Boxes/nodes represent agents or processing steps
   - Subgraphs group related agents
   - Colors indicate agent types (orchestrator, specialist, human)

2. DATA FLOW
   - Arrows show direction of data/control flow
   - Labels on arrows describe what data is passed
   - Fan-out (one-to-many) shows parallelization
   - Fan-in (many-to-one) shows aggregation

3. DECISION POINTS
   - Diamond shapes for conditional branching
   - Human checkpoints shown distinctly
   - Feedback loops for iterative processes

4. PARALLEL EXECUTION
   - Agents in same subgraph can run concurrently
   - Synchronization points (aggregators) wait for all inputs
   - Reduces total execution time for independent tasks
""")

    # Summary of outputs
    print(f"{separator}")
    print("OUTPUT FILES")
    print(f"{separator}")
    print(f"  1. {deterministic_png}")
    print(f"  2. {agentic_png}")
    print(f"{separator}")


if __name__ == "__main__":
    run_workflow_visualization_demo()
