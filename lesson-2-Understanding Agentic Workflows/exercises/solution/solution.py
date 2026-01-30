"""
Module 2 Exercise: Variant Interpretation Agent

Build a complete AI agent with explicit components:
- Goals (system prompt defining agent's purpose)
- Tools (ClinVar, NCBI Gene queries)
- Memory (evidence accumulator)
- Reasoning (LLM-based planning and synthesis)

Learning Objective:
    Define what constitutes a modern AI agent and identify its fundamental
    components by implementing them hands-on.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from ls_action_space.action_space import query_clinvar, query_ncbi_gene


VARIANTS = [
    {"hgvs_c": "NM_000546.6:c.743G>A", "hgvs_p": "p.Arg248Gln", "gene": "TP53"},
    {"hgvs_c": "NM_000059.4:c.5946delT", "hgvs_p": "p.Ser1982ArgfsTer22", "gene": "BRCA2"},
]


class VariantInterpretationAgent:
    """
    An AI agent that interprets genetic variants using the fundamental
    agent architecture: Goals + Tools + Memory + Reasoning.

    Components:
        1. GOALS - System prompt defining purpose, priorities, constraints
        2. TOOLS - External capabilities (ClinVar, NCBI Gene)
        3. MEMORY - Evidence gathered during interpretation
        4. REASONING - LLM-based planning and synthesis
    """

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

        # COMPONENT 1: GOALS (system prompt)
        self.system_prompt = """You are a clinical genetics agent. Your goals are:
1. Gather comprehensive evidence for each variant using available tools
2. Synthesize evidence into actionable clinical classifications
3. Clearly communicate uncertainty and limitations

You have access to tools. Always gather evidence before making conclusions.
Be thorough but efficient - don't query the same tool twice for the same data."""

        # COMPONENT 2: TOOLS (external capabilities)
        self.tools = {
            "query_clinvar": {
                "function": self._tool_query_clinvar,
                "description": "Query ClinVar database for variant clinical significance"
            },
            "query_gene_info": {
                "function": self._tool_query_gene_info,
                "description": "Query NCBI Gene for gene function and disease associations"
            }
        }

        # COMPONENT 3: MEMORY (tracks state across agent loop)
        self.evidence_memory: List[Dict] = []
        self.tools_used: List[str] = []
        self.session_history: List[str] = []

    # =========================================================================
    # COMPONENT 2: TOOL IMPLEMENTATIONS
    # =========================================================================

    def _tool_query_clinvar(self, hgvs: str) -> Dict[str, Any]:
        """
        Tool: Query ClinVar for variant clinical information.

        Args:
            hgvs: HGVS notation for the variant (e.g., "NM_000546.6:c.743G>A")

        Returns:
            Dict with: found, significance, review_status, conditions, pmid_count
        """
        # Step 1: Call ClinVar API
        clinvar_data = query_clinvar(hgvs)

        # Step 2: Check if variant was found
        if "error" in clinvar_data:
            return {
                "found": False,
                "significance": "Not in ClinVar",
                "review_status": "N/A",
                "conditions": [],
                "pmid_count": 0
            }

        # Step 3: Extract and structure relevant fields
        significance = clinvar_data.get("clinical_significance", "Unknown")
        review_status = clinvar_data.get("review_status", "Unknown")
        conditions = clinvar_data.get("conditions", [])
        pmids = clinvar_data.get("pubmed_pmids", [])

        # Step 4: Return structured dict
        return {
            "found": True,
            "significance": significance,
            "review_status": review_status,
            "conditions": conditions[:5],  # Limit to top 5
            "pmid_count": len(pmids)
        }

    def _tool_query_gene_info(self, gene_symbol: str) -> Dict[str, Any]:
        """
        Tool: Query NCBI Gene for gene context.

        Args:
            gene_symbol: Gene symbol (e.g., "TP53")

        Returns:
            Dict with: found, gene_name, summary, function_description
        """
        # Step 1: Call NCBI Gene API
        gene_data = query_ncbi_gene(gene_symbol)

        # Step 2: Check if gene was found
        if gene_data.get("error") or gene_data.get("note"):
            return {
                "found": False,
                "gene_name": gene_symbol,
                "summary": "Gene information not available",
                "function_description": "Unknown"
            }

        results = gene_data.get("results", [])
        if not results:
            return {
                "found": False,
                "gene_name": gene_symbol,
                "summary": "No results returned",
                "function_description": "Unknown"
            }

        # Step 3: Extract first result and structure
        entry = results[0]
        name = entry.get("name", gene_symbol)
        summary = entry.get("summary", "")
        if len(summary) > 500:
            summary = summary[:500] + "..."

        # Step 4: Return structured dict
        return {
            "found": True,
            "gene_name": name,
            "summary": summary,
            "function_description": entry.get("description", name)
        }

    # =========================================================================
    # COMPONENT 4: REASONING
    # =========================================================================

    def _decide_next_action(self, variant: Dict) -> Dict[str, Any]:
        """
        Reasoning: Decide what tool to use next or if ready to conclude.

        Uses LLM to examine current evidence and decide next step.

        Args:
            variant: The variant being interpreted

        Returns:
            Dict with: action ("use_tool" or "conclude"), tool (if use_tool), args (if use_tool)
        """
        hgvs = variant["hgvs_c"]
        gene = variant["gene"]

        # Step 1: Build prompt with current state
        tools_info = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])

        evidence_summary = "None yet" if not self.evidence_memory else json.dumps(
            self.evidence_memory, indent=2
        )

        prompt = f"""You are deciding what action to take for variant interpretation.

VARIANT: {hgvs} (Gene: {gene})

AVAILABLE TOOLS:
{tools_info}

TOOLS ALREADY USED: {self.tools_used if self.tools_used else "None"}

EVIDENCE GATHERED SO FAR:
{evidence_summary}

Based on the evidence gathered, decide:
1. If you need more information, choose a tool that hasn't been used yet
2. If you have sufficient evidence (ClinVar significance + gene context), conclude

Respond with JSON:
- To use a tool: {{"action": "use_tool", "tool": "<tool_name>", "args": {{"<arg_name>": "<value>"}}}}
- To conclude: {{"action": "conclude"}}

For query_clinvar, args should be: {{"hgvs": "{hgvs}"}}
For query_gene_info, args should be: {{"gene_symbol": "{gene}"}}"""

        try:
            # Step 2-3: Call LLM
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            # Step 4: Parse response
            decision = json.loads(response.choices[0].message.content)
            return decision

        except Exception as e:
            # Step 5: Fallback logic
            if "query_clinvar" not in self.tools_used:
                return {"action": "use_tool", "tool": "query_clinvar", "args": {"hgvs": hgvs}}
            elif "query_gene_info" not in self.tools_used:
                return {"action": "use_tool", "tool": "query_gene_info", "args": {"gene_symbol": gene}}
            else:
                return {"action": "conclude"}

    def _synthesize_conclusion(self, variant: Dict) -> Dict[str, Any]:
        """
        Reasoning: Synthesize gathered evidence into final interpretation.

        Args:
            variant: The variant being interpreted

        Returns:
            Dict with: classification, confidence, evidence_summary,
                      clinical_action, limitations
        """
        hgvs = variant["hgvs_c"]
        hgvs_p = variant.get("hgvs_p", "N/A")
        gene = variant["gene"]

        # Step 1: Build prompt with all evidence
        evidence_text = json.dumps(self.evidence_memory, indent=2)

        prompt = f"""Synthesize the following evidence into a clinical variant interpretation.

VARIANT INFORMATION:
- HGVS (coding): {hgvs}
- HGVS (protein): {hgvs_p}
- Gene: {gene}

GATHERED EVIDENCE:
{evidence_text}

Provide your interpretation as JSON with these fields:
- classification: One of "Pathogenic", "Likely Pathogenic", "VUS", "Likely Benign", "Benign"
- confidence: "High", "Medium", or "Low"
- evidence_summary: 2-3 sentence summary of key evidence points
- clinical_action: Recommended next steps for the clinician
- limitations: Array of any caveats or limitations in this interpretation

Base your classification on:
1. ClinVar clinical significance and review status
2. Gene function and known disease associations
3. Variant type (missense, frameshift, etc.)"""

        try:
            # Step 2-3: Call LLM
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            # Step 4: Parse response
            result = json.loads(response.choices[0].message.content)

            # Ensure all required fields exist
            return {
                "classification": result.get("classification", "VUS"),
                "confidence": result.get("confidence", "Low"),
                "evidence_summary": result.get("evidence_summary", "Insufficient evidence"),
                "clinical_action": result.get("clinical_action", "Genetic counseling recommended"),
                "limitations": result.get("limitations", ["Automated interpretation - requires clinical review"])
            }

        except Exception as e:
            # Step 5: Fallback
            return {
                "classification": "VUS",
                "confidence": "Low",
                "evidence_summary": f"Error during synthesis: {str(e)}",
                "clinical_action": "Manual review required",
                "limitations": ["Automated interpretation failed"]
            }

    # =========================================================================
    # AGENT LOOP (shows how components connect)
    # =========================================================================

    def interpret(self, variant: Dict) -> Dict[str, Any]:
        """
        Main agent loop: Perceive -> Think -> Act -> Observe -> Repeat

        This method orchestrates the components:
        1. Receives variant (Perception)
        2. Decides what tool to use (Reasoning)
        3. Executes tool (Action)
        4. Stores result (Memory)
        5. Repeats until ready to conclude
        6. Synthesizes final interpretation (Reasoning)
        """
        hgvs = variant["hgvs_c"]
        gene = variant["gene"]

        # Reset memory for new variant
        self.evidence_memory = []
        self.tools_used = []

        print(f"\n[Agent] Interpreting: {hgvs} ({gene})")

        # Agent loop (max 5 iterations for safety)
        for step in range(5):
            # THINK: Decide what to do next
            decision = self._decide_next_action(variant)

            # Handle unimplemented method
            if decision is None:
                if "query_clinvar" not in self.tools_used:
                    decision = {"action": "use_tool", "tool": "query_clinvar",
                               "args": {"hgvs": hgvs}}
                elif "query_gene_info" not in self.tools_used:
                    decision = {"action": "use_tool", "tool": "query_gene_info",
                               "args": {"gene_symbol": gene}}
                else:
                    decision = {"action": "conclude"}

            # Check if ready to conclude
            if decision.get("action") == "conclude":
                print(f"  [Step {step+1}] Concluding - sufficient evidence gathered")
                break

            # ACT: Execute the chosen tool
            tool_name = decision.get("tool", "query_clinvar")
            tool_args = decision.get("args", {})

            # Prevent duplicate tool calls
            if tool_name in self.tools_used:
                print(f"  [Step {step+1}] Skipping {tool_name} (already used)")
                continue

            print(f"  [Step {step+1}] Tool: {tool_name}")

            # Execute tool
            tool_fn = self.tools[tool_name]["function"]
            observation = tool_fn(**tool_args)

            # OBSERVE: Store result in memory
            if observation:
                self.evidence_memory.append({
                    "source": tool_name,
                    "data": observation
                })
                self.tools_used.append(tool_name)
                status = "found" if observation.get("found", True) else "not found"
                print(f"       -> Evidence {status}")
            else:
                print(f"       -> No data (method not implemented)")

        # SYNTHESIZE: Generate final interpretation
        result = self._synthesize_conclusion(variant)

        # Update session history
        self.session_history.append(hgvs)

        # Return result or placeholder
        if result is None:
            return {
                "variant": hgvs,
                "gene": gene,
                "classification": "NOT IMPLEMENTED",
                "confidence": "N/A",
                "evidence_summary": "Complete _synthesize_conclusion() method",
                "clinical_action": "N/A",
                "limitations": ["Agent methods not implemented"]
            }

        result["variant"] = hgvs
        result["gene"] = gene
        return result


def display_interpretation(result: Dict) -> None:
    """Display a single variant interpretation."""
    print(f"\n{'─'*60}")
    print(f"VARIANT: {result.get('variant', 'N/A')}")
    print(f"GENE: {result.get('gene', 'N/A')}")
    print(f"{'─'*60}")
    print(f"Classification: {result.get('classification', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    print(f"\nEvidence Summary:")
    print(f"  {result.get('evidence_summary', 'N/A')}")
    print(f"\nClinical Action:")
    print(f"  {result.get('clinical_action', 'N/A')}")
    if result.get('limitations'):
        print(f"\nLimitations:")
        for lim in result.get('limitations', []):
            print(f"  - {lim}")


def display_agent_state(agent: VariantInterpretationAgent) -> None:
    """Display the agent's memory state after processing."""
    print(f"\n{'='*60}")
    print("AGENT STATE (Memory Component)")
    print(f"{'='*60}")
    print(f"Variants analyzed this session: {len(agent.session_history)}")
    for v in agent.session_history:
        print(f"  - {v}")


def main():
    """Demonstrate the variant interpretation agent."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

    print("=" * 60)
    print("VARIANT INTERPRETATION AGENT")
    print("=" * 60)
    print("\nAgent Components:")
    print("  1. GOALS    - Clinical accuracy, evidence-based reasoning")
    print("  2. TOOLS    - ClinVar, NCBI Gene queries")
    print("  3. MEMORY   - Evidence accumulator, session history")
    print("  4. REASONING- LLM-based planning and synthesis")
    print("\nAgent Loop: Perceive -> Think -> Act -> Observe -> Synthesize")

    agent = VariantInterpretationAgent(client)

    for variant in VARIANTS:
        result = agent.interpret(variant)
        display_interpretation(result)

    display_agent_state(agent)
    print("=" * 60)


if __name__ == "__main__":
    main()
