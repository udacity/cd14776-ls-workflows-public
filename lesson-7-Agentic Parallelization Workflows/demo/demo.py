"""
Module 7 Demo: Parallelization Workflow - Cross-Database Adverse Drug Reaction Detector

This module demonstrates a parallelization workflow that processes independent tasks
concurrently and aggregates results from multiple data sources:
- FAERS Agent: FDA Adverse Event Reporting System
- Literature Agent: PubMed case reports and studies
- Social Agent: Simulated patient forum data

All agents run concurrently using ThreadPoolExecutor, then a Summary Agent synthesizes findings.

Learning Objective: Implement a parallel workflow using Python to perform concurrent analysis and synthesize the results.
"""

import os
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ls_action_space.action_space import query_faers, query_pubmed


# =============================================================================
# PARALLEL AGENT BASE
# =============================================================================

class ParallelAgentBase:
    """Base class for agents that run in parallel."""

    def __init__(self, name: str, llm_client: OpenAI):
        self.name = name
        self.llm = llm_client

    def search(self, drug_name: str, adverse_event: str) -> Dict[str, Any]:
        """Search this agent's data source. Must be implemented by subclasses."""
        raise NotImplementedError

    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Helper method to call the LLM."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content


# =============================================================================
# FAERS AGENT
# =============================================================================

class FAERSAgent(ParallelAgentBase):
    """Searches FDA FAERS database for adverse event reports."""

    def __init__(self, llm_client: OpenAI):
        super().__init__("FAERS Agent", llm_client)

    def search(self, drug_name: str, adverse_event: str) -> Dict[str, Any]:
        """Search FAERS for drug-AE association."""
        faers_data = query_faers(drug_name)
        total_events = faers_data.get("total_events", 0)
        serious_events = faers_data.get("serious_events", 0)
        top_reactions = faers_data.get("top_reactions", [])

        if total_events == 0:
            return {
                "source": "FAERS",
                "drug": drug_name,
                "adverse_event": adverse_event,
                "total_reports": 0,
                "reports_with_ae": 0,
                "signal_strength": "None",
                "interpretation": "No FAERS data available for this drug."
            }

        # Check if adverse event is in top reactions
        ae_lower = adverse_event.lower()
        num_with_ae = 0
        for r in top_reactions:
            reaction_name, count = r["reaction"], r["count"]
            if ae_lower in reaction_name.lower() or reaction_name.lower() in ae_lower:
                num_with_ae = count
                break

        ae_frequency = num_with_ae / total_events if total_events > 0 else 0.0

        # Determine signal strength
        if num_with_ae >= 100 and ae_frequency >= 0.02:
            signal_strength = "Strong"
        elif num_with_ae >= 20 and ae_frequency >= 0.01:
            signal_strength = "Moderate"
        elif num_with_ae >= 5:
            signal_strength = "Weak"
        else:
            signal_strength = "None"

        # LLM interpretation
        interpretation = self._call_llm(
            f"""FAERS data for {drug_name}: {total_events} total reports, {num_with_ae} mention "{adverse_event}" ({ae_frequency*100:.1f}%).
Top reactions: {[r for r in top_reactions[:5]]}
Provide a 2-sentence clinical interpretation of this drug-AE association.""",
            "You are a pharmacovigilance expert. Be concise."
        )

        return {
            "source": "FAERS",
            "drug": drug_name,
            "adverse_event": adverse_event,
            "total_reports": total_events,
            "reports_with_ae": num_with_ae,
            "ae_frequency": ae_frequency,
            "signal_strength": signal_strength,
            "interpretation": interpretation
        }


# =============================================================================
# LITERATURE AGENT
# =============================================================================

class LiteratureAgent(ParallelAgentBase):
    """Searches PubMed for case reports and studies."""

    def __init__(self, llm_client: OpenAI):
        super().__init__("Literature Agent", llm_client)

    def search(self, drug_name: str, adverse_event: str) -> Dict[str, Any]:
        """Search PubMed for drug-AE literature evidence."""
        query = f"{drug_name} {adverse_event} adverse event"
        articles = query_pubmed(query, max_results=20)
        total_articles = len(articles)

        if total_articles == 0:
            return {
                "source": "PubMed",
                "drug": drug_name,
                "adverse_event": adverse_event,
                "total_articles": 0,
                "signal_strength": "None",
                "interpretation": "No relevant literature found in PubMed."
            }

        # Classify articles by type
        case_reports = sum(1 for a in articles if "case" in str(a.get("pub_types", [])).lower())
        reviews = sum(1 for a in articles if "review" in str(a.get("pub_types", [])).lower())
        meta_analyses = sum(1 for a in articles if "meta-analysis" in str(a.get("pub_types", [])).lower())

        # Determine signal strength
        if meta_analyses > 0 or total_articles >= 20:
            signal_strength = "Strong"
        elif total_articles >= 10:
            signal_strength = "Moderate"
        elif total_articles >= 3:
            signal_strength = "Weak"
        else:
            signal_strength = "None"

        # Get years
        years = [a.get("pub_year") for a in articles if a.get("pub_year")]

        # LLM interpretation
        interpretation = self._call_llm(
            f"""PubMed search for {drug_name} and {adverse_event} found {total_articles} articles:
- Case reports: {case_reports}, Reviews: {reviews}, Meta-analyses: {meta_analyses}
- Publication years: {min(years) if years else 'N/A'} to {max(years) if years else 'N/A'}
Provide a 2-sentence interpretation of the literature evidence.""",
            "You are a medical literature expert. Be concise."
        )

        return {
            "source": "PubMed",
            "drug": drug_name,
            "adverse_event": adverse_event,
            "total_articles": total_articles,
            "article_types": {"case_reports": case_reports, "reviews": reviews, "meta_analyses": meta_analyses},
            "year_range": f"{min(years) if years else 'N/A'}-{max(years) if years else 'N/A'}",
            "signal_strength": signal_strength,
            "interpretation": interpretation
        }


# =============================================================================
# SOCIAL MEDIA AGENT (Simulated)
# =============================================================================

class SocialMediaAgent(ParallelAgentBase):
    """Simulates patient forum/social media monitoring."""

    # Mock data for known drug-AE combinations
    MOCK_DATA = {
        ("metformin", "lactic acidosis"): {"mentions": 127, "negative_pct": 0.85, "trend": "stable"},
        ("fluoroquinolone", "tendon rupture"): {"mentions": 342, "negative_pct": 0.92, "trend": "increasing"},
        ("statin", "muscle pain"): {"mentions": 856, "negative_pct": 0.78, "trend": "stable"},
    }

    def __init__(self, llm_client: OpenAI):
        super().__init__("Social Media Agent", llm_client)

    def search(self, drug_name: str, adverse_event: str) -> Dict[str, Any]:
        """Search simulated social media data."""
        # Look for match in mock data
        drug_lower = drug_name.lower()
        ae_lower = adverse_event.lower()

        mock_result = None
        for (drug, ae), data in self.MOCK_DATA.items():
            if drug_lower in drug or drug in drug_lower:
                if ae_lower in ae or ae in ae_lower:
                    mock_result = data
                    break

        # Default if no match
        if mock_result is None:
            mock_result = {"mentions": 45, "negative_pct": 0.72, "trend": "stable"}

        mentions = mock_result["mentions"]
        negative_pct = mock_result["negative_pct"]

        # Determine signal strength
        if mentions >= 200 and negative_pct >= 0.85:
            signal_strength = "Strong"
        elif mentions >= 50 and negative_pct >= 0.70:
            signal_strength = "Moderate"
        elif mentions >= 10:
            signal_strength = "Weak"
        else:
            signal_strength = "None"

        interpretation = self._call_llm(
            f"""Social media monitoring for {drug_name} and {adverse_event}:
- Mentions: {mentions}, Negative sentiment: {negative_pct*100:.0f}%, Trend: {mock_result['trend']}
Provide a 2-sentence interpretation. Note this is simulated data.""",
            "You are a patient insights analyst. Be concise."
        )

        return {
            "source": "Social Media (simulated)",
            "drug": drug_name,
            "adverse_event": adverse_event,
            "total_mentions": mentions,
            "negative_sentiment_pct": negative_pct,
            "temporal_trend": mock_result["trend"],
            "signal_strength": signal_strength,
            "interpretation": interpretation
        }


# =============================================================================
# SUMMARY AGENT
# =============================================================================

class SummaryAgent:
    """Aggregates results from parallel agents into coherent report."""

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def synthesize(self, drug_name: str, adverse_event: str, agent_results: List[Dict]) -> Dict[str, Any]:
        """Synthesize parallel agent results into unified assessment."""
        # Calculate overall signal strength
        strength_scores = {"Strong": 3, "Moderate": 2, "Weak": 1, "None": 0}
        signals = [r.get("signal_strength", "None") for r in agent_results]
        avg_score = sum(strength_scores.get(s, 0) for s in signals) / len(signals)

        if avg_score >= 2.5:
            overall_signal, confidence = "Established", "High"
        elif avg_score >= 1.5:
            overall_signal, confidence = "Emerging", "Medium"
        elif avg_score >= 0.5:
            overall_signal, confidence = "Potential", "Low"
        else:
            overall_signal, confidence = "Unlikely", "Low"

        # Build evidence summary
        evidence = {}
        for r in agent_results:
            source = r.get("source", "Unknown")
            if "FAERS" in source:
                evidence["faers"] = f"{r.get('reports_with_ae', 0)} reports, {r.get('signal_strength')} signal"
            elif "PubMed" in source:
                evidence["literature"] = f"{r.get('total_articles', 0)} articles, {r.get('signal_strength')} signal"
            elif "Social" in source:
                evidence["social"] = f"{r.get('total_mentions', 0)} mentions (simulated)"

        # LLM synthesis
        synthesis_prompt = f"""Synthesize these multi-source findings for {drug_name} and {adverse_event}:
{json.dumps(agent_results, indent=2, default=str)}

Return JSON with: key_findings (3 items), data_gaps (2 items), recommended_actions (2 items), executive_summary (2 sentences)"""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a pharmacovigilance expert. Return only valid JSON."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            synthesis_data = json.loads(response.choices[0].message.content)
        except Exception:
            synthesis_data = {
                "key_findings": [f"Signal detected for {drug_name}-{adverse_event} association"],
                "data_gaps": ["Limited prospective data"],
                "recommended_actions": ["Review individual case reports"],
                "executive_summary": f"Multi-source evidence suggests {overall_signal.lower()} signal."
            }

        return {
            "drug": drug_name,
            "adverse_event": adverse_event,
            "overall_signal": overall_signal,
            "confidence": confidence,
            "evidence_summary": evidence,
            "key_findings": synthesis_data.get("key_findings", []),
            "data_gaps": synthesis_data.get("data_gaps", []),
            "recommended_actions": synthesis_data.get("recommended_actions", []),
            "executive_summary": synthesis_data.get("executive_summary", "")
        }


# =============================================================================
# PARALLEL ORCHESTRATOR
# =============================================================================

class ParallelADRDetector:
    """
    Orchestrates parallel execution of data source agents.

    This is the core of the parallelization pattern:
    1. Define independent tasks (agent searches)
    2. Execute concurrently with ThreadPoolExecutor
    3. Collect results as they complete
    4. Synthesize into final output
    """

    def __init__(self, llm_client: OpenAI):
        self.faers_agent = FAERSAgent(llm_client)
        self.literature_agent = LiteratureAgent(llm_client)
        self.social_agent = SocialMediaAgent(llm_client)
        self.summary_agent = SummaryAgent(llm_client)

    def run_parallel(self, drug_name: str, adverse_event: str) -> Dict[str, Any]:
        """
        Run all agents in parallel and aggregate results.
        Uses ThreadPoolExecutor for concurrent execution.
        """
        print(f"\n{'='*70}")
        print(f"PARALLEL ADR DETECTION")
        print(f"Drug: {drug_name} | Adverse Event: {adverse_event}")
        print(f"{'='*70}")

        start_time = time.time()

        # Define agent tasks - these are independent and can run in parallel
        agents = [
            ("FAERS", self.faers_agent),
            ("Literature", self.literature_agent),
            ("Social", self.social_agent)
        ]

        # === KEY PATTERN: Parallel execution with ThreadPoolExecutor ===
        print(f"\n[PARALLEL EXECUTION - {len(agents)} agents]")
        results = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks concurrently
            future_to_agent = {
                executor.submit(agent.search, drug_name, adverse_event): name
                for name, agent in agents
            }

            # Collect results as they complete (not in submission order)
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    result = future.result()
                    results[agent_name] = result
                    elapsed = time.time() - start_time
                    print(f"  + {agent_name} completed ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"  - {agent_name} failed: {e}")
                    results[agent_name] = {"error": str(e), "source": agent_name}

        parallel_time = time.time() - start_time

        # Synthesize results
        print(f"\n[SYNTHESIS]")
        agent_results = [r for r in results.values() if "error" not in r]
        synthesis = self.summary_agent.synthesize(drug_name, adverse_event, agent_results)

        total_time = time.time() - start_time

        # Show timing advantage of parallelization
        print(f"\n[TIMING]")
        print(f"  Parallel execution: {parallel_time:.1f}s")
        print(f"  Total with synthesis: {total_time:.1f}s")
        print(f"  Sequential would be: ~{parallel_time * 3:.1f}s (3x longer)")

        return {
            "individual_results": results,
            "synthesis": synthesis,
            "timing": {"parallel_ms": int(parallel_time * 1000), "total_ms": int(total_time * 1000)}
        }


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_results(results: Dict[str, Any]) -> str:
    """Format the detection results for display."""
    lines = []

    # Individual agent results
    lines.append(f"\n{'='*70}")
    lines.append("INDIVIDUAL AGENT RESULTS")
    lines.append("="*70)

    for name, result in results.get("individual_results", {}).items():
        lines.append(f"\n[{name.upper()}]")
        if "error" in result:
            lines.append(f"  Error: {result['error']}")
        else:
            lines.append(f"  Signal Strength: {result.get('signal_strength', 'N/A')}")
            if "total_reports" in result:
                lines.append(f"  Reports with AE: {result.get('reports_with_ae', 0)} of {result.get('total_reports', 0)}")
            if "total_articles" in result:
                lines.append(f"  Articles Found: {result.get('total_articles', 0)}")
            if "total_mentions" in result:
                lines.append(f"  Social Mentions: {result.get('total_mentions', 0)} ({result.get('negative_sentiment_pct', 0)*100:.0f}% negative)")
            lines.append(f"  Interpretation: {result.get('interpretation', 'N/A')[:150]}...")

    # Synthesis
    synthesis = results.get("synthesis", {})
    lines.append(f"\n{'='*70}")
    lines.append("SYNTHESIS")
    lines.append("="*70)
    lines.append(f"\nOVERALL SIGNAL: {synthesis.get('overall_signal', 'N/A')} (Confidence: {synthesis.get('confidence', 'N/A')})")

    lines.append("\nEvidence Summary:")
    for source, summary in synthesis.get("evidence_summary", {}).items():
        lines.append(f"  - {source.upper()}: {summary}")

    lines.append("\nKey Findings:")
    for finding in synthesis.get("key_findings", [])[:3]:
        lines.append(f"  - {finding}")

    lines.append("\nRecommended Actions:")
    for action in synthesis.get("recommended_actions", [])[:2]:
        lines.append(f"  - {action}")

    lines.append(f"\nEXECUTIVE SUMMARY: {synthesis.get('executive_summary', 'N/A')}")

    return "\n".join(lines)


def main():
    """Main entry point for the parallel ADR detector demo."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

    print("="*70)
    print("PARALLEL ADR DETECTION DEMO")
    print("="*70)
    print("\nThis demo shows how parallelization improves efficiency:")
    print("  1. Multiple agents search different data sources concurrently")
    print("  2. ThreadPoolExecutor manages parallel execution")
    print("  3. Results are collected as they complete (as_completed)")
    print("  4. Summary agent synthesizes all findings")

    # Run parallel detection
    detector = ParallelADRDetector(client)
    results = detector.run_parallel("metformin", "lactic acidosis")

    # Display formatted results
    print(format_results(results))
    print("="*70)


if __name__ == "__main__":
    main()
