#!/usr/bin/env python3
"""
Module 4 Exercise: Gene-Drug Repurposing Scout

A three-agent sequential pipeline for drug repurposing discovery:
    DiseaseAgent -> GeneAgent -> ReportAgent

Demonstrates:
- Sequential state passing between agents
- Multi-source data integration (PubMed, OpenTargets)
- Agent specialization with focused expertise
"""

import json
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from ls_action_space.action_space import query_pubmed, query_opentargets


# =============================================================================
# Base Agent
# =============================================================================

class BaseAgent(ABC):
    """Abstract base class providing common LLM functionality for all agents."""

    def __init__(self, name: str, llm_client: OpenAI):
        self.name = name
        self.llm_client = llm_client
        self.system_prompt = ""
        self.model = "gpt-4o-mini"

    def call_llm(self, user_prompt: str, temperature: float = 0.3, json_mode: bool = False) -> str:
        """Make an LLM call with optional JSON response format."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs = {"model": self.model, "messages": messages, "temperature": temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.llm_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Execute the agent's main task."""
        pass


# =============================================================================
# Disease Agent
# =============================================================================

class DiseaseAgent(BaseAgent):
    """Agent 1: Validates and normalizes disease queries."""

    def __init__(self, llm_client: OpenAI):
        super().__init__("Disease Agent", llm_client)
        self.system_prompt = """You are an expert in rare diseases and medical genetics.
Your role is to validate disease queries and extract key clinical features.
You have deep knowledge of rare genetic disorders, lysosomal storage disorders,
metabolic diseases, neurogenetic conditions, and inheritance patterns."""

    def run(self, disease_query: str) -> Dict[str, Any]:
        """Validate and structure the disease query."""
        prompt = f"""Analyze this disease query and extract structured information.

Disease Query: "{disease_query}"

Return JSON with:
{{
    "normalized_name": "Standard disease name",
    "disease_category": "Classification (e.g., lysosomal storage disorder)",
    "key_features": ["Key clinical symptoms"],
    "inheritance_pattern": "e.g., Autosomal Recessive",
    "affected_systems": ["Affected organ systems"],
    "search_terms": ["Optimized PubMed search terms"],
    "validation_status": "Valid/Ambiguous/Unknown",
    "notes": "Any important clarifications"
}}"""

        try:
            result = json.loads(self.call_llm(prompt, json_mode=True))
            result["original_query"] = disease_query
            return result
        except json.JSONDecodeError:
            return {
                "original_query": disease_query,
                "normalized_name": disease_query,
                "disease_category": "Unknown",
                "key_features": [],
                "inheritance_pattern": "Unknown",
                "affected_systems": [],
                "search_terms": [disease_query],
                "validation_status": "Unknown",
                "notes": "Failed to parse LLM response"
            }


# =============================================================================
# Gene Agent
# =============================================================================

class GeneAgent(BaseAgent):
    """Agent 2: Finds causal genes and associated drugs."""

    def __init__(self, llm_client: OpenAI):
        super().__init__("Gene Agent", llm_client)
        self.system_prompt = """You are an expert in human genetics, disease mechanisms,
and drug repurposing. Identify genes associated with diseases and find existing
drugs that target those genes. Provide accurate, evidence-based assessments."""

    def run(self, disease_info: Dict[str, Any]) -> Dict[str, Any]:
        """Find genes and drugs associated with the disease."""
        disease_name = disease_info.get("normalized_name", disease_info.get("original_query", ""))
        search_terms = disease_info.get("search_terms", [disease_name])

        # Step 1: Find genes via PubMed
        print(f"    Searching PubMed for gene associations...")
        genes = self._find_genes(disease_name, search_terms)

        # Step 2: Find drugs via OpenTargets
        print(f"    Searching OpenTargets for drug candidates...")
        drugs = self._find_drugs(genes)

        # Step 3: Analyze and rank candidates
        print(f"    Analyzing candidates...")
        analysis = self._analyze_candidates(disease_info, genes, drugs)

        return {
            "disease": disease_name,
            "genes": analysis.get("genes", []),
            "primary_gene": analysis.get("primary_gene", genes[0] if genes else "Unknown"),
            "drug_candidates": analysis.get("drug_candidates", []),
            "pathway_info": analysis.get("pathway_info", ""),
            "total_candidates_found": len(analysis.get("drug_candidates", []))
        }

    def _find_genes(self, disease_name: str, search_terms: List[str]) -> List[str]:
        """Extract genes from PubMed literature."""
        all_genes = []

        for term in search_terms[:2]:
            results = query_pubmed(f"{term} gene mutation causal", max_results=10)
            if not results:
                continue

            abstracts = "\n\n".join([
                f"Title: {r.get('title', '')}\nAbstract: {r.get('abstract', '')}"
                for r in results[:5] if r.get('abstract')
            ])

            if abstracts:
                prompt = f"""From these abstracts about {disease_name}, extract causally associated gene symbols.

{abstracts}

Return JSON: {{"genes": ["GENE1", "GENE2", ...]}}"""
                try:
                    data = json.loads(self.call_llm(prompt, json_mode=True))
                    all_genes.extend(data.get("genes", []))
                except (json.JSONDecodeError, Exception):
                    pass

        # Deduplicate preserving order
        unique_genes = list(dict.fromkeys(g.upper() for g in all_genes))

        # Fallback to LLM knowledge if no genes found
        if not unique_genes:
            print(f"    Using LLM knowledge to identify genes...")
            prompt = f'What genes cause {disease_name}? Return JSON: {{"genes": ["GENE1", "GENE2"]}}'
            try:
                data = json.loads(self.call_llm(prompt, json_mode=True))
                unique_genes = [g.upper() for g in data.get("genes", [])]
            except (json.JSONDecodeError, Exception):
                pass

        return unique_genes[:5]

    def _find_drugs(self, genes: List[str]) -> List[Dict]:
        """Find drugs targeting the identified genes via OpenTargets."""
        all_drugs = []
        seen = set()

        for gene in genes[:3]:
            for drug_entry in query_opentargets(gene).get("known_drugs", []):
                drug_name = drug_entry.get("drug", {}).get("name", "")
                if drug_name and drug_name.lower() not in seen:
                    seen.add(drug_name.lower())
                    all_drugs.append({
                        "drug_name": drug_name,
                        "target_gene": gene,
                        "mechanism": drug_entry.get("mechanismOfAction", ""),
                        "current_indication": drug_entry.get("disease", {}).get("name", ""),
                        "max_phase": drug_entry.get("phase", 0),
                    })
        return all_drugs

    def _analyze_candidates(self, disease_info: Dict, genes: List[str], drugs: List[Dict]) -> Dict:
        """Use LLM to analyze and rank candidates."""
        disease_name = disease_info.get("normalized_name", "")

        prompt = f"""Analyze gene and drug candidates for {disease_name}.

Disease: {disease_info.get('disease_category', 'Unknown')} | {disease_info.get('inheritance_pattern', 'Unknown')}
Features: {', '.join(disease_info.get('key_features', []))}
Genes: {', '.join(genes) if genes else 'None'}
Drugs: {json.dumps(drugs[:15], indent=2) if drugs else 'None'}

Return JSON:
{{
    "genes": [{{"symbol": "GENE", "mechanism": "Role in disease", "confidence": "High/Medium/Low"}}],
    "primary_gene": "Main target",
    "pathway_info": "Pathway description",
    "drug_candidates": [{{
        "drug_name": "Name", "target_gene": "GENE", "mechanism": "MOA",
        "current_indication": "Current use", "repurposing_rationale": "Why it might work",
        "priority": "High/Medium/Low", "safety_notes": "Key considerations"
    }}]
}}"""

        try:
            return json.loads(self.call_llm(prompt, json_mode=True))
        except (json.JSONDecodeError, Exception):
            return {
                "genes": [{"symbol": g, "mechanism": "", "confidence": "Low"} for g in genes],
                "primary_gene": genes[0] if genes else "Unknown",
                "drug_candidates": [{**d, "priority": "Medium", "repurposing_rationale": ""} for d in drugs[:10]]
            }


# =============================================================================
# Report Agent
# =============================================================================

class ReportAgent(BaseAgent):
    """Agent 3: Generates the final drug repurposing report."""

    def __init__(self, llm_client: OpenAI):
        super().__init__("Report Agent", llm_client)
        self.system_prompt = """You are an expert medical writer specializing in drug repurposing.
Synthesize biomedical data into clear, actionable reports for researchers."""

    def run(self, all_data: Dict[str, Any]) -> str:
        """Generate prioritized drug repurposing report."""
        disease = all_data.get("disease", {})
        genes = all_data.get("genes", {})

        prompt = f"""Generate a drug repurposing report.

DISEASE: {disease.get('normalized_name', 'Unknown')}
- Category: {disease.get('disease_category', 'Unknown')}
- Inheritance: {disease.get('inheritance_pattern', 'Unknown')}
- Features: {', '.join(disease.get('key_features', []))}

GENES:
- Primary: {genes.get('primary_gene', 'Unknown')}
- All: {json.dumps(genes.get('genes', []))}
- Pathway: {genes.get('pathway_info', '')}

DRUGS ({genes.get('total_candidates_found', 0)} found):
{json.dumps(genes.get('drug_candidates', [])[:10], indent=2)}

Create markdown report with:
1. **Executive Summary** - Disease overview + top 3 candidates table (Rank|Drug|Rationale|Priority)
2. **Disease Profile** - Clinical features, genetic basis
3. **Target Analysis** - Primary gene, pathway context
4. **Drug Candidates** - Top 5 with mechanism, indication, rationale, safety
5. **Recommendations** - Top candidates, key risks

Date: {datetime.now().strftime('%Y-%m-%d')}"""

        report = self.call_llm(prompt, temperature=0.4)
        return f"\n{'='*70}\nDRUG REPURPOSING REPORT\n{'='*70}\n\n{report}\n{'='*70}"


# =============================================================================
# Pipeline Orchestrator
# =============================================================================

class DrugRepurposingPipeline:
    """Orchestrates the sequential drug repurposing workflow."""

    def __init__(self, llm_client: OpenAI):
        self.disease_agent = DiseaseAgent(llm_client)
        self.gene_agent = GeneAgent(llm_client)
        self.report_agent = ReportAgent(llm_client)

    def run(self, disease_query: str) -> Dict[str, Any]:
        """Execute the full pipeline with state passing between agents."""
        print(f"\n{'='*70}")
        print(f"DRUG REPURPOSING PIPELINE: {disease_query}")
        print(f"{'='*70}")

        # Step 1: Disease Validation
        print(f"\n[1/3] Disease Agent: Validating...")
        disease_info = self.disease_agent.run(disease_query)
        print(f"  -> {disease_info.get('normalized_name')} ({disease_info.get('validation_status')})")

        # Step 2: Gene & Drug Discovery
        print(f"\n[2/3] Gene Agent: Finding genes and drugs...")
        gene_info = self.gene_agent.run(disease_info)
        print(f"  -> {len(gene_info.get('genes', []))} genes, {gene_info.get('total_candidates_found', 0)} drug candidates")

        # Step 3: Report Generation
        print(f"\n[3/3] Report Agent: Generating report...")
        report = self.report_agent.run({"disease": disease_info, "genes": gene_info})

        return {"disease_info": disease_info, "gene_info": gene_info, "report": report}


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the drug repurposing pipeline."""
    # Setup API client
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    llm_client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

    # Run pipeline
    pipeline = DrugRepurposingPipeline(llm_client)
    result = pipeline.run("Niemann-Pick disease type C")

    # Print report
    print(result["report"])

    # Summary
    print(f"\nPipeline complete: {len(result['gene_info'].get('genes', []))} genes, "
          f"{result['gene_info'].get('total_candidates_found', 0)} drug candidates identified.")


if __name__ == "__main__":
    main()
