#!/usr/bin/env python3
"""
Module 4 Exercise: Gene-Drug Repurposing Scout

A three-agent sequential pipeline for drug repurposing discovery:
    DiseaseAgent -> GeneAgent -> ReportAgent

Learning Objective:
    Implement a sequential agentic workflow by creating and coordinating
    multiple agent classes.
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
# Base Agent (PROVIDED)
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
        """
        Validate and structure the disease query.

        Returns dict with: original_query, normalized_name, disease_category,
        key_features, inheritance_pattern, affected_systems, search_terms,
        validation_status, notes
        """
        # TODO: Create a prompt asking the LLM to analyze the disease query
        #       and return JSON with all fields listed in the docstring above.
        #       Use self.call_llm(prompt, json_mode=True) to get the response.
        #       Parse with json.loads() and add original_query to the result.
        #       Handle JSONDecodeError by returning a fallback dict.
        pass


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
        """
        Find genes and drugs associated with the disease.

        Returns dict with: disease, genes, primary_gene, drug_candidates,
        pathway_info, total_candidates_found
        """
        disease_name = disease_info.get("normalized_name", disease_info.get("original_query", ""))
        search_terms = disease_info.get("search_terms", [disease_name])

        # TODO: Implement the sequential workflow:
        #   1. Print status, then call: genes = self._find_genes(disease_name, search_terms)
        #   2. Print status, then call: drugs = self._find_drugs(genes)
        #   3. Print status, then call: analysis = self._analyze_candidates(disease_info, genes, drugs)
        #   4. Return dict with: disease, genes, primary_gene, drug_candidates,
        #      pathway_info, total_candidates_found (from analysis)
        pass

    def _find_genes(self, disease_name: str, search_terms: List[str]) -> List[str]:
        """Extract genes from PubMed literature. PROVIDED."""
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
        """Find drugs targeting the identified genes via OpenTargets. PROVIDED."""
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
        """
        Use LLM to analyze and rank drug candidates.

        Returns dict with: genes (list with symbol/mechanism/confidence),
        primary_gene, pathway_info, drug_candidates (ranked list with priority)
        """
        # TODO: Create a prompt that includes disease context (category, inheritance,
        #       features), genes found, and drug candidates (limit to 15).
        #       Ask LLM to return JSON with genes, primary_gene, pathway_info,
        #       and drug_candidates (each with priority and repurposing_rationale).
        #       Use self.call_llm(prompt, json_mode=True), parse with json.loads().
        #       Handle errors by returning a fallback dict.
        pass


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
        """
        Generate prioritized drug repurposing report.

        Input: dict with "disease" (from DiseaseAgent) and "genes" (from GeneAgent)
        Returns: markdown report string
        """
        disease = all_data.get("disease", {})
        genes = all_data.get("genes", {})

        # TODO: Create a prompt that includes disease info (name, category,
        #       inheritance, features) and gene info (primary_gene, genes,
        #       pathway_info, drug_candidates).
        #       Ask for a markdown report with: Executive Summary, Disease Profile,
        #       Target Analysis, Drug Candidates (top 5), Recommendations.
        #       Use self.call_llm(prompt, temperature=0.4) and add header/footer.
        pass


# =============================================================================
# Pipeline Orchestrator (PROVIDED)
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
        if disease_info is None:
            disease_info = {"original_query": disease_query, "normalized_name": "NOT IMPLEMENTED",
                           "validation_status": "TODO"}
        print(f"  -> {disease_info.get('normalized_name')} ({disease_info.get('validation_status')})")

        # Step 2: Gene & Drug Discovery
        print(f"\n[2/3] Gene Agent: Finding genes and drugs...")
        gene_info = self.gene_agent.run(disease_info)
        if gene_info is None:
            gene_info = {"genes": [], "primary_gene": "NOT IMPLEMENTED",
                        "drug_candidates": [], "total_candidates_found": 0}
        print(f"  -> {len(gene_info.get('genes', []))} genes, {gene_info.get('total_candidates_found', 0)} drug candidates")

        # Step 3: Report Generation
        print(f"\n[3/3] Report Agent: Generating report...")
        report = self.report_agent.run({"disease": disease_info, "genes": gene_info})
        if report is None:
            report = "REPORT NOT IMPLEMENTED - Complete the ReportAgent.run() method"

        return {"disease_info": disease_info, "gene_info": gene_info, "report": report}


# =============================================================================
# Main (PROVIDED)
# =============================================================================

def main():
    """Run the drug repurposing pipeline."""
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
