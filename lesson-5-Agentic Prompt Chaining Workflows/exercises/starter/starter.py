"""
Module 5 Exercise: Variant-to-Diagnosis Chain

Chain 1 (ClinVar) → Chain 2 (Gene Enricher) → Chain 3 (Literature) → Chain 4 (ACMG Classifier)

Each chain produces a natural language summary that becomes context for the next chain.

Learning Objective:
    Implement a prompt chaining workflow where output of one step becomes input for the next.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from ls_action_space.action_space import query_clinvar, query_pubmed, query_ncbi_gene


VARIANTS_TO_CLASSIFY = [
    "NM_000546.6:c.743G>A",   # TP53 - well-characterized pathogenic
    "NM_000059.4:c.5946delT", # BRCA2 - frameshift
]


def clinvar_fetcher_chain(hgvs: str, llm_client: OpenAI) -> Dict[str, Any]:
    """
    Chain 1: Fetch and summarize ClinVar data for a variant.

    Returns: {hgvs, gene, clinvar_significance, review_status, conditions,
              supporting_pmids, allele_frequencies, summary}
    The "summary" field feeds Chain 2.
    """
    # TODO: Step 1 - Query ClinVar using query_clinvar(hgvs)

    # TODO: Step 2 - Handle variant not found case:
    #       If "error" in clinvar_data, return dict with hgvs, gene=None,
    #       clinvar_significance="Not in ClinVar", empty lists, and
    #       summary = f"The variant {hgvs} was not found in ClinVar database."

    # TODO: Step 3 - Extract structured data from ClinVar response:
    #       gene, clinical_significance, review_status, conditions, pubmed_pmids

    # TODO: Step 4 - Use LLM to generate 2-3 sentence summary covering:
    #       classification, review status, key conditions, publication count

    # TODO: Step 5 - Return dict with structured data AND summary
    pass


def gene_enricher_chain(chain1_output: Dict[str, Any], llm_client: OpenAI) -> Dict[str, Any]:
    """
    Chain 2: Enrich with gene context from NCBI Gene.

    Returns: {**chain1_output, gene_phenotypes, inheritance_patterns,
              gene_summary, gene_function, gene_disease_context}
    The "gene_disease_context" field feeds Chain 3.
    """
    # TODO: Step 1 - Extract gene from chain1_output
    #       If no gene, return chain1_output with defaults and
    #       gene_disease_context = "Gene information not available."

    # TODO: Step 2 - Query NCBI Gene using query_ncbi_gene(gene)
    #       Extract: gene summary, function/name, aliases, phenotypes

    # TODO: Step 3 - Use LLM to synthesize gene-disease context:
    #       Include chain1_output['summary'] as context (KEY CHAINING pattern)
    #       Request 3-4 sentences about gene function, disease mechanism, inheritance

    # TODO: Step 4 - Return dict with all chain1_output fields plus new gene fields
    pass


def literature_collector_chain(chain2_output: Dict[str, Any], llm_client: OpenAI) -> Dict[str, Any]:
    """
    Chain 3: Collect and summarize supporting literature evidence.

    Returns: {**chain2_output, literature_evidence: [{pmid, finding_summary,
              study_type, evidence_strength}], evidence_synthesis}
    The "evidence_synthesis" field feeds Chain 4.
    """
    # TODO: Step 1 - Build PubMed search queries using gene and variant

    # TODO: Step 2 - Query PubMed and deduplicate by PMID

    # TODO: Step 3 - Use LLM to classify each article:
    #       finding_summary, study_type (Functional/Clinical/Population/etc),
    #       evidence_strength (Strong/Moderate/Weak)

    # TODO: Step 4 - Use LLM to synthesize overall evidence:
    #       Include chain2_output['summary'] and ['gene_disease_context']
    #       Request 3-4 sentence synthesis

    # TODO: Step 5 - Return dict with all chain2_output fields plus evidence fields
    pass


def acmg_classifier_chain(chain3_output: Dict[str, Any], llm_client: OpenAI) -> Dict[str, Any]:
    """
    Chain 4: Apply ACMG criteria and generate final classification.

    ACMG Criteria: PVS1 (null variant), PS1-PS4 (strong), PM1-PM6 (moderate), PP1-PP5 (supporting)
    Classification: Pathogenic/Likely Pathogenic/VUS/Likely Benign/Benign

    Returns: {hgvs, gene, final_classification, acmg_criteria_applied,
              criteria_justification, confidence, clinical_summary, limitations}
    """
    # Build context from ALL previous chains
    context = f"""
=== CLINVAR DATA (Chain 1) ===
{chain3_output.get('summary', 'Not available')}

=== GENE-DISEASE CONTEXT (Chain 2) ===
{chain3_output.get('gene_disease_context', 'Not available')}

=== LITERATURE EVIDENCE (Chain 3) ===
{chain3_output.get('evidence_synthesis', 'Not available')}

=== STRUCTURED DATA ===
Variant: {chain3_output.get('hgvs')}
Gene: {chain3_output.get('gene')}
ClinVar: {chain3_output.get('clinvar_significance')} ({chain3_output.get('review_status')})
Conditions: {', '.join(chain3_output.get('conditions', []))}
Inheritance: {', '.join(chain3_output.get('inheritance_patterns', []))}
Literature: {len(chain3_output.get('literature_evidence', []))} articles
"""

    # TODO: Step 1 - Create system prompt for ACMG classification
    #       Include criteria definitions and classification rules
    #       Request JSON: final_classification, acmg_criteria_applied,
    #       criteria_justification, confidence, clinical_summary, limitations

    # TODO: Step 2 - Create user prompt with the accumulated context

    # TODO: Step 3 - Call the LLM with response_format={"type": "json_object"}

    # TODO: Step 4 - Parse JSON response; handle errors with fallback values

    # TODO: Step 5 - Return the final classification result dict
    pass


def variant_to_diagnosis_chain(hgvs: str, llm_client: OpenAI) -> Dict[str, Any]:
    """Execute the full four-chain classification workflow."""
    print(f"\n{'='*60}\nVARIANT CHAIN: {hgvs}\n{'='*60}")

    # Chain 1: ClinVar
    print(f"\n[Chain 1 - ClinVar Fetcher]")
    chain1 = clinvar_fetcher_chain(hgvs, llm_client)
    if chain1 is None:
        chain1 = {"hgvs": hgvs, "gene": None, "summary": "NOT IMPLEMENTED",
                  "clinvar_significance": "N/A", "review_status": "N/A",
                  "conditions": [], "supporting_pmids": [], "allele_frequencies": None}
    print(f"  {chain1.get('summary', 'N/A')[:100]}...")

    # Chain 2: Gene Enrichment
    print(f"\n[Chain 2 - Gene Enricher]")
    chain2 = gene_enricher_chain(chain1, llm_client)
    if chain2 is None:
        chain2 = {**chain1, "gene_disease_context": "NOT IMPLEMENTED",
                  "gene_phenotypes": [], "inheritance_patterns": [],
                  "gene_summary": "N/A", "gene_function": "N/A"}
    print(f"  {chain2.get('gene_disease_context', 'N/A')[:100]}...")

    # Chain 3: Literature
    print(f"\n[Chain 3 - Literature Collector]")
    chain3 = literature_collector_chain(chain2, llm_client)
    if chain3 is None:
        chain3 = {**chain2, "evidence_synthesis": "NOT IMPLEMENTED", "literature_evidence": []}
    print(f"  {chain3.get('evidence_synthesis', 'N/A')[:100]}...")

    # Chain 4: Classification
    print(f"\n[Chain 4 - ACMG Classifier]")
    chain4 = acmg_classifier_chain(chain3, llm_client)
    if chain4 is None:
        chain4 = {"hgvs": hgvs, "gene": chain3.get("gene"), "final_classification": "NOT IMPLEMENTED",
                  "acmg_criteria_applied": [], "criteria_justification": {},
                  "confidence": "N/A", "clinical_summary": "TODO", "limitations": ["Not implemented"]}
    print(f"  -> {chain4.get('final_classification', 'N/A')} ({chain4.get('confidence', 'N/A')})")
    print(f"  Criteria: {', '.join(chain4.get('acmg_criteria_applied', []))}")

    return chain4


def main():
    """Main entry point."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

    print("Module 5: Variant-to-Diagnosis Chain")
    print("Chain 1 (ClinVar) → Chain 2 (Gene) → Chain 3 (Literature) → Chain 4 (ACMG)\n")

    results = [variant_to_diagnosis_chain(v, client) for v in VARIANTS_TO_CLASSIFY]

    # Summary table
    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    print(f"{'Variant':<28} {'Gene':<8} {'Classification':<18} {'Confidence'}")
    print("-"*80)
    for r in results:
        gene = r.get("gene") or "N/A"
        hgvs = r.get("hgvs") or "N/A"
        classification = r.get("final_classification") or "N/A"
        confidence = r.get("confidence") or "N/A"

        print(f"{hgvs:<30} {gene:<8} {classification:<18} {criteria:<25} {confidence}")

if __name__ == "__main__":
    main()
