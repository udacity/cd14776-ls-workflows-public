"""
Module 5 Exercise: Variant-to-Diagnosis Chain

A four-chain workflow that classifies variants of uncertain significance:
Chain 1 (ClinVar Fetcher) → Chain 2 (Gene Enricher) → Chain 3 (Literature Collector) → Chain 4 (ACMG Classifier)

Each chain produces a natural language summary that becomes context for the next chain,
while maintaining structured data for programmatic use.
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


# Input variants for classification
VARIANTS_TO_CLASSIFY = [
    "NM_000546.6:c.743G>A",   # TP53 - well-characterized pathogenic
    "NM_000059.4:c.5946delT", # BRCA2 - frameshift
]


def clinvar_fetcher_chain(hgvs: str, llm_client: OpenAI) -> Dict[str, Any]:
    """
    Chain 1: Fetch and summarize ClinVar data for a variant.

    Args:
        hgvs: HGVS variant notation
        llm_client: OpenAI client

    Returns:
    {
        "hgvs": str,
        "gene": str,
        "clinvar_significance": str,
        "review_status": str,
        "conditions": List[str],
        "supporting_pmids": List[str],
        "allele_frequencies": Dict[str, float] | None,
        "summary": str  # Natural language summary for Chain 2 input
    }
    """
    # 1. Query ClinVar
    clinvar_data = query_clinvar(hgvs)

    # 2. Handle not found
    if "error" in clinvar_data:
        return {
            "hgvs": hgvs,
            "gene": None,
            "clinvar_significance": "Not in ClinVar",
            "review_status": None,
            "conditions": [],
            "supporting_pmids": [],
            "allele_frequencies": None,
            "summary": f"The variant {hgvs} was not found in ClinVar database."
        }

    # Extract structured data
    gene = clinvar_data.get("gene")
    significance = clinvar_data.get("clinical_significance", "Unknown")
    review_status = clinvar_data.get("review_status", "Unknown")
    conditions = clinvar_data.get("conditions", [])
    pmids = clinvar_data.get("pubmed_pmids", [])
    allele_freq = clinvar_data.get("allele_frequencies")

    # 3. Use LLM to generate summary from structured data
    system_prompt = """You are a clinical geneticist summarizing ClinVar variant data.
Generate a concise 2-3 sentence summary that:
- States the current ClinVar classification
- Notes the review status and confidence level
- Lists key associated conditions
- Mentions number of supporting publications

Be precise and clinical in tone."""

    user_prompt = f"""Generate a summary for this ClinVar variant data:

Variant: {hgvs}
Gene: {gene}
Clinical Significance: {significance}
Review Status: {review_status}
Associated Conditions: {', '.join(conditions) if conditions else 'None listed'}
Number of Supporting Publications: {len(pmids)}
Allele Frequencies: {json.dumps(allele_freq) if allele_freq else 'Not available'}"""

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    summary = response.choices[0].message.content.strip()

    return {
        "hgvs": hgvs,
        "gene": gene,
        "clinvar_significance": significance,
        "review_status": review_status,
        "conditions": conditions,
        "supporting_pmids": pmids,
        "allele_frequencies": allele_freq,
        "summary": summary
    }


def gene_enricher_chain(chain1_output: Dict[str, Any], llm_client: OpenAI) -> Dict[str, Any]:
    """
    Chain 2: Enrich variant data with gene context from NCBI Gene.

    Args:
        chain1_output: Complete output from Chain 1
        llm_client: OpenAI client

    Returns:
    {
        **chain1_output,
        "gene_phenotypes": List[str],
        "inheritance_patterns": List[str],
        "gene_summary": str,
        "gene_function": str,
        "gene_disease_context": str  # Natural language enrichment for Chain 3
    }
    """
    # 1. Extract gene from Chain 1
    gene = chain1_output.get("gene")
    if not gene:
        return {
            **chain1_output,
            "gene_phenotypes": [],
            "inheritance_patterns": [],
            "gene_summary": "Unknown",
            "gene_function": "Unknown",
            "gene_disease_context": "Gene information not available for enrichment."
        }

    # 2. Query NCBI Gene
    gene_data = query_ncbi_gene(gene)

    # Extract gene information
    gene_phenotypes = []
    inheritance_patterns = []
    gene_summary = "Unknown"
    gene_function = "Unknown"
    gene_aliases = []

    if gene_data.get("results"):
        result = gene_data["results"][0]

        # Get gene summary (very informative from NCBI)
        gene_summary = result.get("summary", "Unknown") or "Unknown"

        # Get gene function from the name/description
        gene_function = result.get("name", "Unknown") or "Unknown"

        # Get aliases
        gene_aliases = result.get("aliases", [])

        # Extract phenotypes
        for pheno in result.get("phenotypes", []):
            if pheno.get("phenotype"):
                gene_phenotypes.append(pheno["phenotype"])

    # 3. Use LLM to synthesize gene-disease context
    system_prompt = """You are a clinical geneticist synthesizing gene-disease relationships.
Generate a 3-4 sentence context summary that:
- Describes the gene's normal function
- Explains the disease mechanism
- Notes inheritance patterns if known
- Connects to the variant's potential impact

Build on the ClinVar summary context provided. Be precise and clinically relevant."""

    user_prompt = f"""Generate gene-disease context for this variant:

=== CLINVAR CONTEXT (Chain 1) ===
{chain1_output['summary']}

=== NCBI GENE DATA ===
Gene: {gene}
Full Name: {gene_function}
Aliases: {', '.join(gene_aliases[:5]) if gene_aliases else 'None'}
Gene Summary: {gene_summary[:800] if gene_summary != "Unknown" else 'Not available'}
Associated Phenotypes: {', '.join(gene_phenotypes) if gene_phenotypes else 'None listed'}
Variant: {chain1_output['hgvs']}"""

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    gene_disease_context = response.choices[0].message.content.strip()

    return {
        **chain1_output,
        "gene_phenotypes": gene_phenotypes,
        "inheritance_patterns": inheritance_patterns,
        "gene_summary": gene_summary,
        "gene_function": gene_function,
        "gene_disease_context": gene_disease_context
    }


def literature_collector_chain(chain2_output: Dict[str, Any], llm_client: OpenAI) -> Dict[str, Any]:
    """
    Chain 3: Collect and summarize supporting literature evidence.

    Args:
        chain2_output: Complete output from Chain 2
        llm_client: OpenAI client

    Returns:
    {
        **chain2_output,
        "literature_evidence": List[{
            "pmid": str,
            "finding_summary": str,
            "study_type": str,
            "evidence_strength": str
        }],
        "evidence_synthesis": str  # Natural language synthesis for Chain 4
    }
    """
    # 1. Build search queries from previous context
    gene = chain2_output.get("gene", "")
    hgvs = chain2_output.get("hgvs", "")
    conditions = chain2_output.get("conditions", [])

    # Extract variant notation for search
    variant_notation = hgvs.split(':')[-1] if ':' in hgvs else hgvs

    # Build search queries
    search_queries = [
        f"{gene} {variant_notation} functional",
        f"{gene} pathogenic variant clinical",
    ]

    # 2. Query PubMed
    all_articles = []
    seen_pmids = set()

    for query in search_queries:
        articles = query_pubmed(query, max_results=10)
        for article in articles:
            pmid = article.get("pmid")
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                all_articles.append(article)

    # Limit to top 15 articles
    all_articles = all_articles[:15]

    # 3. Use LLM to classify and summarize each article
    literature_evidence = []

    if all_articles:
        system_prompt = """You are a clinical geneticist analyzing research articles for variant classification.
For each article, provide:
1. A 1-2 sentence summary of the key finding relevant to the variant
2. The study type (Functional, Clinical, Population, Computational, Case Report, Review)
3. Evidence strength (Strong, Moderate, Weak)

Respond in JSON format:
{
    "finding_summary": "...",
    "study_type": "...",
    "evidence_strength": "..."
}"""

        for article in all_articles[:10]:  # Limit LLM calls
            title = article.get("title", "")
            abstract = article.get("abstract", "")

            if not abstract:
                continue

            user_prompt = f"""Analyze this article for relevance to {gene} variant classification:

Title: {title}

Abstract: {abstract[:1500]}

Focus on evidence relevant to pathogenicity assessment."""

            try:
                response = llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )

                result_text = response.choices[0].message.content.strip()

                # Parse JSON response
                if result_text.startswith("```"):
                    result_text = result_text.split("```")[1]
                    if result_text.startswith("json"):
                        result_text = result_text[4:]

                article_analysis = json.loads(result_text)

                literature_evidence.append({
                    "pmid": article.get("pmid"),
                    "title": title,
                    "finding_summary": article_analysis.get("finding_summary", ""),
                    "study_type": article_analysis.get("study_type", "Unknown"),
                    "evidence_strength": article_analysis.get("evidence_strength", "Unknown")
                })
            except (json.JSONDecodeError, Exception) as e:
                # If parsing fails, add basic info
                literature_evidence.append({
                    "pmid": article.get("pmid"),
                    "title": title,
                    "finding_summary": f"Article discusses {gene} variants.",
                    "study_type": "Unknown",
                    "evidence_strength": "Unknown"
                })

    # 4. Use LLM to synthesize overall evidence
    system_prompt = """You are a clinical geneticist synthesizing literature evidence for ACMG variant classification.
Generate a 3-4 sentence evidence synthesis that:
- Summarizes the types of evidence found
- Highlights the strongest supporting studies
- Notes any conflicting evidence
- Assesses overall literature support level

Build on the previous chain summaries provided."""

    # Count study types
    study_type_counts = {}
    strong_evidence = []
    for ev in literature_evidence:
        st = ev.get("study_type", "Unknown")
        study_type_counts[st] = study_type_counts.get(st, 0) + 1
        if ev.get("evidence_strength") == "Strong":
            strong_evidence.append(ev.get("finding_summary", ""))

    user_prompt = f"""Synthesize the literature evidence for this variant:

=== CLINVAR CONTEXT (Chain 1) ===
{chain2_output['summary']}

=== GENE-DISEASE CONTEXT (Chain 2) ===
{chain2_output['gene_disease_context']}

=== LITERATURE EVIDENCE ===
Total articles found: {len(literature_evidence)}
Study type distribution: {json.dumps(study_type_counts)}
Strong evidence findings: {json.dumps(strong_evidence[:5]) if strong_evidence else 'None identified'}

Individual article summaries:
{json.dumps([{"pmid": e["pmid"], "summary": e["finding_summary"], "type": e["study_type"], "strength": e["evidence_strength"]} for e in literature_evidence[:8]], indent=2)}"""

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    evidence_synthesis = response.choices[0].message.content.strip()

    return {
        **chain2_output,
        "literature_evidence": literature_evidence,
        "evidence_synthesis": evidence_synthesis
    }


def acmg_classifier_chain(chain3_output: Dict[str, Any], llm_client: OpenAI) -> Dict[str, Any]:
    """
    Chain 4: Apply ACMG criteria and generate final classification.

    Args:
        chain3_output: Complete output from Chain 3
        llm_client: OpenAI client

    Returns:
    {
        "hgvs": str,
        "gene": str,
        "final_classification": str,
        "acmg_criteria_applied": List[str],
        "criteria_justification": Dict[str, str],
        "confidence": str,
        "clinical_summary": str,
        "limitations": List[str]
    }
    """
    system_prompt = """You are an expert clinical geneticist applying ACMG/AMP criteria for variant classification.

You have been provided with:
1. ClinVar data and classification
2. Gene-disease context from NCBI Gene
3. Literature evidence synthesis

Your task is to:
1. Identify which ACMG criteria apply based on the evidence
2. Justify each criterion with specific evidence
3. Combine criteria to reach final classification
4. Assess confidence level
5. Write a clinical summary suitable for a genetic report

ACMG criteria to consider:

Very Strong (PVS1):
- Null variant in gene where LOF is known mechanism

Strong (PS1-PS4):
- PS1: Same amino acid change as established pathogenic variant
- PS2: De novo (confirmed)
- PS3: Well-established functional studies support damaging effect
- PS4: Prevalence in affected > controls

Moderate (PM1-PM6):
- PM1: Located in mutational hot spot or functional domain
- PM2: Absent from controls (or extremely low frequency)
- PM3: For recessive, detected in trans with pathogenic variant
- PM4: Protein length change
- PM5: Novel missense at residue where different missense is pathogenic
- PM6: Assumed de novo

Supporting (PP1-PP5):
- PP1: Cosegregation with disease
- PP2: Missense in gene with low benign missense rate
- PP3: Computational evidence supports deleterious effect
- PP4: Patient's phenotype highly specific for gene
- PP5: Reputable source reports as pathogenic

Classification rules:
- Pathogenic: (PVS1 AND ≥1 PS) OR (≥2 PS) OR (1 PS AND ≥3 PM) OR (1 PS AND 2 PM AND ≥2 PP)
- Likely Pathogenic: (1 PVS AND 1 PM) OR (1 PS AND 1-2 PM) OR (1 PS AND ≥2 PP) OR (≥3 PM)
- Uncertain Significance: Criteria don't meet above thresholds
- Likely Benign / Benign: Benign criteria apply

Be rigorous: only apply criteria when clearly supported by evidence.

Respond in JSON format:
{
    "final_classification": "Pathogenic|Likely Pathogenic|VUS|Likely Benign|Benign",
    "acmg_criteria_applied": ["PVS1", "PS3", ...],
    "criteria_justification": {"PVS1": "explanation...", "PS3": "explanation..."},
    "confidence": "High|Medium|Low",
    "clinical_summary": "Detailed clinical summary paragraph...",
    "limitations": ["limitation 1", "limitation 2"]
}"""

    # Build comprehensive context from all previous chains
    context = f"""
=== CLINVAR DATA (Chain 1) ===
{chain3_output['summary']}

=== GENE-DISEASE CONTEXT (Chain 2) ===
{chain3_output['gene_disease_context']}

=== LITERATURE EVIDENCE (Chain 3) ===
{chain3_output['evidence_synthesis']}

=== STRUCTURED DATA ===
Variant: {chain3_output.get('hgvs')}
Gene: {chain3_output.get('gene')}
ClinVar Classification: {chain3_output.get('clinvar_significance')}
Review Status: {chain3_output.get('review_status')}
Conditions: {', '.join(chain3_output.get('conditions', []))}
Inheritance Patterns: {', '.join(chain3_output.get('inheritance_patterns', []))}
Gene Function: {chain3_output.get('gene_function')}
Population Frequency: {json.dumps(chain3_output.get('allele_frequencies')) if chain3_output.get('allele_frequencies') else 'Not available'}
Number of Literature Evidence: {len(chain3_output.get('literature_evidence', []))}
Strong Evidence Studies: {len([e for e in chain3_output.get('literature_evidence', []) if e.get('evidence_strength') == 'Strong'])}
"""

    user_prompt = f"""Apply ACMG criteria to classify this variant:

{context}

Provide your classification analysis in JSON format."""

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    result_text = response.choices[0].message.content.strip()

    # Parse JSON response
    try:
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]

        classification_result = json.loads(result_text)
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        classification_result = {
            "final_classification": "VUS",
            "acmg_criteria_applied": [],
            "criteria_justification": {},
            "confidence": "Low",
            "clinical_summary": "Classification could not be completed due to parsing error.",
            "limitations": ["LLM response parsing failed"]
        }

    return {
        "hgvs": chain3_output.get("hgvs"),
        "gene": chain3_output.get("gene"),
        "final_classification": classification_result.get("final_classification", "VUS"),
        "acmg_criteria_applied": classification_result.get("acmg_criteria_applied", []),
        "criteria_justification": classification_result.get("criteria_justification", {}),
        "confidence": classification_result.get("confidence", "Low"),
        "clinical_summary": classification_result.get("clinical_summary", ""),
        "limitations": classification_result.get("limitations", [])
    }


def variant_to_diagnosis_chain(hgvs: str, llm_client: OpenAI) -> Dict[str, Any]:
    """
    Execute the full four-chain classification workflow.

    Shows how each chain's natural language summary feeds the next chain.
    """
    print(f"\n{'='*70}")
    print(f"VARIANT CLASSIFICATION CHAIN")
    print(f"Variant: {hgvs}")
    print(f"{'='*70}")

    # Chain 1: ClinVar
    print(f"\n[Chain 1 - ClinVar Fetcher]")
    print("-" * 40)
    chain1 = clinvar_fetcher_chain(hgvs, llm_client)
    print(f"Summary: {chain1['summary']}")

    # Chain 2: Gene Enrichment
    print(f"\n[Chain 2 - Gene Enricher]")
    print("-" * 40)
    chain2 = gene_enricher_chain(chain1, llm_client)
    print(f"Context: {chain2['gene_disease_context']}")

    # Chain 3: Literature
    print(f"\n[Chain 3 - Literature Collector]")
    print("-" * 40)
    chain3 = literature_collector_chain(chain2, llm_client)
    print(f"Synthesis: {chain3['evidence_synthesis']}")

    # Chain 4: Classification
    print(f"\n[Chain 4 - ACMG Classifier]")
    print("-" * 40)
    chain4 = acmg_classifier_chain(chain3, llm_client)
    print(f"Classification: {chain4['final_classification']}")
    print(f"Criteria: {', '.join(chain4['acmg_criteria_applied'])}")
    print(f"Confidence: {chain4['confidence']}")

    print(f"\nCLINICAL SUMMARY:")
    print(chain4['clinical_summary'])

    if chain4.get('limitations'):
        print(f"\nLIMITATIONS:")
        for limitation in chain4['limitations']:
            print(f"  - {limitation}")

    return chain4


def run_all_variants(variants: List[str], llm_client: OpenAI) -> List[Dict[str, Any]]:
    """Process all variants and display summary table."""
    results = []

    for hgvs in variants:
        result = variant_to_diagnosis_chain(hgvs, llm_client)
        results.append(result)

    # Print summary table
    print(f"\n{'='*90}")
    print("CLASSIFICATION SUMMARY")
    print("="*90)
    print(f"{'Variant':<30} {'Gene':<8} {'Classification':<18} {'Criteria':<25} {'Confidence'}")
    print("-"*90)
    for r in results:
        criteria = ', '.join(r.get('acmg_criteria_applied', [])[:4])
        if len(r.get('acmg_criteria_applied', [])) > 4:
            criteria += '...'
        gene = r.get("gene") or "N/A"
        hgvs = r.get("hgvs") or "N/A"
        classification = r.get("final_classification") or "N/A"
        confidence = r.get("confidence") or "N/A"

        print(f"{hgvs:<30} {gene:<8} {classification:<18} {criteria:<25} {confidence}")


    return results


def main():
    """Main entry point."""
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

    print("="*70)
    print("MODULE 5 EXERCISE: VARIANT-TO-DIAGNOSIS CHAIN")
    print("="*70)
    print("\nThis workflow demonstrates prompt chaining where each chain's output")
    print("becomes the input context for the next chain:")
    print("  Chain 1 (ClinVar) → Chain 2 (NCBI Gene) → Chain 3 (Literature) → Chain 4 (ACMG)")
    print()

    # Run classification for all variants
    results = run_all_variants(VARIANTS_TO_CLASSIFY, client)

    # Print detailed criteria justifications
    print(f"\n{'='*90}")
    print("DETAILED CRITERIA JUSTIFICATIONS")
    print("="*90)

    for result in results:
        print(f"\n{result.get('hgvs')} ({result.get('gene')}):")
        print("-"*60)
        justifications = result.get('criteria_justification', {})
        for criterion, justification in justifications.items():
            print(f"  {criterion}: {justification}")

    return results


if __name__ == "__main__":
    main()
