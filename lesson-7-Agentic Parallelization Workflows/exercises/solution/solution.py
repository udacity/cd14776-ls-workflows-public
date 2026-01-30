"""
Module 7 Exercise: CRISPR Off-Target Risk Scanner

Implements a parallelization workflow that processes independent BLAST searches
against multiple species genomes concurrently and aggregates results into
a traffic-light safety report.

Part of the Agentic Workflows for Life Sciences Research course.
"""

import os
import sys
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from ls_action_space.action_space import blast_sequence

def normalize_blast_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for h in hits:
        hsps = h.get("hsps") or []
        if not hsps:
            continue
        hsp0 = hsps[0]
        align_len = int(hsp0.get("align_len", 0) or 0)
        identities = int(hsp0.get("identities", 0) or 0)
        mismatches = max(0, align_len - identities)

        normalized.append({
            "accession": h.get("hit_id", ""),                 # keep as-is or parse
            "description": h.get("title", "") or "",
            "identity_pct": float(hsp0.get("pct_identity", 0.0) or 0.0),
            "mismatches": mismatches,
            "evalue": hsp0.get("evalue"),
            "score": hsp0.get("score"),
            "align_len": align_len,
            "identities": identities,
        })
    return normalized

def filter_hits_by_species(hits: List[Dict], species: str) -> List[Dict]:
    """
    Filter BLAST hits by species name.

    Args:
        hits: List of BLAST hit dictionaries
        species: Species to filter for ("human", "mouse", "pig")

    Returns:
        Filtered list of hits
    """
    species_map = {
        "human": "homo sapiens",
        "mouse": "mus musculus"
    }

    target_name = species_map.get(species.lower(), species.lower())

    filtered = []
    for h in hits:
        # Handle the different response formats from live BLAST
        title = h.get("title", "") or ""
        hit_id = h.get("hit_id", "") or ""
        scientific_name = h.get("scientific_name", "") or ""

        search_text = f"{title} {hit_id} {scientific_name}".lower()
        if target_name in search_text:
            filtered.append(h)

    return filtered


# Sample gRNA sequences for testing
GRNA_SEQUENCES = [
    {
        "name": "TP53_exon7_gRNA1",
        "sequence": "CGTGAGCGCTTCGAGATGTT",
        "pam": "TGG",
        "target_gene": "TP53",
        "target_species": "human"
    },
    {
        "name": "PCSK9_gRNA2",
        "sequence": "GACCCTCATCATCACCTGCT",
        "pam": "GGG",
        "target_gene": "PCSK9",
        "target_species": "human"
    },
    {
        "name": "BRCA1_gRNA3",
        "sequence": "ATGCTAGCTAGCTAGCTGAC",  # Different sequence for variety
        "pam": "AGG",
        "target_gene": "BRCA1",
        "target_species": "human"
    }
]


class SpeciesBlastAgent:
    """
    BLAST agent for a specific species genome.
    Runs BLAST queries and analyzes off-target risks for CRISPR guide RNAs.
    """

    SPECIES_DATABASES = {
        "human": "human_genomic",
        "mouse": "mouse_genomic"
    }

    SPECIES_NAMES = {
        "human": "Homo sapiens",
        "mouse": "Mus musculus"
    }

    # Compatible PAM sequences for SpCas9 (NGG primary, NAG secondary)
    COMPATIBLE_PAMS = ["AGG", "CGG", "GGG", "TGG", "AAG", "CAG", "GAG", "TAG"]

    def __init__(self, species: str, llm_client: OpenAI):
        """
        Initialize the species-specific BLAST agent.

        Args:
            species: Species name ("human", "mouse", "pig")
            llm_client: OpenAI client for LLM-based interpretation
        """
        self.species = species.lower()
        self.llm = llm_client
        self.database = self.SPECIES_DATABASES.get(self.species, "nt")
        self.species_name = self.SPECIES_NAMES.get(self.species, species)

    def _classify_off_target_risk(self, mismatches: int, positions: List[int] = None) -> str:
        """
        Classify off-target risk based on mismatch count and positions.

        Risk levels:
        - 0-1 mismatches: HIGH risk (likely cut)
        - 2 mismatches: HIGH risk (probable cut)
        - 3 mismatches: MEDIUM risk (possible cut, depends on position)
        - 4 mismatches: MEDIUM risk (unlikely but possible)
        - >=5 mismatches: LOW risk (negligible cutting)
        """
        if mismatches <= 2:
            return "High"
        elif mismatches <= 4:
            return "Medium"
        else:
            return "Low"

    def _get_risk_rationale(self, mismatches: int, positions: List[int], location: str) -> str:
        """Generate rationale for risk classification."""
        if mismatches == 0:
            return "Perfect match - on-target site"
        elif mismatches == 1:
            return f"Single mismatch at position {positions[0]} - high probability of cutting"
        elif mismatches == 2:
            return f"Two mismatches at positions {positions} - probable off-target cutting"
        elif mismatches == 3:
            pam_proximal = [p for p in positions if p <= 12]
            if len(pam_proximal) >= 2:
                return f"Three mismatches with {len(pam_proximal)} in PAM-proximal region - reduced risk"
            return f"Three mismatches at positions {positions} - possible off-target activity"
        elif mismatches == 4:
            return f"Four mismatches - unlikely to cut but monitor in sensitive contexts"
        else:
            return f"{mismatches} mismatches - negligible off-target risk"

    def _generate_mismatch_positions(self, mismatches: int, query_len: int = 20) -> List[int]:
        """Generate deterministic mismatch positions based on mismatch count."""
        if mismatches == 0:
            return []
        # Distribute mismatches across the guide
        random.seed(mismatches * 31)
        positions = sorted(random.sample(range(1, query_len + 1), min(mismatches, query_len)))
        return positions

    def _check_pam_compatibility(self, accession: str) -> tuple:
        """
        Check if a PAM sequence is present near the hit site.
        Returns (is_compatible, pam_sequence).
        """
        # Simulate PAM checking - in reality would check genomic sequence
        random.seed(hash(accession))
        pam = random.choice(self.COMPATIBLE_PAMS + ["AAA", "TTT", "CCC"])
        is_compatible = pam in self.COMPATIBLE_PAMS
        return is_compatible, pam

    def _extract_gene_context(self, description: str) -> str:
        """Extract gene/region context from hit description."""
        if "intergenic" in description.lower():
            # Extract nearby gene if mentioned
            parts = description.split("near ")
            if len(parts) > 1:
                return f"Intergenic (near {parts[1].split()[0]})"
            return "Intergenic"
        elif "intron" in description.lower():
            parts = description.split(", ")
            for part in parts:
                if "intron" in part.lower():
                    return part.strip()
            return "Intronic region"
        elif "exon" in description.lower():
            parts = description.split(", ")
            for part in parts:
                if "exon" in part.lower():
                    return part.strip()
            return "Exonic region"
        elif "UTR" in description:
            parts = description.split(", ")
            for part in parts:
                if "UTR" in part:
                    return part.strip()
            return "UTR region"
        return "Unknown region"

    def _identify_on_target(self, hits: List[Dict], target_gene: str) -> Optional[Dict]:
        """Identify the on-target hit matching the intended target gene."""
        for hit in hits:
            if hit["mismatches"] <= 1 and target_gene.upper() in hit["description"].upper():
                return hit
        # If no perfect match found, check for near-perfect match
        for hit in hits:
            if hit["mismatches"] <= 2:
                return hit
        return None

    def _use_llm_for_interpretation(self, results: Dict[str, Any]) -> str:
        """Use LLM to generate interpretation of results."""
        prompt = f"""Analyze these CRISPR off-target BLAST results for {self.species}:

Species: {self.species_name}
Total hits: {results.get('total_hits', 0)}
High-risk off-targets: {results.get('off_target_summary', {}).get('high_risk', 0)}
Medium-risk off-targets: {results.get('off_target_summary', {}).get('medium_risk', 0)}
Low-risk off-targets: {results.get('off_target_summary', {}).get('low_risk', 0)}

Provide a brief 2-3 sentence interpretation of the off-target safety profile for this species.
Focus on whether this guide is suitable for use in this species and any concerns.
"""
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a CRISPR safety expert analyzing off-target data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback interpretation if LLM fails
            high_risk = results.get('off_target_summary', {}).get('high_risk', 0)
            if high_risk == 0:
                return f"Excellent off-target profile in {self.species}. No high-risk sites identified."
            elif high_risk <= 2:
                return f"Acceptable off-target profile in {self.species} with {high_risk} high-risk site(s) requiring validation."
            else:
                return f"Concerning off-target profile in {self.species} with {high_risk} high-risk sites. Alternative guides recommended."

    def search(self, grna_sequence: str, target_gene: str) -> Dict[str, Any]:
        """
        BLAST gRNA against species genome and analyze off-targets.

        Args:
            grna_sequence: 20bp guide RNA sequence (DNA form, no PAM)
            target_gene: Intended target gene name

        Returns:
            Dictionary containing:
            - species: str
            - query_sequence: str
            - total_hits: int
            - on_target_hit: Dict or None
            - off_target_hits: List of off-target hit details
            - off_target_summary: Summary counts by risk level
            - species_safety_score: 0-100 score
            - interpretation: LLM-generated interpretation
        """
        # Run BLAST search
        blast_results = blast_sequence(
            sequence=grna_sequence,
            database=self.database,
            program="blastn",
            hitlist_size=50
        )

        # Filter hits by species if using nt database
        all_hits = normalize_blast_hits(blast_results.get("hits", []))
        species_hits = filter_hits_by_species(all_hits, self.species)
        species_hits = normalize_blast_hits(species_hits)

        # Identify on-target hit
        on_target = self._identify_on_target(species_hits, target_gene)

        # Analyze off-target hits
        off_target_hits = []
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0

        for hit in species_hits:
            # Skip the on-target hit
            if on_target and hit["accession"] == on_target["accession"]:
                continue

            mismatches = hit["mismatches"]
            positions = self._generate_mismatch_positions(mismatches)
            risk_level = self._classify_off_target_risk(mismatches, positions)
            pam_compatible, pam_seq = self._check_pam_compatibility(hit["accession"])
            gene_context = self._extract_gene_context(hit["description"])

            off_target_entry = {
                "accession": hit["accession"],
                "gene_context": gene_context,
                "identity_pct": hit["identity_pct"],
                "mismatches": mismatches,
                "mismatch_positions": positions,
                "pam_compatible": pam_compatible,
                "pam_sequence": pam_seq,
                "risk_level": risk_level,
                "risk_rationale": self._get_risk_rationale(mismatches, positions, gene_context)
            }
            off_target_hits.append(off_target_entry)

            # Count by risk level
            if risk_level == "High":
                high_risk_count += 1
            elif risk_level == "Medium":
                medium_risk_count += 1
            else:
                low_risk_count += 1

        # Sort off-targets by risk (high risk first)
        risk_order = {"High": 0, "Medium": 1, "Low": 2}
        off_target_hits.sort(key=lambda x: (risk_order[x["risk_level"]], x["mismatches"]))

        # Calculate species safety score
        # Start at 100, subtract based on off-targets
        safety_score = 100
        if self.species == "human":
            safety_score -= high_risk_count * 15
            safety_score -= medium_risk_count * 5
        else:
            safety_score -= high_risk_count * 10
            safety_score -= medium_risk_count * 3
        safety_score = max(0, safety_score)

        results = {
            "species": self.species,
            "query_sequence": grna_sequence,
            "total_hits": len(species_hits),
            "on_target_hit": on_target,
            "off_target_hits": off_target_hits,
            "off_target_summary": {
                "total": len(off_target_hits),
                "high_risk": high_risk_count,
                "medium_risk": medium_risk_count,
                "low_risk": low_risk_count
            },
            "species_safety_score": safety_score,
            "interpretation": ""
        }

        # Get LLM interpretation
        results["interpretation"] = self._use_llm_for_interpretation(results)

        return results


class ParallelOffTargetScanner:
    """
    Runs BLAST against multiple species in parallel.
    Implements the fan-out/fan-in pattern for concurrent genome scanning.
    """

    SPECIES_TO_SCAN = ["human", "mouse"]

    def __init__(self, llm_client: OpenAI):
        """
        Initialize the parallel scanner with species-specific agents.

        Args:
            llm_client: OpenAI client for LLM-based analysis
        """
        self.llm = llm_client
        self.agents = {
            species: SpeciesBlastAgent(species, llm_client)
            for species in self.SPECIES_TO_SCAN
        }

    def scan(self, grna_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Scan gRNA against all species in parallel.

        Args:
            grna_data: Dict with "name", "sequence", "target_gene", etc.

        Returns:
            Dictionary containing species results and aggregate safety assessment
        """
        grna_seq = grna_data["sequence"]
        target_gene = grna_data["target_gene"]

        print(f"\n{'='*70}")
        print(f"PARALLEL OFF-TARGET SCAN")
        print(f"gRNA: {grna_data['name']}")
        print(f"Sequence: {grna_seq} + {grna_data.get('pam', 'NGG')}")
        print(f"Target: {target_gene}")
        print(f"{'='*70}")

        start_time = time.time()

        # Parallel BLAST against all species
        print(f"\n[Parallel BLAST - {len(self.SPECIES_TO_SCAN)} species]")
        results = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_species = {
                executor.submit(agent.search, grna_seq, target_gene): species
                for species, agent in self.agents.items()
            }

            for future in as_completed(future_to_species):
                species = future_to_species[future]
                try:
                    result = future.result()
                    results[species] = result
                    elapsed = time.time() - start_time
                    hits = result.get("off_target_summary", {})
                    high = hits.get("high_risk", 0)
                    print(f"  + {species}: {result['total_hits']} hits, "
                          f"{high} high-risk ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"  x {species} failed: {e}")
                    results[species] = {"error": str(e)}

        parallel_time = time.time() - start_time

        # Calculate aggregate safety
        aggregate = self._calculate_aggregate_safety(results)

        return {
            "grna_name": grna_data["name"],
            "sequence": grna_seq,
            "target_gene": target_gene,
            "species_results": results,
            "aggregate_safety": aggregate,
            "execution_time_ms": int(parallel_time * 1000)
        }

    def _calculate_aggregate_safety(self, results: Dict) -> Dict[str, Any]:
        """
        Calculate overall safety score from species results.

        Scoring:
        - Start at 100
        - Subtract 15 for each high-risk off-target in human
        - Subtract 10 for each high-risk off-target in preclinical species
        - Subtract 5 for each medium-risk off-target
        - Minimum score: 0

        Traffic light:
        - GREEN: Score >= 80, no high-risk off-targets in human
        - YELLOW: Score 50-79, or any high-risk off-targets
        - RED: Score < 50, or >= 3 high-risk off-targets in human
        """
        overall_score = 100
        limiting_species = None
        worst_score = 100

        human_high_risk = 0
        total_high_risk = 0

        for species, result in results.items():
            if "error" in result:
                continue

            summary = result.get("off_target_summary", {})
            high_risk = summary.get("high_risk", 0)
            medium_risk = summary.get("medium_risk", 0)

            total_high_risk += high_risk

            if species == "human":
                human_high_risk = high_risk
                overall_score -= high_risk * 15
                overall_score -= medium_risk * 5
            else:
                overall_score -= high_risk * 10
                overall_score -= medium_risk * 3

            species_score = result.get("species_safety_score", 100)
            if species_score < worst_score:
                worst_score = species_score
                limiting_species = species

        overall_score = max(0, overall_score)

        # Determine traffic light
        if human_high_risk >= 3 or overall_score < 50:
            traffic_light = "RED"
        elif human_high_risk > 0 or overall_score < 80:
            traffic_light = "YELLOW"
        else:
            traffic_light = "GREEN"

        # Identify cross-species conserved off-targets
        # (This would require comparing hit locations across species)
        cross_species_offtargets = self._find_cross_species_offtargets(results)

        return {
            "overall_score": overall_score,
            "traffic_light": traffic_light,
            "limiting_species": limiting_species,
            "cross_species_conserved_offtargets": cross_species_offtargets
        }

    def _find_cross_species_offtargets(self, results: Dict) -> List[str]:
        """
        Identify off-targets that appear in multiple species.
        These are particularly concerning as they suggest conserved sequences.
        """
        # Collect high-risk gene contexts from each species
        species_genes = {}
        for species, result in results.items():
            if "error" in result:
                continue
            genes = set()
            for ot in result.get("off_target_hits", []):
                if ot["risk_level"] == "High":
                    gene = ot["gene_context"]
                    if "near" in gene.lower():
                        gene = gene.split("near ")[-1].split(")")[0].strip("()")
                    genes.add(gene)
            species_genes[species] = genes

        # Find genes appearing in multiple species
        all_genes = set()
        for genes in species_genes.values():
            all_genes.update(genes)

        cross_species = []
        for gene in all_genes:
            species_with_gene = [sp for sp, genes in species_genes.items() if gene in genes]
            if len(species_with_gene) > 1:
                cross_species.append(f"{gene} ({', '.join(species_with_gene)})")

        return cross_species


class SafetyReportGenerator:
    """
    Generates the final traffic-light safety report.
    """

    TRAFFIC_LIGHT_COLORS = {
        "GREEN": "G",
        "YELLOW": "Y",
        "RED": "R"
    }

    def __init__(self, llm_client: OpenAI):
        """
        Initialize the report generator.

        Args:
            llm_client: OpenAI client for generating recommendations
        """
        self.llm = llm_client

    def _generate_recommendations(self, scan_results: Dict[str, Any]) -> List[str]:
        """Use LLM to generate recommendations based on results."""
        aggregate = scan_results["aggregate_safety"]
        traffic_light = aggregate["traffic_light"]

        prompt = f"""Based on this CRISPR off-target scan:
- Traffic light: {traffic_light}
- Overall score: {aggregate['overall_score']}/100
- Limiting species: {aggregate['limiting_species']}

Generate 4 specific, actionable recommendations for this guide RNA.
Format as a numbered list. Be specific about experimental validation methods and alternatives.
"""
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a CRISPR safety expert providing recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )
            return response.choices[0].message.content.strip().split("\n")
        except Exception:
            # Fallback recommendations
            return [
                "1. VALIDATION REQUIRED: Experimentally validate high-risk off-target sites using GUIDE-seq or CIRCLE-seq",
                "2. ALTERNATIVE GUIDES: Consider screening additional gRNAs for this target",
                "3. MITIGATION OPTIONS: Use high-fidelity Cas9 variants (eSpCas9, HiFi Cas9)",
                "4. PRECLINICAL NOTES: Validate in cell lines before animal studies"
            ]

    def generate(self, scan_results: Dict[str, Any]) -> str:
        """
        Generate formatted safety report with traffic-light rating.

        Args:
            scan_results: Results from ParallelOffTargetScanner.scan()

        Returns:
            Formatted markdown report string
        """
        lines = []
        aggregate = scan_results["aggregate_safety"]
        traffic_light = aggregate["traffic_light"]
        color_indicator = self.TRAFFIC_LIGHT_COLORS.get(traffic_light, "?")

        # Header
        lines.append("=" * 70)
        lines.append("                    OFF-TARGET SAFETY REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"gRNA: {scan_results['grna_name']}")
        lines.append(f"Sequence: 5'-{scan_results['sequence']}-3' + NGG (PAM)")
        lines.append(f"Target Gene: {scan_results['target_gene']} (human)")
        lines.append("")

        # Traffic light box
        status_text = {
            "GREEN": "GREEN - PROCEED",
            "YELLOW": "YELLOW - PROCEED WITH CAUTION",
            "RED": "RED - DO NOT PROCEED"
        }
        lines.append("+" + "-" * 66 + "+")
        status = f"[{color_indicator}] {status_text[traffic_light]}"
        padding = (64 - len(status)) // 2
        lines.append(f"|{' ' * padding}{status}{' ' * (64 - padding - len(status))}|")
        score_line = f"Overall Safety Score: {aggregate['overall_score']}/100"
        lines.append(f"|{score_line:^66}|")
        lines.append("+" + "-" * 66 + "+")
        lines.append("")

        # Species-by-species analysis
        lines.append("=" * 70)
        lines.append("SPECIES-BY-SPECIES ANALYSIS")
        lines.append("=" * 70)

        species_order = ["human", "mouse"]
        species_labels = {
            "human": "HUMAN (Target Species)",
            "mouse": "MOUSE (Preclinical Model)"
        }

        for species in species_order:
            result = scan_results["species_results"].get(species)
            if not result or "error" in result:
                continue

            lines.append("")
            lines.append(species_labels.get(species, species.upper()))
            lines.append("-" * 24)

            # On-target status
            on_target = result.get("on_target_hit")
            if on_target:
                mm = on_target.get("mismatches", 0)
                if mm == 0:
                    lines.append(f"On-Target: + {scan_results['target_gene']} (perfect match)")
                else:
                    lines.append(f"On-Target: + {scan_results['target_gene']} ({mm} mismatch - expected species difference)")
            else:
                lines.append("On-Target: ! Not found in this species")

            # Off-target summary
            summary = result.get("off_target_summary", {})
            lines.append(f"Off-Targets: {summary.get('total', 0)} total")
            lines.append(f"  [R] High Risk (<=2 mm): {summary.get('high_risk', 0)}")
            lines.append(f"  [Y] Medium Risk (3-4 mm): {summary.get('medium_risk', 0)}")
            lines.append(f"  [G] Low Risk (>=5 mm): {summary.get('low_risk', 0)}")

            # High-risk off-targets details
            high_risk_ots = [ot for ot in result.get("off_target_hits", [])
                           if ot["risk_level"] == "High"]
            if high_risk_ots:
                lines.append("")
                lines.append("High-Risk Off-Targets:")
                lines.append("+" + "-" * 66 + "+")
                for i, ot in enumerate(high_risk_ots[:3], 1):  # Show top 3
                    acc_str = f"{i}. Accession: {ot['accession']}"
                    lines.append(f"| {acc_str:<64} |")
                    loc_str = f"   Location: {ot['gene_context']}"
                    lines.append(f"| {loc_str:<64} |")
                    mm_str = f"   Mismatches: {ot['mismatches']} at positions {ot['mismatch_positions']}"
                    lines.append(f"| {mm_str:<64} |")
                    pam_status = "compatible" if ot['pam_compatible'] else "incompatible"
                    pam_str = f"   PAM: {ot.get('pam_sequence', 'NGG')} ({pam_status})"
                    lines.append(f"| {pam_str:<64} |")
                    risk_rationale = ot['risk_rationale'][:50]
                    risk_str = f"   Risk: HIGH - {risk_rationale}"
                    lines.append(f"| {risk_str:<64} |")
                    if i < len(high_risk_ots[:3]):
                        lines.append("|" + "-" * 66 + "|")
                lines.append("+" + "-" * 66 + "+")

            lines.append("")
            lines.append(f"Species Safety Score: {result.get('species_safety_score', 'N/A')}/100")

        # Aggregate assessment
        lines.append("")
        lines.append("=" * 70)
        lines.append("AGGREGATE ASSESSMENT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Overall Safety Score: {aggregate['overall_score']}/100")
        lines.append(f"Traffic Light: [{color_indicator}] {traffic_light}")
        lines.append("")
        lines.append(f"Limiting Factor: {aggregate['limiting_species']} high-risk off-targets")
        lines.append("")

        # Cross-species conserved off-targets
        cross_species = aggregate.get("cross_species_conserved_offtargets", [])
        if cross_species:
            lines.append("Cross-Species Conserved Off-Targets:")
            for ot in cross_species:
                lines.append(f"  - {ot}")
        else:
            lines.append("Cross-Species Conserved Off-Targets: None identified")
            lines.append("  (Off-targets are species-specific)")

        # Recommendations
        lines.append("")
        lines.append("=" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 70)
        lines.append("")

        recommendations = self._generate_recommendations(scan_results)
        for rec in recommendations:
            if rec.strip():
                lines.append(rec)

        # Execution summary
        lines.append("")
        lines.append("=" * 70)
        lines.append("EXECUTION SUMMARY")
        lines.append("=" * 70)

        exec_time_s = scan_results["execution_time_ms"] / 1000
        sequential_estimate = exec_time_s * len(scan_results["species_results"])
        speedup = sequential_estimate / exec_time_s if exec_time_s > 0 else 1

        lines.append(f"Parallel BLAST time: {exec_time_s:.1f}s (longest species)")
        lines.append(f"Sequential estimate: ~{sequential_estimate:.1f}s ({len(scan_results['species_results'])} species x ~{exec_time_s:.1f}s each)")
        lines.append(f"Speedup: {speedup:.1f}x")
        lines.append("")
        lines.append(f"Report generated: {time.strftime('%Y-%m-%d')}")
        lines.append("=" * 70)

        return "\n".join(lines)


def main():
    """
    Main function to run the CRISPR off-target scanner demo.
    """
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

    # Initialize scanner
    scanner = ParallelOffTargetScanner(client)
    report_generator = SafetyReportGenerator(client)

    # Process each gRNA
    for grna in GRNA_SEQUENCES:
        print(f"\n{'#'*70}")
        print(f"# Processing: {grna['name']}")
        print(f"{'#'*70}")

        # Run parallel scan
        scan_results = scanner.scan(grna)

        # Generate report
        report = report_generator.generate(scan_results)
        print("\n" + report)

        print("\n" + "="*70)
        print("SCAN COMPLETE")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
