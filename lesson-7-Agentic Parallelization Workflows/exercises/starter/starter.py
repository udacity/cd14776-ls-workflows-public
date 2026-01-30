"""
Module 7 Exercise: CRISPR Off-Target Risk Scanner

Parallelization workflow: concurrent BLAST searches against multiple species genomes,
aggregated into a traffic-light safety report.

Learning Objective:
    Implement a parallel workflow using Python to perform concurrent analysis and synthesize the results.
"""

import os
import sys
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from ls_action_space.action_space import blast_sequence


def filter_hits_by_species(hits: List[Dict], species: str) -> List[Dict]:
    """Filter BLAST hits by species name."""
    species_map = {"human": "homo sapiens", "mouse": "mus musculus"}
    target_name = species_map.get(species.lower(), species.lower())
    return [h for h in hits if target_name in f"{h.get('title', '')} {h.get('scientific_name', '')}".lower()]


GRNA_SEQUENCES = [
    {"name": "TP53_exon7_gRNA1", "sequence": "CGTGAGCGCTTCGAGATGTT", "pam": "TGG",
     "target_gene": "TP53", "target_species": "human"},
    {"name": "PCSK9_gRNA2", "sequence": "GACCCTCATCATCACCTGCT", "pam": "GGG",
     "target_gene": "PCSK9", "target_species": "human"}
]


class SpeciesBlastAgent:
    """BLAST agent for a specific species genome."""

    SPECIES_DATABASES = {"human": "human_genomic", "mouse": "mouse_genomic"}
    SPECIES_NAMES = {"human": "Homo sapiens", "mouse": "Mus musculus"}
    COMPATIBLE_PAMS = ["AGG", "CGG", "GGG", "TGG", "AAG", "CAG", "GAG", "TAG"]

    def __init__(self, species: str, llm_client: OpenAI):
        self.species = species.lower()
        self.llm = llm_client
        self.database = self.SPECIES_DATABASES.get(self.species, "nt")
        self.species_name = self.SPECIES_NAMES.get(self.species, species)

    def _classify_off_target_risk(self, mismatches: int) -> str:
        """Classify risk: 0-2=High, 3-4=Medium, >=5=Low."""
        if mismatches <= 2:
            return "High"
        elif mismatches <= 4:
            return "Medium"
        return "Low"

    def _get_risk_rationale(self, mismatches: int, positions: List[int], location: str) -> str:
        """Generate rationale for risk classification."""
        if mismatches == 0:
            return "Perfect match - on-target site"
        elif mismatches <= 2:
            return f"{mismatches} mismatch(es) at {positions} - high probability of cutting"
        elif mismatches <= 4:
            return f"{mismatches} mismatches - possible off-target activity"
        return f"{mismatches} mismatches - negligible off-target risk"

    def _generate_mismatch_positions(self, mismatches: int, query_len: int = 20) -> List[int]:
        """Generate deterministic mismatch positions."""
        if mismatches == 0:
            return []
        random.seed(mismatches * 31)
        return sorted(random.sample(range(1, query_len + 1), min(mismatches, query_len)))

    def _check_pam_compatibility(self, accession: str) -> tuple:
        """Check if PAM sequence is present near hit site."""
        random.seed(hash(accession))
        pam = random.choice(self.COMPATIBLE_PAMS + ["AAA", "TTT", "CCC"])
        return pam in self.COMPATIBLE_PAMS, pam

    def _extract_gene_context(self, description: str) -> str:
        """Extract gene/region context from hit description."""
        desc_lower = description.lower()
        if "intergenic" in desc_lower:
            return "Intergenic"
        elif "intron" in desc_lower:
            return "Intronic"
        elif "exon" in desc_lower:
            return "Exonic"
        return "Unknown region"

    def _identify_on_target(self, hits: List[Dict], target_gene: str) -> Optional[Dict]:
        """Identify on-target hit matching intended target gene."""
        for hit in hits:
            if hit["mismatches"] <= 1 and target_gene.upper() in hit["description"].upper():
                return hit
        return hits[0] if hits and hits[0]["mismatches"] <= 2 else None

    def search(self, grna_sequence: str, target_gene: str) -> Dict[str, Any]:
        """
        BLAST gRNA against species genome and analyze off-targets.

        Returns: {species, query_sequence, total_hits, on_target_hit, off_target_hits: [{
        accession, gene_context, identity_pct, mismatches, mismatch_positions,
        pam_compatible, risk_level, risk_rationale}], off_target_summary,
        species_safety_score, interpretation}
        """
        # TODO: Step 1 - Run BLAST search using blast_sequence()
        #       Parameters: sequence=grna_sequence, database=self.database,
        #                  program="blastn", hitlist_size=50

        # TODO: Step 2 - Filter hits by species if using "nt" database

        # TODO: Step 3 - Identify on-target hit using _identify_on_target()

        # TODO: Step 4 - Analyze off-target hits:
        #       For each hit (excluding on-target):
        #       - Get mismatches, generate positions, classify risk
        #       - Check PAM, extract gene context
        #       - Track counts: high_risk, medium_risk, low_risk

        # TODO: Step 5 - Sort off-targets by risk (High first)

        # TODO: Step 6 - Calculate species safety score:
        #       Start at 100; human: -15 per high-risk, -5 per medium
        #       Other species: -10 per high-risk, -3 per medium; min 0

        # TODO: Step 7 - Return complete results dict
        pass


class ParallelOffTargetScanner:
    """Runs BLAST against multiple species in parallel (fan-out/fan-in pattern)."""

    SPECIES_TO_SCAN = ["human", "mouse"]

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client
        self.agents = {sp: SpeciesBlastAgent(sp, llm_client) for sp in self.SPECIES_TO_SCAN}

    def scan(self, grna_data: Dict[str, str]) -> Dict[str, Any]:
        """Scan gRNA against all species in parallel."""
        grna_seq = grna_data["sequence"]
        target_gene = grna_data["target_gene"]

        print(f"\n{'='*60}\nPARALLEL SCAN: {grna_data['name']}\n{'='*60}")
        start_time = time.time()
        results = {}

        # KEY PARALLELIZATION PATTERN
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_species = {
                executor.submit(agent.search, grna_seq, target_gene): species
                for species, agent in self.agents.items()
            }
            for future in as_completed(future_to_species):
                species = future_to_species[future]
                try:
                    result = future.result() or {"error": "search() not implemented"}
                    results[species] = result
                    high = result.get("off_target_summary", {}).get("high_risk", 0)
                    print(f"  + {species}: {result.get('total_hits', 0)} hits, {high} high-risk")
                except Exception as e:
                    results[species] = {"error": str(e)}

        aggregate = self._calculate_aggregate_safety(results)
        if aggregate is None:
            aggregate = {"overall_score": 0, "traffic_light": "RED",
                        "limiting_species": "unknown", "cross_species_conserved_offtargets": []}

        return {"grna_name": grna_data["name"], "sequence": grna_seq, "target_gene": target_gene,
                "species_results": results, "aggregate_safety": aggregate,
                "execution_time_ms": int((time.time() - start_time) * 1000)}

    def _calculate_aggregate_safety(self, results: Dict) -> Dict[str, Any]:
        """
        Calculate overall safety score from species results.

        Scoring: Start 100; human: -15/high, -5/medium; other: -10/high, -3/medium
        Traffic light: GREEN (>=80, no human high), YELLOW (50-79 or any high), RED (<50 or >=3 human high)
        """
        # TODO: Step 1 - Initialize: overall_score=100, human_high_risk=0, limiting_species=None

        # TODO: Step 2 - Loop through species results; apply penalties

        # TODO: Step 3 - Ensure minimum score of 0

        # TODO: Step 4 - Determine traffic light based on rules above

        # TODO: Step 5 - Find cross-species conserved off-targets via _find_cross_species_offtargets()

        # TODO: Step 6 - Return aggregate safety dict
        pass

    def _find_cross_species_offtargets(self, results: Dict) -> List[str]:
        """Identify high-risk off-targets appearing in multiple species."""
        species_genes = {}
        for species, result in results.items():
            if "error" in result:
                continue
            genes = {ot["gene_context"] for ot in result.get("off_target_hits", []) if ot["risk_level"] == "High"}
            species_genes[species] = genes

        all_genes = set().union(*species_genes.values()) if species_genes else set()
        return [g for g in all_genes if sum(1 for genes in species_genes.values() if g in genes) > 1]


class SafetyReportGenerator:
    """Generates traffic-light safety report."""

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def generate(self, scan_results: Dict[str, Any]) -> str:
        """Generate formatted safety report."""
        agg = scan_results["aggregate_safety"]
        light = agg["traffic_light"]
        lines = [
            "=" * 60, "OFF-TARGET SAFETY REPORT", "=" * 60, "",
            f"gRNA: {scan_results['grna_name']}",
            f"Sequence: 5'-{scan_results['sequence']}-3' + NGG",
            f"Target: {scan_results['target_gene']}", "",
            f"[{light[0]}] {light} - Score: {agg['overall_score']}/100", ""
        ]

        for species in ["human", "mouse"]:
            result = scan_results["species_results"].get(species, {})
            if "error" in result:
                lines.append(f"{species.upper()}: Error")
                continue
            summary = result.get("off_target_summary", {})
            lines.append(f"{species.upper()}: {summary.get('high_risk', 0)} high, "
                        f"{summary.get('medium_risk', 0)} medium, {summary.get('low_risk', 0)} low risk")

        lines.extend(["", f"Limiting: {agg.get('limiting_species', 'N/A')}",
                     f"Time: {scan_results['execution_time_ms']}ms", "=" * 60])
        return "\n".join(lines)


def main():
    """Main entry point."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")
    scanner = ParallelOffTargetScanner(client)
    report_gen = SafetyReportGenerator(client)

    print("Module 7: CRISPR Off-Target Risk Scanner")
    print("Parallelization: concurrent species BLAST searches\n")

    results = scanner.scan(GRNA_SEQUENCES[0])
    print(report_gen.generate(results))


if __name__ == "__main__":
    main()
