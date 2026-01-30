"""
Drug Repositioning Agentic Workflow - Student Scaffold
=======================================================

CAPSTONE PROJECT: Build an agentic workflow for drug repurposing

You will implement a multi-agent system that:
1. Plans a drug repurposing investigation (ActionPlanningAgent)
2. Routes tasks to specialized agents (RoutingAgent)
3. Mines biomedical databases for candidate drugs (DataMiningAgent)
4. Enriches candidates with literature and safety data (LiteratureAgent, SafetyAgent)
5. Scores and ranks candidates using an evaluator-optimizer loop (EvaluationAgent)
6. Generates a validation roadmap (EvaluationAgent)

LEARNING OBJECTIVES DEMONSTRATED:
- Sequential workflows (DataMiningAgent)
- Prompt chaining (LiteratureAgent: search -> assess)
- LLM-based routing (RoutingAgent)
- Evaluator-optimizer pattern (EvaluationAgent)
- Multi-agent orchestration (DrugRepurposingOrchestrator)

ESTIMATED TIME: 6-7 hours

HOW TO RUN:
    # Mock mode (no API keys needed - use for development)
    USE_MOCK_DATA=true python scaffold.py

    # Live mode (requires API keys)
    OPENAI_API_KEY=your_key python scaffold.py

INSTRUCTIONS:
    Search for "TODO" comments - there are 10 tasks to complete.
    Each TODO includes hints and expected behavior.

    Focus on:
    - Writing effective prompts that produce structured JSON output
    - Implementing the workflow patterns (sequential, chaining, eval-optimizer)
    - Understanding how agents coordinate through the orchestrator
"""

import os
import json
import time
import logging
import requests
import re
import unicodedata
import pandas as pd
import hashlib
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# =============================================================================
# API QUERY FUNCTIONS TO OBTAIN DATA  (PROVIDED - DO NOT MODIFY)
# =============================================================================

from ls_action_space.action_space import (
    query_chembl,        # Query ChEMBL for compounds active against a gene target
    query_opentargets,   # Query Open Targets for known drugs for a gene target
    query_pubmed,        # Search PubMed and return article summaries for a term
    query_faers,         # Query FDA FAERS for adverse event counts/reactions for a drug name
    query_fda_labels,    # Query FDA drug labels for boxed warnings (and whether a label was found)
)

# =============================================================================
# CONFIGURATION (PROVIDED - DO NOT MODIFY)
# =============================================================================

load_dotenv(".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DrugRepurposing")

USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
FIXED_MOCK_ISO_TS = "2025-01-15T00:00:00"

CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

# Scope control
MAX_CANDIDATES_TO_ENRICH = int(os.getenv("MAX_CANDIDATES_TO_ENRICH", "5"))
TOP_N_DISPLAY = int(os.getenv("TOP_N_DISPLAY", "3"))
MAX_PUBMED_QUERIES_PER_CANDIDATE = 3
MAX_PUBMED_ARTICLES_PER_CANDIDATE = 5
MAX_CHEMBL_ACTIVITIES = 200
MAX_CHEMBL_COMPOUNDS = 15

REQUIRED_TASK_TYPES = ["data_mining", "enrichment", "scoring", "roadmap"]

# Agent registry for routing
AGENT_REGISTRY = {
    "data_mining": "DataMiningAgent",
    "enrichment": "LiteratureAgent + SafetyAgent",
    "scoring": "EvaluationAgent",
    "roadmap": "EvaluationAgent",
}


# =============================================================================
# OPENAI CLIENT SETUP (PROVIDED - DO NOT MODIFY)
# =============================================================================

def init_openai_client():
    """Initialize OpenAI client."""
    if USE_MOCK_DATA:
        logger.info("Mock mode enabled - skipping OpenAI client initialization")
        return None

    # For production (Vocareum): use UDACITY_OPENAI_API_KEY with base_url
    # For local testing: use OPENAI_API_KEY without base_url
    udacity_key = os.getenv("UDACITY_OPENAI_API_KEY")
    local_key = os.getenv("OPENAI_API_KEY")

    if udacity_key:
        from openai import OpenAI
        return OpenAI(api_key=udacity_key, base_url="https://openai.vocareum.com/v1")
    elif local_key:
        from openai import OpenAI
        return OpenAI(api_key=local_key)
    else:
        raise EnvironmentError(
            "Set either UDACITY_OPENAI_API_KEY (for Vocareum) or OPENAI_API_KEY (for local testing)"
        )


client = init_openai_client()

# =============================================================================
# UTILITY FUNCTIONS (PROVIDED - DO NOT MODIFY)
# =============================================================================

STOPWORDS = {"a", "an", "and", "or", "the", "of", "to", "in", "with", "for", "on", "by", "from",
             "disease", "disorder", "syndrome", "condition"}


def now_iso() -> str:
    return FIXED_MOCK_ISO_TS if USE_MOCK_DATA else datetime.now().isoformat()


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s]+", "_", text).strip("_")[:60]


def normalize_drug_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower().strip()
    for pattern in [r"\s+hydrochloride$", r"\s+hcl$", r"\s+sodium$", r"\s+potassium$",
                    r"\s+acetate$", r"\s+sulfate$", r"\s+phosphate$", r"\s+mesylate$"]:
        name = re.sub(pattern, "", name)
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"[^\w\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def disease_match_score(disease: str, indication: str) -> float:
    d, i = normalize_text(disease), normalize_text(indication)
    if not d or not i:
        return 0.0
    if d in i or i in d:
        return 1.0
    dt = set(t for t in d.split() if t not in STOPWORDS and len(t) > 2)
    it = set(t for t in i.split() if t not in STOPWORDS and len(t) > 2)
    if not dt or not it:
        return 0.0
    overlap = len(dt & it)
    frac = overlap / max(1, len(dt))
    if overlap < 2 and len(dt) >= 3:
        return 0.0
    return min(0.80, max(0.0, frac))

def pick_searchable_name(name: str, synonyms: List[str], fallback_id: str) -> str:
    """Choose a primary name that works for PubMed/FAERS."""
    if is_valid_search_name(name):
        return name
    for s in (synonyms or []):
        if is_valid_search_name(s):
            return s
    return fallback_id

def is_valid_search_name(name: str) -> bool:
    """Check if a name is valid for PubMed/FAERS searches."""
    if not name:
        return False
    if re.match(r"^CHEMBL\d+$", name, re.IGNORECASE):
        return False
    if len(name) < 3:
        return False
    return True


def get_searchable_names(candidate) -> List[str]:
    """Get list of searchable names for a candidate (name + synonyms)."""
    names = []
    if is_valid_search_name(candidate.name):
        names.append(candidate.name)
    for syn in candidate.synonyms:
        if is_valid_search_name(syn) and syn.lower() not in [n.lower() for n in names]:
            names.append(syn)
    return names[:3]


def robust_parse_json(response: str, fallback: Any = None) -> Any:
    if not response:
        return fallback
    text = response.strip()
    try:
        return json.loads(text)
    except:
        pass
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except:
            pass
    for sc, ec in [("{", "}"), ("[", "]")]:
        idx = text.find(sc)
        if idx != -1:
            depth = 0
            for i in range(idx, len(text)):
                if text[i] == sc:
                    depth += 1
                elif text[i] == ec:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[idx:i + 1])
                        except:
                            break
    return fallback


# =============================================================================
# HTTP HELPERS (PROVIDED - DO NOT MODIFY)
# =============================================================================

def _cache_path(source: str, key: str) -> str:
    return os.path.join(CACHE_DIR, f"{slugify(source)}_{key}.json")


def _make_cache_key(url: str, params: Optional[Dict] = None, json_data: Optional[Dict] = None) -> str:
    blob = {"url": url, "params": params or {}, "json": json_data or {}}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:24]


def robust_request(method: str, url: str, *, params: Optional[Dict] = None,
                   json_data: Optional[Dict] = None, timeout: int = 30,
                   retries: int = 3, source: str = "generic") -> Optional[Dict]:
    if USE_MOCK_DATA:
        return None
    cache_key = _make_cache_key(url, params, json_data)
    cpath = _cache_path(source, cache_key)
    if CACHE_ENABLED and os.path.exists(cpath):
        try:
            with open(cpath) as f:
                return json.load(f)
        except:
            pass
    for attempt in range(retries + 1):
        try:
            time.sleep(0.2)
            resp = requests.get(url, params=params, timeout=timeout) if method.upper() == "GET" \
                else requests.post(url, params=params, json=json_data, timeout=timeout)
            if resp.status_code == 404:
                return None
            if resp.status_code in (429, 503, 502, 504):
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            if CACHE_ENABLED:
                try:
                    with open(cpath, "w") as f:
                        json.dump(data, f, indent=2)
                except:
                    pass
            return data
        except requests.exceptions.HTTPError as e:
            if e.response is not None and 400 <= e.response.status_code < 500:
                logger.warning(f"{source} {method} client error: {url} - {e}")
                return None
            if attempt >= retries:
                logger.warning(f"{source} {method} failed: {url} - {e}")
                return None
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt >= retries:
                logger.warning(f"{source} {method} failed: {url} - {e}")
                return None
            time.sleep(2 ** attempt)
    return None


# =============================================================================
# DATA CLASSES (PROVIDED - DO NOT MODIFY)
# =============================================================================

@dataclass
class RoutingDecision:
    decision_id: str
    task_type: str
    agent_name: str
    inputs: Dict[str, Any]
    reasoning: str
    created_at: str = field(default_factory=now_iso)


@dataclass
class Provenance:
    chembl_id: str = ""
    opentargets_id: str = ""
    pubmed_pmids: List[str] = field(default_factory=list)
    pubmed_queries_tried: List[str] = field(default_factory=list)
    faers_query_matched: str = ""
    routing_decision_ids: List[str] = field(default_factory=list)


@dataclass
class AuditLog:
    started_at: str = ""
    completed_at: str = ""
    target: str = ""
    disease: str = ""
    mock_mode: bool = False
    plan_generated: Dict = field(default_factory=dict)
    steps_executed: List[Dict] = field(default_factory=list)
    candidates_found: int = 0
    candidates_enriched: int = 0
    llm_calls: int = 0
    routing_decisions: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class DrugCandidate:
    chembl_id: str = ""
    opentargets_id: str = ""
    name: str = ""
    normalized_name: str = ""
    synonyms: List[str] = field(default_factory=list)
    source: str = "chembl"
    pchembl_value: float = 0.0
    max_phase: int = 0
    molecule_type: str = ""
    mechanism_of_action: str = ""
    indications: List[str] = field(default_factory=list)
    disease_relevance_score: float = 0.0
    # Enrichment tracking
    is_enriched: bool = False
    enrichment_timestamp: str = ""
    literature_strategy: str = ""
    safety_strategy: str = ""
    # Literature
    pubmed_articles: List[Dict] = field(default_factory=list)
    literature_summary: str = ""
    literature_support_level: str = "none"
    literature_strength: int = 0
    literature_supporting_pmids: List[str] = field(default_factory=list)
    # Safety
    safety_summary: str = ""
    safety_flags: List[str] = field(default_factory=list)
    faers_event_count: int = 0
    faers_serious_count: int = 0
    has_boxed_warning: bool = False
    label_found: bool = False
    # Scoring
    scores: Dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    ranking_rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)


@dataclass
class ActionPlan:
    objective: str
    steps: List[Dict]


@dataclass
class RepurposingResult:
    target: str
    disease: str
    timestamp: str
    action_plan: ActionPlan
    candidates: List[DrugCandidate]
    validation_roadmap: str
    ranking_history: List[Dict] = field(default_factory=list)
    audit_log: AuditLog = field(default_factory=AuditLog)
    errors: List[str] = field(default_factory=list)


# =============================================================================
# MOCK DATA (PROVIDED - DO NOT MODIFY)
# =============================================================================

MOCK_CHEMBL_DATA = {
    "default": [
        {"molecule_id": "CHEMBL53", "name": "Haloperidol", "pchembl_value": 8.5, "max_phase": 4,
         "molecule_type": "Small molecule", "synonyms": ["Haldol"]},
        {"molecule_id": "CHEMBL266912", "name": "Siramesine", "pchembl_value": 8.1, "max_phase": 2,
         "molecule_type": "Small molecule", "synonyms": ["Lu 28-179"]},
        {"molecule_id": "CHEMBL404539", "name": "PB28", "pchembl_value": 7.8, "max_phase": 0,
         "molecule_type": "Small molecule", "synonyms": []},
        {"molecule_id": "CHEMBL1627", "name": "Rimcazole", "pchembl_value": 7.2, "max_phase": 2,
         "molecule_type": "Small molecule", "synonyms": ["BW-234U"]},
        {"molecule_id": "CHEMBL319898", "name": "Cutamesine", "pchembl_value": 6.9, "max_phase": 2,
         "molecule_type": "Small molecule", "synonyms": ["SA-4503"]},
    ]
}

MOCK_OT_DATA = {
    "default": {
        "target_id": "ENSG00000109G23",
        "known_drugs": [
            {"drug": {"id": "CHEMBL53", "name": "Haloperidol"}, "mechanismOfAction": "Sigma receptor modulator",
             "disease": {"name": "Schizophrenia"}, "phase": 4},
            {"drug": {"id": "CHEMBL1201135", "name": "Pridopidine"}, "mechanismOfAction": "Sigma-1 receptor agonist",
             "disease": {"name": "Huntington disease"}, "phase": 3},
        ]
    }
}

MOCK_PUBMED_DATA = {
    "haloperidol": [
        {"pmid": "31234567", "title": "Sigma-2 receptor binding of haloperidol and implications for liver disease",
         "abstract": "This study demonstrates that haloperidol binds sigma-2 receptors with high affinity. We investigated effects on hepatic lipid accumulation in steatosis models, finding significant reduction in lipid droplet formation.",
         "journal": "J Pharmacol", "year": "2023"},
        {"pmid": "30987654", "title": "Antipsychotics and metabolic effects in liver",
         "abstract": "Review of antipsychotic medications and their hepatic effects. Haloperidol showed mixed results.",
         "journal": "Hepatology Rev", "year": "2022"},
    ],
    "siramesine": [
        {"pmid": "29876543", "title": "Siramesine induces lysosomal membrane permeabilization",
         "abstract": "Siramesine, a sigma-2 receptor ligand, demonstrates potent effects on lysosomal function and autophagy pathways relevant to metabolic disease.",
         "journal": "Cell Death Dis", "year": "2021"},
    ],
    "default": [
        {"pmid": "28765432", "title": "Sigma receptors in metabolic disease",
         "abstract": "General overview of sigma receptor biology in metabolic disorders.",
         "journal": "Trends Pharmacol Sci", "year": "2020"},
    ]
}

MOCK_FAERS_DATA = {
    "haloperidol": {"total_events": 45000, "serious_events": 28000,
                    "top_reactions": [{"reaction": "Dystonia", "count": 2500}]},
    "siramesine": {"total_events": 50, "serious_events": 15, "top_reactions": []},
    "default": {"total_events": 0, "serious_events": 0, "top_reactions": []}
}

MOCK_FDA_LABELS = {
    "haloperidol": {
        "boxed_warning": "WARNING: INCREASED MORTALITY IN ELDERLY PATIENTS WITH DEMENTIA-RELATED PSYCHOSIS.",
        "label_found": True},
    "default": {"boxed_warning": "", "label_found": False}
}


def get_mock_chembl(gene: str) -> List[Dict]:
    return MOCK_CHEMBL_DATA.get(gene.upper(), MOCK_CHEMBL_DATA["default"])


def get_mock_ot(gene: str) -> Dict:
    return MOCK_OT_DATA.get(gene.upper(), MOCK_OT_DATA["default"])


def get_mock_pubmed(drug_name: str) -> List[Dict]:
    return MOCK_PUBMED_DATA.get(normalize_drug_name(drug_name), MOCK_PUBMED_DATA["default"])


def get_mock_faers(drug_name: str) -> Dict:
    return MOCK_FAERS_DATA.get(normalize_drug_name(drug_name), MOCK_FAERS_DATA["default"])


def get_mock_fda_label(drug_name: str) -> Dict:
    return MOCK_FDA_LABELS.get(normalize_drug_name(drug_name), MOCK_FDA_LABELS["default"])

# =============================================================================
# BASE AGENT CLASS (PROVIDED - DO NOT MODIFY)
# =============================================================================

_llm_call_count = 0


def get_llm_call_count() -> int:
    return _llm_call_count


def reset_llm_call_count():
    global _llm_call_count
    _llm_call_count = 0


class BaseAgent:
    """Base class for all agents with LLM capabilities."""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def _call_llm(self, prompt: str, temperature: float = 0) -> str:
        """Call the LLM with the given prompt. Returns response text."""
        global _llm_call_count
        if USE_MOCK_DATA:
            _llm_call_count += 1
            return self._mock_llm_response(prompt)
        try:
            _llm_call_count += 1
            response = client.chat.completions.create(
                model=MODEL, max_tokens=2048, temperature=temperature,
                messages=[{"role": "system", "content": f"You are {self.name}. {self.role}"},
                          {"role": "user", "content": prompt}])
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM error in {self.name}: {e}")
            return ""

    def _call_llm_json(self, prompt: str, fallback: Any = None) -> Any:
        """Call LLM and parse response as JSON."""
        response = self._call_llm(prompt)
        parsed = robust_parse_json(response, None)
        if parsed is None and not USE_MOCK_DATA:
            response = self._call_llm(prompt + "\n\nIMPORTANT: Return ONLY valid JSON.")
            parsed = robust_parse_json(response, fallback)
        return parsed if parsed is not None else fallback

    def _mock_llm_response(self, prompt: str) -> str:
        """Mock responses for testing without API keys."""
        pl = prompt.lower()
        if "create an action plan" in pl:
            return json.dumps({"objective": "Find repurposable drugs for target in disease",
                               "steps": [{"step": 1, "action": "Mine databases", "task_type": "data_mining"},
                                         {"step": 2, "action": "Enrich candidates", "task_type": "enrichment"},
                                         {"step": 3, "action": "Score and rank", "task_type": "scoring"},
                                         {"step": 4, "action": "Generate roadmap", "task_type": "roadmap"}]})
        if "determine the best enrichment strategy" in pl or "enrichment strategy" in pl:
            if "haloperidol" in pl:
                return json.dumps(
                    {"literature_strategy": "target_focused", "literature_reasoning": "Approved drug with known MOA",
                     "safety_strategy": "comprehensive", "safety_reasoning": "Phase 4 needs full review"})
            return json.dumps(
                {"literature_strategy": "disease_focused", "literature_reasoning": "Look for disease evidence",
                 "safety_strategy": "faers_only", "safety_reasoning": "Limited clinical data"})
        if "support_level" in pl and "supporting_pmids" in pl:
            return json.dumps({"support_level": "moderate", "strength": 6, "supporting_pmids": ["31234567"],
                               "summary": "Evidence suggests target engagement with potential disease relevance."})
        if "critique" in pl and "ranking" in pl:
            return "Rankings are appropriate. The candidates are well-ordered by binding affinity and development stage."
        if "suggest score adjustments" in pl or "adjustments" in pl:
            return json.dumps([])
        if "2-sentence rationale" in pl or "rationale" in pl:
            return "Strong target binding combined with clinical data supports repurposing potential. Safety is manageable with monitoring."
        if "summarize the safety" in pl:
            return "Based on FAERS data, the drug has a known adverse event profile consistent with its class."
        if "validation roadmap" in pl or "roadmap" in pl:
            return "## Validation Roadmap\n\n### Executive Summary\nTop candidates warrant validation.\n\n### Month 1\nTarget engagement assays\n\n### Month 2\nDisease models\n\n### Month 3\nGo/no-go decision"
        return "Mock response"


# =============================================================================
# =============================================================================
#
#                         YOUR IMPLEMENTATION BELOW
#
# =============================================================================
# =============================================================================


class ActionPlanningAgent(BaseAgent):
    """
    Creates action plans for drug repurposing workflows.

    LEARNING OBJECTIVE: Task decomposition and planning
    """

    def __init__(self):
        super().__init__(
            "ActionPlanningAgent",
            "You create structured action plans for drug repurposing workflows."
        )

    def create_plan(self, target: str, disease: str) -> ActionPlan:
        """
        Create an action plan for the drug repurposing task.

        TODO 1: Write a prompt that asks the LLM to create a structured action plan.

        Your prompt should:
        - Include the TARGET gene and DISEASE as context
        - Request JSON output with "objective" and "steps" fields
        - Specify that each step needs: "step" (number), "action" (description), "task_type"
        - List the required task_types in order: data_mining, enrichment, scoring, roadmap

        Example JSON output format:
        {
            "objective": "Find repurposable drugs for TMEM97 in NASH",
            "steps": [
                {"step": 1, "action": "Query databases for compounds", "task_type": "data_mining"},
                ...
            ]
        }

        HINTS:
        - Use self._call_llm_json(prompt, fallback={}) to get parsed JSON
        - The _build_plan helper validates the response and handles fallbacks
        """

        # =====================================================================
        # TODO 1: Write your prompt here
        # =====================================================================
        prompt = f"""
        # YOUR PROMPT HERE
        # Include: TARGET={target}, DISEASE={disease}
        # Request JSON with objective and steps
        # Each step needs: step number, action description, task_type
        # Required task_types: data_mining, enrichment, scoring, roadmap
        """

        raw = self._call_llm_json(prompt, {})
        # =====================================================================
        # END TODO 1
        # =====================================================================

        return self._build_plan(raw, target, disease)

    def _build_plan(self, parsed: Dict, target: str, disease: str) -> ActionPlan:
        """Build ActionPlan from parsed response with fallback. (PROVIDED)"""
        obj = parsed.get("objective", f"Find repurposable drugs for {target} in {disease}")
        steps = parsed.get("steps", [])
        task_types_found = {s.get("task_type") for s in steps}
        if set(REQUIRED_TASK_TYPES) - task_types_found:
            steps = [{"step": 1, "action": "Mine ChEMBL and Open Targets", "task_type": "data_mining"},
                     {"step": 2, "action": "Enrich with literature and safety", "task_type": "enrichment"},
                     {"step": 3, "action": "Score and rank candidates", "task_type": "scoring"},
                     {"step": 4, "action": "Generate validation roadmap", "task_type": "roadmap"}]
        return ActionPlan(objective=obj, steps=steps)


class RoutingAgent(BaseAgent):
    """
    Routes tasks to specialized agents based on candidate characteristics.

    LEARNING OBJECTIVE: LLM-based routing/classification

    This agent demonstrates how LLMs can make routing decisions dynamically,
    rather than using hardcoded if/else rules. The LLM analyzes candidate
    properties and chooses the appropriate strategy.
    """

    def __init__(self):
        super().__init__(
            "RoutingAgent",
            "You analyze drug candidates and determine optimal processing strategies."
        )

    def route_enrichment_strategy(self, candidate: DrugCandidate, target: str, disease: str) -> RoutingDecision:
        """
        Use LLM to determine the best enrichment strategy for a candidate.

        TODO 2: Write a prompt that asks the LLM to decide the enrichment strategy.

        This demonstrates LLM-BASED ROUTING - the LLM classifies the candidate
        and decides which processing strategy to use based on its characteristics.

        Your prompt should provide candidate information:
        - Name, ChEMBL ID, development phase, mechanism of action, pchembl_value
        - Target gene and disease context

        Ask the LLM to choose:
        - literature_strategy: "target_focused", "disease_focused", or "broad_search"
        - safety_strategy: "comprehensive", "faers_only", or "basic"

        Include guidelines in your prompt:
        - Phase 3-4 drugs: comprehensive safety (more clinical data exists)
        - Phase 0-2 drugs: faers_only or basic safety
        - Unknown MOA: prefer disease_focused literature search
        - Known MOA: prefer target_focused literature search

        Expected JSON output:
        {
            "literature_strategy": "target_focused|disease_focused|broad_search",
            "literature_reasoning": "why this strategy",
            "safety_strategy": "comprehensive|faers_only|basic",
            "safety_reasoning": "why this depth"
        }
        """

        # =====================================================================
        # TODO 2: Write your routing prompt here
        # =====================================================================
        prompt = f"""
        # YOUR PROMPT HERE
        # Provide: candidate name, phase, MOA, pchembl_value
        # Context: target gene, disease
        # Ask for: literature_strategy, safety_strategy with reasoning
        # Include routing guidelines
        """

        fallback = {
            "literature_strategy": "broad_search",
            "literature_reasoning": "Default fallback",
            "safety_strategy": "basic",
            "safety_reasoning": "Default fallback"
        }

        result = self._call_llm_json(prompt, fallback)
        # =====================================================================
        # END TODO 2
        # =====================================================================

        decision_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"route_{candidate.name}")) if USE_MOCK_DATA else str(
            uuid.uuid4())

        return RoutingDecision(
            decision_id=decision_id,
            task_type="enrichment_routing",
            agent_name="RoutingAgent",
            inputs={
                "literature_strategy": result.get("literature_strategy", "broad_search"),
                "safety_strategy": result.get("safety_strategy", "basic")
            },
            reasoning=f"Literature: {result.get('literature_reasoning', 'N/A')}. Safety: {result.get('safety_reasoning', 'N/A')}"
        )


class DataMiningAgent(BaseAgent):
    """
    Mines drug-target databases for candidate compounds.

    LEARNING OBJECTIVE: Sequential workflow pattern

    This agent demonstrates sequential workflows where each step depends
    on the previous step's output. The pattern is:
    Step 1 (ChEMBL) -> Step 2 (Open Targets) -> Step 3 (Merge)
    """

    def __init__(self):
        super().__init__(
            "DataMiningAgent",
            "You query drug-target databases to find candidate compounds."
        )

    def run(self, target: str, disease: str) -> Tuple[List[DrugCandidate], str]:
        """
        Execute a SEQUENTIAL data mining workflow.

        TODO 3: Implement the three-step sequential workflow:

        Step 1: Query ChEMBL for compounds that bind to the target
            - Call: query_chembl(target, max_compounds=MAX_CHEMBL_COMPOUNDS)
            - Returns: (list_of_compounds, chembl_target_id)
            - Log: logger.info(f"  Step 1/3: Found {len(compounds)} compounds from ChEMBL")

        Step 2: Query Open Targets for known drugs
            - Call: query_opentargets(target)
            - Returns: dict with "target_id" and "known_drugs" list
            - Log: logger.info(f"  Step 2/3: Found {len(known_drugs)} drugs from Open Targets")

        Step 3: Merge and deduplicate results from both sources
            - Call: self._merge_sources(chembl_compounds, ot_data, chembl_target_id, disease)
            - Sort the merged list by (pchembl_value, max_phase) descending
            - Log: logger.info(f"  Step 3/3: Merged to {len(candidates)} unique candidates")

        Return: (sorted_candidates, chembl_target_id)

        HINTS:
        - The query functions handle all API details - just call them
        - Sort with: candidates.sort(key=lambda c: (c.pchembl_value, c.max_phase), reverse=True)
        """
        logger.info("DataMiningAgent: Starting sequential data mining...")

        # =====================================================================
        # TODO 3: Implement sequential workflow
        # =====================================================================

        # Step 1: Query ChEMBL
        # chembl_compounds, chembl_target_id = query_chembl(...)

        # Step 2: Query Open Targets
        # ot_data = query_opentargets(...)

        # Step 3: Merge and sort
        # candidates = self._merge_sources(...)
        # candidates.sort(...)

        candidates = []  # Replace with your implementation
        chembl_target_id = ""  # Replace with your implementation

        # =====================================================================
        # END TODO 3
        # =====================================================================

        logger.info(f"  Final: {len(candidates)} unique candidates")
        return candidates, chembl_target_id

    def _merge_sources(self, chembl_data: List[Dict], ot_data: Dict,
                       chembl_target_id: str, disease: str) -> List[DrugCandidate]:
        """Merge candidates from ChEMBL and Open Targets. (PROVIDED)"""
        candidates = {}

        for compound in chembl_data:
            chembl_id = compound["molecule_id"]

            synonyms = compound.get("synonyms", []) or []
            raw_name = (compound.get("name") or chembl_id).strip()

            # Ensure candidate.name is a human-readable/searchable name when possible
            name = pick_searchable_name(raw_name, synonyms, chembl_id)

            # Keep synonyms clean + include the other name variant if useful
            syn_set = []
            for s in (synonyms or []):
                s2 = (s or "").strip()
                if s2 and s2.lower() not in {x.lower() for x in syn_set}:
                    syn_set.append(s2)
            if raw_name and raw_name != name and is_valid_search_name(raw_name):
                if raw_name.lower() not in {x.lower() for x in syn_set}:
                    syn_set.insert(0, raw_name)

            candidates[chembl_id] = DrugCandidate(
                chembl_id=chembl_id,
                name=name,
                normalized_name=normalize_drug_name(name),
                synonyms=syn_set,
                source="chembl",
                pchembl_value=float(compound.get("pchembl_value") or 0.0),
                max_phase=int(float(compound.get("max_phase")) or 0),
                molecule_type=compound.get("molecule_type", ""),
                provenance=Provenance(chembl_id=chembl_id, chembl_target_id=chembl_target_id)
            )

        for row in ot_data.get("known_drugs", []) or []:
            drug = row.get("drug", {}) or {}
            ot_id = drug.get("id", "")
            ot_name = drug.get("name", "")
            moa = row.get("mechanismOfAction", "")
            indication = (row.get("disease", {}) or {}).get("name", "")
            phase = int(row.get("phase") or 0)

            matched = None
            if ot_id.startswith("CHEMBL") and ot_id in candidates:
                matched = candidates[ot_id]
            else:
                norm_ot = normalize_drug_name(ot_name)
                for c in candidates.values():
                    if c.normalized_name == norm_ot:
                        matched = c
                        break

            if matched:
                matched.source = "both"
                matched.opentargets_id = ot_id
                if moa:
                    matched.mechanism_of_action = moa
                if indication and indication not in matched.indications:
                    matched.indications.append(indication)
                matched.max_phase = max(matched.max_phase, phase)
                matched.disease_relevance_score = max(matched.disease_relevance_score,
                                                      disease_match_score(disease, indication))
            elif ot_id and ot_id not in candidates:
                candidates[ot_id] = DrugCandidate(
                    opentargets_id=ot_id, name=ot_name, normalized_name=normalize_drug_name(ot_name),
                    source="opentargets", mechanism_of_action=moa,
                    indications=[indication] if indication else [], max_phase=phase,
                    disease_relevance_score=disease_match_score(disease, indication),
                    provenance=Provenance(opentargets_id=ot_id))

        return list(candidates.values())


class LiteratureAgent(BaseAgent):
    """
    Searches and assesses biomedical literature.

    LEARNING OBJECTIVE: Prompt chaining pattern

    This agent demonstrates prompt chaining where:
    - Chain Step 1: Search PubMed (data retrieval)
    - Chain Step 2: Assess results with LLM (analysis)

    The output of step 1 becomes input for step 2.
    """

    def __init__(self):
        super().__init__(
            "LiteratureAgent",
            "You assess biomedical literature for drug repurposing evidence."
        )

    def enrich(self, candidate: DrugCandidate, target: str, disease: str,
               strategy: str) -> DrugCandidate:
        """
        Enrich candidate with literature evidence using PROMPT CHAINING.

        TODO 4: Implement a two-step prompt chain:

        CHAIN STEP 1 - Search PubMed:

            a) Get searchable names using get_searchable_names(candidate)
               - This returns valid names (excludes ChEMBL IDs)
               - If no valid names, set literature_summary and return early

            b) Build search queries based on the routing strategy:
               - "target_focused": '"{drug}"[Title/Abstract] AND "{target}"[Title/Abstract]'
               - "disease_focused": '"{drug}"[Title/Abstract] AND "{disease}"[Title/Abstract]'
               - "broad_search": try both target and disease queries

            c) Execute queries with query_pubmed(query_string, max_results=3)
               - Collect articles, avoiding duplicates (track seen PMIDs)
               - Stop when you have MAX_PUBMED_ARTICLES_PER_CANDIDATE articles
               - Limit to MAX_PUBMED_QUERIES_PER_CANDIDATE queries

        CHAIN STEP 2 - Assess with LLM:

            d) If articles were found, call self._assess_articles(articles, drug, target, disease)
               - This chains the search results into LLM analysis

            e) Update candidate fields from assessment:
               - candidate.literature_support_level = assessment["support_level"]
               - candidate.literature_strength = assessment["strength"]
               - candidate.literature_summary = assessment["summary"]
               - candidate.literature_supporting_pmids = assessment["supporting_pmids"]

            f) Also update provenance:
               - candidate.pubmed_articles = list of articles found
               - candidate.provenance.pubmed_pmids = [a["pmid"] for a in articles]
               - candidate.provenance.pubmed_queries_tried = list of queries used

        HINTS:
        - Use a set() to track seen_pmids for deduplication
        - Primary drug name is search_names[0] if search_names is not empty
        """

        # =====================================================================
        # TODO 4: Implement prompt chaining
        # =====================================================================

        # Get searchable names (excludes invalid names like ChEMBL IDs)
        search_names = get_searchable_names(candidate)

        if not search_names:
            candidate.literature_summary = f"No searchable name for {candidate.chembl_id or candidate.opentargets_id}"
            candidate.literature_support_level = "none"
            candidate.literature_strength = 0
            return candidate

        # CHAIN STEP 1: Build queries and search PubMed
        # queries = []
        # Build queries based on strategy...
        # Search and collect articles...

        # CHAIN STEP 2: Assess articles with LLM
        # if articles_found:
        #     assessment = self._assess_articles(...)
        #     Update candidate fields from assessment...

        # =====================================================================
        # END TODO 4
        # =====================================================================

        return candidate

    def _assess_articles(self, articles: List[Dict], drug: str, target: str,
                         disease: str) -> Dict:
        """
        Chain Step 2: LLM assessment of article relevance.

        TODO 5: Write a prompt that asks the LLM to assess the literature evidence.

        Your prompt should:
        - Provide context: drug name, target gene, disease
        - Include article information (already formatted below as articles_text)
        - Ask for assessment of whether articles support repurposing the drug

        Request JSON output with:
        - support_level: "none", "weak", "moderate", or "strong"
        - strength: 0-10 numeric score
        - supporting_pmids: list of PMIDs that actually support the hypothesis
        - summary: 2-3 sentence assessment of the evidence

        Include guidelines:
        - Be conservative - only "strong" if there's direct experimental evidence
        - Only include PMIDs that genuinely support the repurposing hypothesis
        - "none" if articles are unrelated to the drug-target-disease connection
        """

        # Format articles for the prompt (PROVIDED)
        articles_text = "\n".join([
            f"PMID:{a.get('pmid')} - {a.get('title', '')}\nAbstract: {(a.get('abstract', '') or '')[:400]}..."
            for a in articles[:4]
        ])

        # =====================================================================
        # TODO 5: Write your assessment prompt here
        # =====================================================================
        prompt = f"""
        # YOUR PROMPT HERE
        # Context: drug={drug}, target={target}, disease={disease}
        # Include articles_text (provided above)
        # Request: support_level, strength, supporting_pmids, summary
        # Include assessment guidelines

        ARTICLES:
        {articles_text}
        """

        fallback = {
            "support_level": "none",
            "strength": 0,
            "supporting_pmids": [],
            "summary": "Assessment unavailable."
        }

        return self._call_llm_json(prompt, fallback)
        # =====================================================================
        # END TODO 5
        # =====================================================================


class SafetyAgent(BaseAgent):
    """Evaluates drug safety profiles. (PROVIDED - DO NOT MODIFY)"""

    def __init__(self):
        super().__init__("SafetyAgent", "You evaluate drug safety profiles.")

    def evaluate(self, candidate: DrugCandidate, strategy: str) -> DrugCandidate:
        """Evaluate safety based on routing strategy."""
        search_names = get_searchable_names(candidate)

        if not search_names:
            candidate.safety_summary = f"No safety data (no searchable name)"
            return candidate

        # Query FAERS - try multiple names
        faers = {"total_events": 0, "serious_events": 0}
        for name in search_names:
            faers = query_faers(name)
            if faers.get("total_events", 0) > 0:
                candidate.provenance.faers_query_matched = name
                break

        candidate.faers_event_count = faers.get("total_events", 0)
        candidate.faers_serious_count = faers.get("serious_events", 0)

        # Query FDA labels for comprehensive strategy
        if strategy == "comprehensive":
            for name in search_names:
                label = query_fda_labels(name)
                if label.get("label_found"):
                    candidate.has_boxed_warning = bool(label.get("boxed_warning"))
                    candidate.label_found = True
                    break

        # Generate flags
        candidate.safety_flags = []
        if candidate.has_boxed_warning:
            candidate.safety_flags.append("BOXED_WARNING")
        if candidate.faers_event_count > 0 and candidate.faers_serious_count / max(1,
                                                                                   candidate.faers_event_count) > 0.5:
            candidate.safety_flags.append("HIGH_SERIOUS_RATIO")

        candidate.safety_summary = self._generate_summary(candidate)
        return candidate

    def _generate_summary(self, c: DrugCandidate) -> str:
        """Generate safety summary with LLM."""
        data = []
        if c.faers_event_count > 0:
            ratio = c.faers_serious_count / max(1, c.faers_event_count)
            data.append(f"FAERS: {c.faers_event_count} events, {ratio:.1%} serious")
        else:
            data.append("FAERS: No adverse event reports (may indicate limited market exposure)")
        if c.label_found:
            data.append(f"FDA label found, boxed warning: {'Yes' if c.has_boxed_warning else 'No'}")
        prompt = f"Summarize safety for {c.name}. Data: {'; '.join(data)}. Flags: {c.safety_flags or 'None'}. Write 2 sentences."
        return self._call_llm(prompt)


class EvaluationAgent(BaseAgent):
    """
    Scores, ranks, and refines candidate rankings.

    LEARNING OBJECTIVE: Evaluator-optimizer pattern

    This pattern uses iterative refinement:
    1. Generate (score candidates)
    2. Evaluate (critique the rankings)
    3. Optimize (refine based on critique)
    4. Repeat until approved or max iterations

    IMPORTANT: Score adjustments from the optimizer PERSIST across iterations.
    We only do initial scoring once, then adjustments accumulate.
    """

    WEIGHTS = {
        "target_binding": 0.28,
        "literature_support": 0.22,
        "safety_profile": 0.22,
        "development_stage": 0.18,
        "disease_relevance": 0.10,
    }

    def __init__(self):
        super().__init__(
            "EvaluationAgent",
            "You evaluate and rank drug repurposing candidates using iterative refinement."
        )

    def score_and_rank(self, candidates: List[DrugCandidate], target: str,
                       disease: str, max_iterations: int = 2) -> Tuple[List[DrugCandidate], List[Dict]]:
        """
        Implement the EVALUATOR-OPTIMIZER pattern with persistent adjustments.

        TODO 6: Implement the evaluator-optimizer loop:

        IMPORTANT CONCEPT: Score adjustments must PERSIST across iterations.
        - Iteration 1: Calculate initial scores from data (call _score_candidates)
        - Iteration 2+: Keep adjusted component scores, only recalculate totals

        Loop structure (for each iteration up to max_iterations):

            1. SCORE/RECALCULATE:
               - If iteration == 0: Call self._score_candidates(enriched_candidates)
               - If iteration > 0: Just recalculate total_score from existing component scores
                 (This preserves optimizer adjustments from previous iteration)

            2. SORT: Sort candidates by total_score descending

            3. EVALUATE: Call self._critique_rankings(top_candidates, target, disease)
               - This is the "evaluator" - it reviews the current rankings

            4. CHECK APPROVAL:
               - If "appropriate" in critique.lower(): rankings approved, break loop

            5. OPTIMIZE: Call self._refine_rankings(candidates, critique)
               - This is the "optimizer" - it suggests and applies adjustments
               - Returns (candidates, list_of_adjustments)
               - These adjustments modify candidate.scores directly

            6. RECORD: Add iteration record to ranking_history:
               {
                   "iteration": iteration_number,
                   "rankings": [{"name": c.name, "score": c.total_score, "scores": c.scores} for top],
                   "critique": critique_text,
                   "adjustments": list_of_adjustments,
                   "approved": True/False
               }

        After loop: Call self._generate_rationales(top_candidates, target, disease)

        Return: (candidates, ranking_history)

        HINTS:
        - Filter to only enriched candidates: [c for c in candidates if c.is_enriched]
        - Take top TOP_N_DISPLAY for critique
        - Recalculate total: c.total_score = sum(c.scores.get(k,0) * w for k,w in WEIGHTS.items())
        """
        logger.info("EvaluationAgent: Starting evaluator-optimizer loop...")

        # Filter to only enriched candidates
        enriched_candidates = [c for c in candidates if c.is_enriched]

        if not enriched_candidates:
            logger.warning("No enriched candidates to score!")
            return candidates, []

        ranking_history = []

        # =====================================================================
        # TODO 6: Implement evaluator-optimizer loop
        # =====================================================================

        # for iteration in range(max_iterations):
        #
        #     # 1. Score (first iteration) or recalculate totals (subsequent)
        #
        #     # 2. Sort by total_score
        #
        #     # 3. Critique rankings (Evaluator)
        #
        #     # 4. Check if approved
        #
        #     # 5. Refine rankings (Optimizer)
        #
        #     # 6. Record iteration

        # Generate rationales for top candidates
        # self._generate_rationales(enriched_candidates[:TOP_N_DISPLAY], target, disease)

        # =====================================================================
        # END TODO 6
        # =====================================================================

        return enriched_candidates, ranking_history

    def _score_candidates(self, candidates: List[DrugCandidate]) -> List[DrugCandidate]:
        """
        Apply initial scoring criteria to all candidates.

        TODO 7: Implement scoring logic for each component:

        target_binding (based on pchembl_value - higher is better binding):
            - >= 8.0: score 10 (excellent binding)
            - >= 7.0: score 8  (good binding)
            - >= 6.0: score 6  (moderate binding)
            - > 0:    score 4  (weak binding)
            - else:   score 2  (no data)

        literature_support:
            - Use candidate.literature_strength directly (already 0-10)
            - Clamp to range [0, 10] with min/max

        safety_profile (lower risk = higher score):
            - "BOXED_WARNING" in safety_flags: score 3 (high risk)
            - "HIGH_SERIOUS_RATIO" in safety_flags: score 5 (moderate risk)
            - faers_event_count > 0: score 7 (has data, no major flags)
            - else: score 5 (unknown - neutral)

        development_stage (based on max_phase):
            - Phase 4: 10, Phase 3: 8, Phase 2: 6, Phase 1: 4, Phase 0: 2

        disease_relevance:
            - Multiply disease_relevance_score by 10 (it's 0-1 scale)

        Calculate total_score as weighted sum:
            total = sum(scores[component] * WEIGHTS[component] for each component)

        Store scores in candidate.scores dict: candidate.scores["target_binding"] = value
        Store total in candidate.total_score
        """

        # =====================================================================
        # TODO 7: Implement scoring logic
        # =====================================================================
        for c in candidates:
            # target_binding score

            # literature_support score

            # safety_profile score

            # development_stage score

            # disease_relevance score

            # Calculate total_score
            pass
        # =====================================================================
        # END TODO 7
        # =====================================================================

        return candidates

    def _critique_rankings(self, top: List[DrugCandidate], target: str,
                           disease: str) -> str:
        """
        EVALUATOR: Critique current rankings.

        TODO 8: Write a prompt that critiques the current rankings.

        Your prompt should:
        - Show the target and disease context
        - Present current top candidates with key metrics (use the summary below)
        - Ask LLM to evaluate if rankings are reasonable and well-balanced

        Important instruction to include:
        - If rankings are good, respond with exactly: "Rankings are appropriate"
        - If issues exist, identify specific problems and suggest adjustments

        The "Rankings are appropriate" phrase triggers loop exit.
        """
        summary = "\n".join([
            f"{i + 1}. {c.name}: Score={c.total_score:.2f}, pChEMBL={c.pchembl_value}, "
            f"Phase={c.max_phase}, Lit={c.literature_support_level}, Flags={c.safety_flags or 'none'}"
            for i, c in enumerate(top)
        ])

        # =====================================================================
        # TODO 8: Write your critique prompt here
        # =====================================================================
        prompt = f"""
        # YOUR PROMPT HERE
        # Include: target, disease, current rankings (summary)
        # Ask for evaluation of whether rankings are reasonable
        # Specify: respond "Rankings are appropriate" if good

        CURRENT RANKINGS:
        {summary}
        """

        return self._call_llm(prompt)
        # =====================================================================
        # END TODO 8
        # =====================================================================

    def _refine_rankings(self, candidates: List[DrugCandidate],
                         critique: str) -> Tuple[List[DrugCandidate], List[Dict]]:
        """
        OPTIMIZER: Refine rankings based on critique.

        TODO 9: Write a prompt that suggests score adjustments, then apply them.

        Part A - Write prompt requesting adjustments as JSON array:
        [
            {
                "drug_name": "name of drug to adjust",
                "component": "target_binding|literature_support|safety_profile|development_stage|disease_relevance",
                "adjustment": -2 to +2,
                "reason": "brief justification"
            }
        ]

        Part B - Apply adjustments to candidates:
        - Build lookup dict: {name.lower(): candidate for each candidate}
        - For each adjustment:
            - Find candidate by name (case-insensitive)
            - Validate component is in WEIGHTS
            - Clamp adjustment to [-2, +2]
            - Apply: new_score = old_score + adjustment, clamped to [0, 10]
            - Update candidate.scores[component]
            - Recalculate candidate.total_score
            - Track applied adjustment with old/new scores

        Part C - Re-sort candidates by total_score descending

        Return: (candidates, list_of_applied_adjustments)

        HINTS:
        - Also try lookup with normalize_drug_name(drug_name)
        - Track: drug, component, adjustment, old_component_score, new_component_score, old_total, new_total
        """

        # =====================================================================
        # TODO 9: Write prompt and apply adjustments
        # =====================================================================
        prompt = f"""
        # YOUR PROMPT HERE
        # Provide the critique
        # Request JSON array of adjustments
        # Specify adjustment range [-2, +2]

        CRITIQUE:
        {critique}
        """

        adjustments = self._call_llm_json(prompt, [])
        if type(adjustments) == dict:
            adjustments = [adjustments]
        applied = []

        # Build lookup for finding candidates by name
        # lookup = ...

        # Apply each adjustment
        # for adj in adjustments:
        #     Find candidate, validate, apply, track...

        # Re-sort
        candidates.sort(key=lambda c: c.total_score, reverse=True)

        return candidates, applied
        # =====================================================================
        # END TODO 9
        # =====================================================================

    def _generate_rationales(self, candidates: List[DrugCandidate], target: str,
                             disease: str):
        """Generate ranking rationales for top candidates. (PROVIDED)"""
        for i, c in enumerate(candidates[:TOP_N_DISPLAY]):
            prompt = f"Write a 2-sentence rationale for why {c.name} ranks #{i + 1} for repurposing. Target: {target}, Disease: {disease}. Data: pChEMBL={c.pchembl_value}, Phase={c.max_phase}, MOA={c.mechanism_of_action or 'Unknown'}, Literature={c.literature_support_level}, Safety={c.safety_flags or 'None'}"
            c.ranking_rationale = self._call_llm(prompt)

    def generate_roadmap(self, candidates: List[DrugCandidate], target: str,
                         disease: str) -> str:
        """Generate validation roadmap. (PROVIDED)"""
        enriched = [c for c in candidates if c.is_enriched][:TOP_N_DISPLAY]
        if not enriched:
            return "No enriched candidates available for roadmap."
        top = "\n".join([f"- {c.name} (Score: {c.total_score:.2f}, Phase: {c.max_phase})" for c in enriched])
        prompt = f"Create a 3-month validation roadmap for repurposing drugs for {target} in {disease}.\n\nTOP CANDIDATES:\n{top}\n\nInclude: Executive Summary, Month 1 (target engagement), Month 2 (disease models), Month 3 (go/no-go). Note data limitations."
        return self._call_llm(prompt)


class DrugRepurposingOrchestrator:
    """
    Main orchestrator that coordinates all agents.

    LEARNING OBJECTIVE: Multi-agent orchestration

    The orchestrator:
    1. Creates a plan using ActionPlanningAgent
    2. Executes each step by delegating to appropriate agents
    3. Uses RoutingAgent to determine processing strategies
    4. Aggregates results into final output
    """

    def __init__(self):
        self.planner = ActionPlanningAgent()
        self.router = RoutingAgent()
        self.data_miner = DataMiningAgent()
        self.literature = LiteratureAgent()
        self.safety = SafetyAgent()
        self.evaluator = EvaluationAgent()
        self.candidates: List[DrugCandidate] = []
        self.target: str = ""
        self.disease: str = ""
        self.audit = AuditLog()

    def run(self, target: str, disease: str,
            max_candidates: int = MAX_CANDIDATES_TO_ENRICH) -> RepurposingResult:
        """Execute the full drug repurposing workflow."""
        reset_llm_call_count()
        self.target, self.disease = target, disease
        self.candidates = []
        self.audit = AuditLog(started_at=now_iso(), target=target, disease=disease, mock_mode=USE_MOCK_DATA)

        print("\n" + "=" * 60)
        print("DRUG REPURPOSING AGENTIC WORKFLOW")
        print(f"Target: {target} | Disease: {disease}")
        print(f"Mock mode: {USE_MOCK_DATA}")
        print("=" * 60)

        # Create plan
        print("\n[Planning] Creating action plan...")
        plan = self.planner.create_plan(target, disease)
        self.audit.plan_generated = {"objective": plan.objective, "steps": plan.steps}
        print(f"  Objective: {plan.objective}")
        print(f"  Steps: {[s['task_type'] for s in plan.steps]}")

        ranking_history = []
        roadmap = ""

        # Execute steps
        for step in plan.steps:
            task_type = step.get("task_type", "")
            step_num = step.get("step", "?")
            print(f"\n[Step {step_num}] {step.get('action')}")

            step_start = now_iso()
            result = self._execute_step(task_type, max_candidates)

            self.audit.steps_executed.append({
                "step": step_num,
                "task_type": task_type,
                "started_at": step_start,
                "completed_at": now_iso()
            })

            if task_type == "scoring" and isinstance(result, tuple):
                self.candidates, ranking_history = result
            elif task_type == "roadmap" and isinstance(result, str):
                roadmap = result

        self.audit.completed_at = now_iso()
        self.audit.candidates_found = len(self.candidates)
        self.audit.candidates_enriched = sum(1 for c in self.candidates if c.is_enriched)
        self.audit.llm_calls = get_llm_call_count()

        print("\n" + "=" * 60)
        print(
            f"COMPLETE | LLM calls: {self.audit.llm_calls} | Candidates: {len(self.candidates)} ({self.audit.candidates_enriched} enriched)")
        print("=" * 60)

        return RepurposingResult(
            target=target, disease=disease, timestamp=now_iso(),
            action_plan=plan, candidates=self.candidates[:max_candidates],
            validation_roadmap=roadmap, ranking_history=ranking_history,
            audit_log=self.audit
        )

    def _execute_step(self, task_type: str, max_candidates: int) -> Any:
        """
        Execute a workflow step by delegating to the appropriate agent(s).

        TODO 10: Implement step execution with proper agent delegation.

        Handle each task_type:

        "data_mining":
            - Call self.data_miner.run(self.target, self.disease)
            - Store result in self.candidates
            - Return candidates

        "enrichment":
            - Take top max_candidates from self.candidates
            - For each candidate:
                1. Get routing decision: self.router.route_enrichment_strategy(candidate, target, disease)
                2. Record routing in audit: self.audit.routing_decisions.append({...})
                3. Store strategies on candidate:
                   - candidate.literature_strategy = routing.inputs["literature_strategy"]
                   - candidate.safety_strategy = routing.inputs["safety_strategy"]
                4. Enrich with literature: self.literature.enrich(candidate, target, disease, lit_strategy)
                5. Evaluate safety: self.safety.evaluate(candidate, safety_strategy)
                6. Mark as enriched: candidate.is_enriched = True
                7. Print progress: print(f"    ✓ {candidate.name} ...")
            - Return candidates

        "scoring":
            - Call self.evaluator.score_and_rank(self.candidates, target, disease)
            - Returns (candidates, ranking_history)

        "roadmap":
            - Call self.evaluator.generate_roadmap(self.candidates, target, disease)
            - Returns roadmap string

        HINTS:
        - routing.inputs contains "literature_strategy" and "safety_strategy"
        - routing.decision_id can be stored for audit trail
        """

        # =====================================================================
        # TODO 10: Implement step execution
        # =====================================================================

        if task_type == "data_mining":
            # Mine databases
            pass

        elif task_type == "enrichment":
            # Enrich top candidates with routing
            top = self.candidates[:max_candidates]
            print(f"  Enriching top {len(top)} candidates...")

            # for c in top:
            #     1. Get routing decision
            #     2. Record in audit
            #     3. Store strategies on candidate
            #     4. Literature enrichment
            #     5. Safety evaluation
            #     6. Mark enriched
            #     7. Print progress
            pass

        elif task_type == "scoring":
            # Score and rank
            pass

        elif task_type == "roadmap":
            # Generate roadmap
            pass

        return None
        # =====================================================================
        # END TODO 10
        # =====================================================================


# =============================================================================
# OUTPUT FUNCTIONS (PROVIDED - DO NOT MODIFY)
# =============================================================================

def print_results(result: RepurposingResult):
    print("\n" + "=" * 60)
    print("TOP CANDIDATES")
    print("=" * 60)

    enriched = [c for c in result.candidates if c.is_enriched][:TOP_N_DISPLAY]

    if not enriched:
        print("  No enriched candidates to display.")
    else:
        for i, c in enumerate(enriched, 1):
            print(f"\n{i}. {c.name} ({c.chembl_id or c.opentargets_id})")
            print(f"   Score: {c.total_score:.2f} | Phase: {c.max_phase} | pChEMBL: {c.pchembl_value}")
            print(f"   Scores: {c.scores}")
            print(f"   Literature: {c.literature_support_level} | Safety: {c.safety_flags or 'None'}")
            if c.ranking_rationale:
                print(f"   Rationale: {c.ranking_rationale[:120]}...")

    # Show evaluator-optimizer history
    if result.ranking_history:
        print("\n" + "-" * 60)
        print("EVALUATOR-OPTIMIZER HISTORY")
        print("-" * 60)
        for record in result.ranking_history:
            iteration = record.get("iteration", "?")
            approved = record.get("approved", False)
            print(f"\n  Iteration {iteration}:")
            rankings = record.get("rankings", [])
            if rankings:
                print(f"    Rankings: {[r['name'] + ':' + str(round(r['score'], 2)) for r in rankings]}")
            adjustments = record.get("adjustments", [])
            if adjustments and adjustments[0].get("action") != "approved":
                print(f"    Adjustments: {len(adjustments)} applied")
            if approved:
                print("    Status: APPROVED ✓")

    print("\n" + "=" * 60)
    print("VALIDATION ROADMAP")
    print("=" * 60)
    print(result.validation_roadmap)


def export_results(result: RepurposingResult, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    base = f"{slugify(result.target)}_{slugify(result.disease)}"

    rows = [{
        "rank": i, "name": c.name, "chembl_id": c.chembl_id,
        "is_enriched": c.is_enriched,
        "total_score": round(c.total_score, 2) if c.is_enriched else None,
        "pchembl_value": c.pchembl_value, "max_phase": c.max_phase,
        "literature_support": c.literature_support_level,
        "safety_flags": ";".join(c.safety_flags),
        "routing_literature": c.literature_strategy,
        "routing_safety": c.safety_strategy
    } for i, c in enumerate(result.candidates, 1)]

    csv_path = os.path.join(output_dir, f"candidates_{base}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    def serialize(obj):
        if hasattr(obj, "__dict__"):
            return {k: serialize(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, list):
            return [serialize(x) for x in obj]
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        return obj

    json_path = os.path.join(output_dir, f"audit_{base}.json")
    with open(json_path, "w") as f:
        json.dump({
            "audit_log": serialize(result.audit_log),
            "candidates": [serialize(c) for c in result.candidates],
            "ranking_history": result.ranking_history,
            "roadmap": result.validation_roadmap
        }, f, indent=2, default=str)
    print(f"Saved: {json_path}")


# =============================================================================
# MAIN (PROVIDED - DO NOT MODIFY)
# =============================================================================

def main():
    TARGET = os.getenv("TARGET", "TMEM97")
    DISEASE = os.getenv("DISEASE", "non-alcoholic steatohepatitis")

    orchestrator = DrugRepurposingOrchestrator()
    result = orchestrator.run(TARGET, DISEASE, max_candidates=MAX_CANDIDATES_TO_ENRICH)

    print_results(result)
    export_results(result)

    return result


if __name__ == "__main__":
    main()