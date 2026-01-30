"""
Minimal shared life‑sciences action space for agentic LLMs

This module provides a robust, production-grade set of tools for life sciences
research, designed for reliable use by LLM-based agents. It features a
domain-aware rate limiter, structured JSON outputs, and resilient error handling.

It exposes a compact set of tools covering:
- Literature retrieval (PubMed, arXiv, Scholar)
- Evidence extraction (PDF + web)
- Clinical/genetics reasoning (ClinVar, OMIM)
- Sequence analytics (NCBI BLAST)
- Clinical trial interrogation (ClinicalTrials.gov)
- Local pharmacovigilance + knowledge bases (FAERS CSVs, OMIM.csv, ClinVar TSV)
- A persistent REPL for quick data work (with optional Bash/R shebangs)

Nine core callables used by agents:
    query_pubmed, query_scholar, query_arxiv,
    extract_pdf_content, extract_url_content,
    query_clinvar, query_omim,
    blast_sequence,
    run_python_repl
Plus: query_clinicaltrials (live endpoint) and 3 local-KB helpers:
    query_local_clinvar, query_local_omim, query_local_faers

Dependencies (install what you need):
    pip install requests pandas beautifulsoup4 lxml biopython pypdf2 arxiv
Optional (improve extraction):
    pip install readability-lxml feedparser

Environment variables:
    ENTREZ_EMAIL          -> your.email@example.com  (required by NCBI E-utilities)
    NCBI_API_KEY          -> optional E-utilities key (higher rate limits)
    OMIM_API_KEY          -> required for live OMIM API
    ACTIONSPACE_USER_AGENT-> optional, sent to remote services where allowed

Local data (paths can be overridden per-call):
    ./data/clinvar_snapshot.tsv
    ./data/omim_catalog.csv
    ./data/faers/faers_*.csv

Note:
- Google Scholar scraping is inherently brittle; it’s included as a best‑effort backup.
"""
from __future__ import annotations

import os
import re
import sys
import glob
import json
import time
import math
import shutil
import logging
from io import BytesIO, StringIO
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

# --- Environment variables  -------------------------------------------------

from dotenv import load_dotenv
load_dotenv(dotenv_path="entrez.env")

# --- Per-domain rate limiter (polite API usage) -----------------------------
from urllib.parse import urlparse

class RateLimiter:
    def __init__(self, intervals: Dict[str, float]):
        self.intervals = intervals
        self.last: Dict[str, float] = {}

    def wait(self, key: str) -> None:
        interval = self.intervals.get(key, self.intervals.get("generic", 0.0))
        if interval <= 0:
            return
        now = time.monotonic()
        prev = self.last.get(key)
        if prev is not None:
            sleep_for = interval - (now - prev)
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.last[key] = time.monotonic()

# NCBI: 3 req/s without key (~0.34s), ~10 req/s with key (~0.1s)
_ncbi_interval = 0.1 if os.getenv("NCBI_API_KEY") else 0.34
_limiter = RateLimiter({
    "ncbi": _ncbi_interval,
    "omim": 0.25,     # ~4 req/s
    "ctgov": 0.20,    # ~5 req/s
    "scholar": 5.0,   # be extra gentle
    "arxiv": 0.30,    # ~3 req/s
    "generic": 0.20,  # safe default for others
})
_last_request_time = 0.0


# Optional libraries (import lazily and guard usage)
try:
    from Bio import Entrez
    from Bio.Blast import NCBIWWW, NCBIXML
except Exception:  # pragma: no cover
    Entrez = None  # type: ignore
    NCBIWWW = None  # type: ignore
    NCBIXML = None  # type: ignore

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

try:
    import PyPDF2
except Exception:  # pragma: no cover
    PyPDF2 = None  # type: ignore

try:
    import arxiv as arxiv_py
except Exception:  # pragma: no cover
    arxiv_py = None  # type: ignore

try:  # optional, nicer web extraction
    from readability import Document as ReadabilityDoc  # type: ignore
except Exception:  # pragma: no cover
    ReadabilityDoc = None  # type: ignore

# ---------------------------------------------------------------------------
# Configuration & HTTP utilities
# ---------------------------------------------------------------------------
USER_AGENT = os.getenv(
    "ACTIONSPACE_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
)

NCBI_EMAIL = os.getenv("ENTREZ_EMAIL")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")

if Entrez is not None:
    if not NCBI_EMAIL:
        logging.warning(
            "ENTREZ_EMAIL not set. Please export ENTREZ_EMAIL to comply with NCBI policy."
        )
    Entrez.email = NCBI_EMAIL or "anonymous@example.com"
    if NCBI_API_KEY:
        Entrez.api_key = NCBI_API_KEY

DEFAULT_TIMEOUT = 30

_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(max_retries=3)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)
_default_headers = {"User-Agent": USER_AGENT}


def _http_get(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    stream: bool = False,
) -> requests.Response:
    """HTTP GET with retries, timeouts, UA, and polite rate-limiting per domain."""
    def _rate_key_for_url(u: str) -> str:
        host = urlparse(u).netloc.lower()
        if host.endswith("ncbi.nlm.nih.gov"):
            return "ncbi"
        if host == "api.omim.org":
            return "omim"
        if host.endswith("clinicaltrials.gov"):
            return "ctgov"
        if host == "scholar.google.com":
            return "scholar"
        if host.endswith("arxiv.org"):
            return "arxiv"
        return "generic"

    _limiter.wait(_rate_key_for_url(url))
    h = dict(_default_headers)
    if headers:
        h.update(headers)
    resp = _session.get(url, params=params, headers=h, timeout=timeout, stream=stream)
    resp.raise_for_status()
    return resp



def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _entrez_call(method: str, /, **kwargs):
    """Rate-limited wrapper around Bio.Entrez calls."""
    if Entrez is None:
        raise ImportError("biopython is required for NCBI E-utilities")
    _limiter.wait("ncbi")
    fn = getattr(Entrez, method)
    return fn(**kwargs)

import re
from chembl_webresource_client.new_client import new_client

_CHEMBL_ID_RE = re.compile(r"^CHEMBL\d+$", re.IGNORECASE)

def _is_bad_name(name: str, chembl_id: str) -> bool:
    """True when the name is missing or just a ChEMBL identifier."""
    if not name:
        return True
    if name.strip().upper() == chembl_id.strip().upper():
        return True
    return bool(_CHEMBL_ID_RE.match(name.strip()))

def _extract_synonyms(mol_record: dict, limit: int = 10) -> list[str]:
    """Extract a small set of unique synonyms from a ChEMBL molecule record."""
    syns = []
    for s in (mol_record.get("molecule_synonyms") or []):
        val = (s.get("synonyms") or "").strip()
        if val:
            syns.append(val)

    # Deduplicate, preserve order
    seen = set()
    out = []
    for x in syns:
        xl = x.lower()
        if xl not in seen:
            seen.add(xl)
            out.append(x)
        if len(out) >= limit:
            break
    return out

def _hydrate_compound_names(compounds: list[dict]) -> list[dict]:
    """
    Ensure each compound has a searchable name and a synonyms list.
    This enables PubMed/FAERS searches later in the pipeline.
    """
    mol = new_client.molecule

    for row in compounds:
        chembl_id = row.get("molecule_id", "")
        if not chembl_id:
            continue

        row.setdefault("synonyms", [])

        if _is_bad_name(row.get("name", ""), chembl_id) or not row["synonyms"]:
            try:
                m = mol.get(chembl_id) or {}
            except Exception:
                m = {}

            # Prefer a real name when available
            pref = (m.get("pref_name") or "").strip()
            if pref and not _CHEMBL_ID_RE.match(pref):
                row["name"] = pref
            else:
                row["name"] = row.get("name") or chembl_id

            # Add synonyms (keep any you already had)
            extra_syns = _extract_synonyms(m, limit=10)
            if extra_syns:
                # merge + dedupe
                merged = (row.get("synonyms") or []) + extra_syns
                seen = set()
                deduped = []
                for s in merged:
                    sl = s.lower()
                    if sl not in seen:
                        seen.add(sl)
                        deduped.append(s)
                row["synonyms"] = deduped[:10]

            # Optional: fill these if your original query_chembl sometimes lacks them
            if row.get("molecule_type") in (None, "", "Unknown"):
                mt = (m.get("molecule_type") or "").strip()
                if mt:
                    row["molecule_type"] = mt
            if not row.get("max_phase"):
                try:
                    row["max_phase"] = int(m.get("max_phase") or 0)
                except Exception:
                    row["max_phase"] = row.get("max_phase", 0)

    return compounds

def _is_searchable_name(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    if len(s) < 3:
        return False
    return not _CHEMBL_ID_RE.match(s)

def _get_synonyms_for(chembl_id: str, limit: int = 10) -> list[str]:
    mol = new_client.molecule
    try:
        m = mol.get(chembl_id) or {}
    except Exception:
        return []

    syns = []
    for row in (m.get("molecule_synonyms") or []):
        v = (row.get("synonyms") or "").strip()
        if v and v.lower() not in {x.lower() for x in syns}:
            syns.append(v)
        if len(syns) >= limit:
            break
    return syns

# ---------------------------------------------------------------------------
# 1) Literature retrieval & project dependencies
# ---------------------------------------------------------------------------

def query_pubmed(
    query: str,
    max_results: int = 20,
    include_mesh: bool = True,
    include_citations: bool = False,
) -> List[Dict[str, Any]]:
    """Search PubMed and return rich metadata.

    Returns a list of dicts with keys: pmid, title, abstract, journal, year,
    authors, doi, pmcid, mesh_terms (optional), citations (optional: counts + ids).
    """
    if Entrez is None:
        raise ImportError("biopython is required: pip install biopython")

    # ESearch: get IDs
    handle = _entrez_call("esearch", db="pubmed", term=query, retmax=max_results)
    search = Entrez.read(handle)
    handle.close()
    id_list = search.get("IdList", [])
    if not id_list:
        return []

    # EFetch XML for rich fields
    handle = _entrez_call("efetch", db="pubmed", id=",".join(id_list), retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    def _extract_year(article_dict: Dict[str, Any]) -> Optional[int]:
        """Robust year extraction across common PubMed XML layouts."""
        med = article_dict.get("MedlineCitation", {}) or {}
        art = med.get("Article", {}) or {}

        # 1) ArticleDate (often for online-first)
        try:
            ad = art.get("ArticleDate")
            if isinstance(ad, list) and ad:
                y = ad[0].get("Year")
                if y:
                    return int(y)
        except Exception:
            pass

        # 2) Journal PubDate Year
        try:
            ji = (art.get("Journal") or {}).get("JournalIssue") or {}
            pd = ji.get("PubDate") or {}
            y = pd.get("Year")
            if y:
                return int(y)
            # 2b) MedlineDate like "2003 Jan-Feb" or "1998 Spring"
            md = pd.get("MedlineDate")
            if md:
                m = re.search(r"(\d{4})", str(md))
                if m:
                    return int(m.group(1))
        except Exception:
            pass

        # 3) MedlineCitation Date* fallbacks
        for key in ("DateCompleted", "DateRevised", "DateCreated"):
            try:
                y = (med.get(key) or {}).get("Year")
                if y:
                    return int(y)
            except Exception:
                continue

        # 4) PubmedData History (PubStatus timestamps)
        try:
            hist = (article_dict.get("PubmedData") or {}).get("History") or []
            # Prefer 'pubmed'/'entrez'/'medline' statuses if available
            preferred_order = ("pubmed", "entrez", "medline", "received", "accepted", "revised", "aheadofprint")
            # Build a map of status->year
            status_years = {}
            for d in hist:
                status = str(d.get("PubStatus", "")).lower()
                y = d.get("Year")
                if y:
                    status_years[status] = int(y)
            for s in preferred_order:
                if s in status_years:
                    return status_years[s]
            # Otherwise return any year present
            if status_years:
                return sorted(status_years.values())[0]
        except Exception:
            pass

        return None

    articles: List[Dict[str, Any]] = []
    for article in records.get("PubmedArticle", []):
        med = article.get("MedlineCitation", {})
        art = med.get("Article", {})

        pmid = str(med.get("PMID", ""))
        title = art.get("ArticleTitle", "")
        abstract = " ".join(
            [p for p in (art.get("Abstract", {}) or {}).get("AbstractText", [])]
        )
        journal = ((art.get("Journal", {}) or {}).get("Title") or "")

        year = _extract_year(article)

        # Authors
        authors = []
        for au in (art.get("AuthorList") or []):
            last = au.get("LastName") or ""
            fore = au.get("ForeName") or ""
            collective = au.get("CollectiveName")
            name = collective or (f"{fore} {last}".strip())
            if name:
                authors.append(name)

        # Identifiers
        doi = None
        pmcid = None
        for iden in (art.get("ELocationID") or []):
            if getattr(iden, "attributes", {}).get("EIdType", "").lower() == "doi":
                doi = str(iden)
        for iden in (med.get("ArticleIdList") or []):
            if getattr(iden, "attributes", {}).get("IdType") == "doi":
                doi = str(iden)
            if getattr(iden, "attributes", {}).get("IdType") == "pmc":
                pmcid = str(iden)

        rec: Dict[str, Any] = {
            "pmid": pmid,
            "title": _clean_whitespace(str(title)),
            "abstract": _clean_whitespace(str(abstract)),
            "journal": _clean_whitespace(str(journal)),
            "year": year,
            "authors": authors,
            "doi": doi,
            "pmcid": pmcid,
        }

        if include_mesh:
            mesh_terms = []
            for mh in (med.get("MeshHeadingList") or []):
                desc = (mh.get("DescriptorName") or "")
                if desc:
                    mesh_terms.append(str(desc))
            rec["mesh_terms"] = mesh_terms

        articles.append(rec)

    if include_citations and id_list:
        # One ELink call to fetch "cited in" (who cites this paper)
        try:
            link = _entrez_call(
                "elink",
                dbfrom="pubmed",
                id=",".join(id_list),
                linkname="pubmed_pubmed_citedin",
            )
            link_rec = Entrez.read(link)
            link.close()
            cited_map: Dict[str, List[str]] = {}
            for item in link_rec:
                src = None
                if item.get("IdList"):
                    # First Id in IdList is the source
                    src_ids = [str(x) for x in item.get("IdList") if str(x).isdigit()]
                    src = src_ids[0] if src_ids else None
                tgt = []
                for ldb in item.get("LinkSetDb", []):
                    if ldb.get("Link"):
                        tgt.extend([str(x["Id"]) for x in ldb["Link"] if x.get("Id")])
                if src:
                    cited_map[src] = tgt
            for rec in articles:
                pmid = rec["pmid"]
                ids = cited_map.get(pmid, [])
                rec["citations"] = {"count": len(ids), "pmids": ids}
        except Exception:
            # Fail soft; citation metadata is optional
            pass

    return articles


def query_chembl(gene: str, max_compounds: int = 20) -> Tuple[List[Dict], Optional[str]]:
    """Find compounds in ChEMBL with activity against a human SINGLE PROTEIN target for `gene`.

    Returns:
      (compounds, target_chembl_id)

    Each compound dict contains:
      molecule_id, name, pchembl_value, max_phase, molecule_type, synonyms
    """
    use_mock = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
    if use_mock:
        fn = globals().get("get_mock_chembl")
        if callable(fn):
            return fn(gene), "CHEMBL_TARGET_MOCK"
        return [], None

    gene = (gene or "").strip()
    if not gene:
        return [], None

    def _is_searchable_name(name: Optional[str]) -> bool:
        if not name:
            return False
        s = str(name).strip()
        if not s:
            return False
        u = s.upper()
        if u.startswith("CHEMBL"):
            return False
        if u in {"UNKNOWN", "NONE", "N/A", "NA"}:
            return False
        return any(ch.isalpha() for ch in s)

    def _coerce_float(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    def _coerce_int(x: Any, default: int = 0) -> int:
        try:
            if x is None:
                return default
            return int(float(x))
        except Exception:
            return default

    def _extract_synonyms_from_mol(mol: Dict[str, Any], limit: int = 10) -> List[str]:
        syns: List[str] = []
        for item in (mol.get("molecule_synonyms") or []):
            if not isinstance(item, dict):
                continue
            s = (item.get("molecule_synonym") or "").strip()
            if s and s not in syns:
                syns.append(s)
            if len(syns) >= limit:
                break
        return syns

    def _pick_name(mol: Dict[str, Any], synonyms: List[str], fallback: str) -> str:
        pref = (mol.get("pref_name") or "").strip()
        if _is_searchable_name(pref):
            return pref
        for s in synonyms:
            if _is_searchable_name(s):
                return s
        return fallback

    def _pick_molecule_type(mol: Dict[str, Any]) -> str:
        mt = (mol.get("molecule_type") or "").strip()
        if mt:
            return mt
        md = mol.get("molecule_dictionary") or {}
        mt2 = (md.get("molecule_type") or "").strip() if isinstance(md, dict) else ""
        return mt2 or "Unknown"

    try:
        from chembl_webresource_client.new_client import new_client  # type: ignore

        target_api = new_client.target
        activity_api = new_client.activity
        molecule_api = new_client.molecule

        # 1) Pick a human SINGLE PROTEIN target
        targets = list(target_api.search(gene))
        human_single = [
            t for t in targets
            if "homo sapiens" in str(t.get("organism", "")).lower()
            and str(t.get("target_type", "")).upper() == "SINGLE PROTEIN"
        ]
        if not human_single:
            return [], None

        gene_l = gene.lower()

        def _score_target(t: Dict[str, Any]) -> Tuple[int, int]:
            pref = str(t.get("pref_name") or "").lower()
            if pref == gene_l:
                return (3, -len(pref))
            if gene_l in pref or pref in gene_l:
                return (2, -len(pref))
            return (1, -len(pref))

        human_single.sort(key=_score_target, reverse=True)
        chembl_target_id = human_single[0].get("target_chembl_id")
        if not chembl_target_id:
            return [], None

        # 2) Get activities and keep best pChEMBL per molecule
        fetch_n = max(500, max_compounds * 50)
        acts = (
            activity_api
            .filter(target_chembl_id=chembl_target_id)
            .filter(assay_type="B")
            .filter(standard_flag=1)
            .filter(pchembl_value__isnull=False)
            .only(["molecule_chembl_id", "pchembl_value", "standard_relation"])
        )[:fetch_n]

        best_p: Dict[str, float] = {}
        for a in acts:
            mid = a.get("molecule_chembl_id")
            if not mid:
                continue
            rel = (a.get("standard_relation") or "").strip()
            if rel in {">", "<"}:
                continue
            p = _coerce_float(a.get("pchembl_value"))
            if p is None:
                continue
            if mid not in best_p or p > best_p[mid]:
                best_p[mid] = p

        if not best_p:
            return [], chembl_target_id

        mids_sorted = sorted(best_p.keys(), key=lambda m: best_p[m], reverse=True)

        # 3) Hydrate molecules with molecule_api.get(mid) to get max_phase (TOP-LEVEL)
        mol_cache: Dict[str, Dict[str, Any]] = {}
        compounds: List[Dict[str, Any]] = []

        max_to_hydrate = max(200, max_compounds * 20)
        for mid in mids_sorted[:max_to_hydrate]:
            if len(compounds) >= max_compounds:
                break

            if mid not in mol_cache:
                try:
                    mol = molecule_api.get(mid)
                except Exception:
                    mol = {}
                mol_cache[mid] = mol if isinstance(mol, dict) else {}

            mol = mol_cache[mid]

            synonyms = _extract_synonyms_from_mol(mol, limit=10)
            name = _pick_name(mol, synonyms, fallback=mid)

            if not _is_searchable_name(name) and not any(_is_searchable_name(s) for s in synonyms):
                continue

            # ✅ IMPORTANT: max_phase is top-level on molecule detail records (as your debug showed)
            max_phase = _coerce_int(mol.get("max_phase"), default=0)

            compounds.append({
                "molecule_id": mid,
                "name": name,
                "pchembl_value": float(best_p[mid]),
                "max_phase": max_phase,
                "molecule_type": _pick_molecule_type(mol),
                "synonyms": synonyms,
            })

        compounds.sort(key=lambda r: (r.get("pchembl_value", 0.0), r.get("max_phase", 0)), reverse=True)
        return compounds[:max_compounds], chembl_target_id

    except Exception as e:
        logging.getLogger(__name__).warning(f"ChEMBL error for gene={gene}: {type(e).__name__}: {e}")
        return [], None



def query_opentargets(gene: str) -> Dict:
    """Look up known drugs for a target using the Open Targets GraphQL API.

    Returns:
      {
        "target_id": <Ensembl target id or None>,
        "known_drugs": <list of drug evidence rows>
      }

    Each known_drugs row includes:
      drug{id,name}, mechanismOfAction, disease{name}, phase
    """
    use_mock = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
    if use_mock:
        fn = globals().get("get_mock_ot")
        if callable(fn):
            return fn(gene)
        return {"target_id": None, "known_drugs": []}

    gene = (gene or "").strip()
    if not gene:
        return {"target_id": None, "known_drugs": []}

    OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    def _post(query: str, variables: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            _limiter.wait("generic")
            resp = _session.post(
                OT_URL,
                headers={**_default_headers, "Content-Type": "application/json", "Accept": "application/json"},
                json={"query": query, "variables": variables},
                timeout=DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    # 1) Resolve the text query to a specific target id
    search_query = """
    query searchTarget($q: String!) {
      search(queryString: $q, entityNames: ["target"], page: {size: 3, index: 0}) {
        hits { id name }
      }
    }
    """
    data = _post(search_query, {"q": gene})
    hits = (data or {}).get("data", {}).get("search", {}).get("hits", []) or []
    if not hits:
        return {"target_id": None, "known_drugs": []}

    target_id = hits[0].get("id")
    if not target_id:
        return {"target_id": None, "known_drugs": []}

    # 2) Fetch "known drugs" rows for that target
    drug_query = """
    query targetDrugs($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        knownDrugs(size: 50) {
          rows {
            drug { id name }
            mechanismOfAction
            disease { name }
            phase
          }
        }
      }
    }
    """
    data2 = _post(drug_query, {"ensemblId": target_id})

    rows: List[Dict[str, Any]] = (
        (data2 or {})
        .get("data", {})
        .get("target", {})
        .get("knownDrugs", {})
        .get("rows", [])
        or []
    )

    return {"target_id": target_id, "known_drugs": rows}



# --- openFDA term hygiene -----------------------------------------------------

_SALT_WORDS = {
    "hydrochloride", "hcl", "sodium", "potassium", "calcium", "magnesium",
    "sulfate", "sulphate", "phosphate", "acetate", "tartrate", "citrate",
    "mesylate", "besylate", "maleate", "fumarate", "succinate", "tosylate",
    "bromide", "chloride", "iodide", "nitrate", "carbonate", "lactate",
    "hydrobromide", "hydroiodide",
}

def _normalize_fda_term(term: str) -> str:
    # Keep letters/numbers/spaces/hyphens; collapse whitespace; lowercase
    t = re.sub(r"[^\w\s\-]", " ", (term or ""))
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def _strip_salt_forms(term: str) -> str:
    t = _normalize_fda_term(term)
    if not t:
        return ""
    parts = [p for p in t.split() if p not in _SALT_WORDS]
    # Drop trailing single-letter junk (rare)
    parts = [p for p in parts if len(p) > 1]
    return " ".join(parts).strip()

def _is_low_quality_fda_term(term: str) -> bool:
    t = _normalize_fda_term(term)
    if not t:
        return True

    # Reject short single-token terms (e.g. "FMI")
    if " " not in t and len(t) < 5:
        return True

    # Reject code-like tokens (e.g. "bf-2649", "gsk1349572") for FAERS/labels
    # (they might exist in literature but are usually useless for openFDA name matching)
    if re.fullmatch(r"[a-z]{1,6}[\-\s]?\d{2,6}[a-z]{0,3}", t):
        return True

    # Must contain enough letters to be a "name"
    if sum(ch.isalpha() for ch in t) < 4 and " " not in t:
        return True

    return False

def _extract_openfda_names_from_event(sample_event: dict) -> set[str]:
    """Pull a normalized set of names from a single FAERS event record."""
    names: set[str] = set()
    patient = (sample_event or {}).get("patient", {}) or {}
    drugs = patient.get("drug", []) or []
    for d in drugs:
        of = (d.get("openfda") or {})
        for field in ("generic_name", "brand_name", "substance_name"):
            for v in (of.get(field) or []):
                nv = _normalize_fda_term(str(v))
                if nv:
                    names.add(nv)
        mp = d.get("medicinalproduct")
        if mp:
            nv = _normalize_fda_term(str(mp))
            if nv:
                names.add(nv)
    return names

def _extract_openfda_names_from_label(sample_label: dict) -> set[str]:
    names: set[str] = set()
    of = (sample_label or {}).get("openfda", {}) or {}
    for field in ("generic_name", "brand_name", "substance_name"):
        for v in (of.get(field) or []):
            nv = _normalize_fda_term(str(v))
            if nv:
                names.add(nv)
    return names


def query_faers(drug_name: str) -> Dict:
    """Summarize FDA FAERS adverse event reporting for a drug name (openFDA)."""
    use_mock = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
    if use_mock:
        fn = globals().get("get_mock_faers")
        if callable(fn):
            return fn(drug_name)
        return {"total_events": 0, "serious_events": 0, "top_reactions": []}

    base_url = "https://api.fda.gov/drug/event.json"
    result = {
        "total_events": 0,
        "serious_events": 0,
        "top_reactions": [],
        # helpful debugging fields (optional; safe to keep)
        "matched_term": "",
        "matched_query": "",
    }

    if not drug_name:
        return result

    # Prefer "base" over salt forms (e.g., "pitolisant hydrochloride" -> "pitolisant")
    terms_to_try = []
    base = _strip_salt_forms(drug_name)
    raw = _normalize_fda_term(drug_name)
    for t in (base, raw):
        if t and t not in terms_to_try:
            terms_to_try.append(t)

    # If everything is low-quality (e.g., acronym), bail out early
    terms_to_try = [t for t in terms_to_try if not _is_low_quality_fda_term(t)]
    if not terms_to_try:
        return result

    def _try(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            return _http_get(base_url, params=params).json()
        except Exception:
            return None

    # NOTE: keep "medicinalproduct" as last resort AND exact only (it is very noisy)
    def _build_search_fields(clean_term: str) -> list[str]:
        return [
            f'patient.drug.openfda.generic_name.exact:"{clean_term}"',
            f'patient.drug.openfda.brand_name.exact:"{clean_term}"',
            f'patient.drug.openfda.substance_name.exact:"{clean_term}"',
            f'patient.drug.openfda.generic_name:"{clean_term}"',
            f'patient.drug.openfda.brand_name:"{clean_term}"',
            f'patient.drug.openfda.substance_name:"{clean_term}"',
            # last resort (noisy)
            f'patient.drug.medicinalproduct.exact:"{clean_term}"',
        ]

    for term in terms_to_try:
        for search_query in _build_search_fields(term):
            # Grab a sample result (not just meta) so we can validate the match
            sample = _try({"search": search_query, "limit": 1})
            total = int((sample or {}).get("meta", {}).get("results", {}).get("total", 0) or 0)
            if total <= 0 or not (sample or {}).get("results"):
                continue

            # Validate: the returned record should actually include the searched term
            names = _extract_openfda_names_from_event(sample["results"][0])
            if term not in names:
                # likely a false positive (e.g. short code matching some other product)
                continue

            # Accept this query
            result["matched_term"] = term
            result["matched_query"] = search_query
            result["total_events"] = total

            serious = _try({"search": f"({search_query}) AND serious:1", "limit": 1})
            result["serious_events"] = int(
                (serious or {}).get("meta", {}).get("results", {}).get("total", 0) or 0
            )

            agg = _try(
                {
                    "search": search_query,
                    "count": "patient.reaction.reactionmeddrapt.exact",
                    "limit": 10,
                }
            )
            top = []
            for row in (agg or {}).get("results", []) or []:
                term_pt = row.get("term")
                cnt = row.get("count")
                if term_pt and cnt is not None:
                    top.append({"reaction": term_pt, "count": int(cnt)})
            result["top_reactions"] = top

            return result  # stop at first validated match

    return result


def query_fda_labels(drug_name: str) -> Dict:
    """Fetch boxed warning text from FDA drug labels (openFDA)."""
    use_mock = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
    if use_mock:
        fn = globals().get("get_mock_fda_label")
        if callable(fn):
            return fn(drug_name)
        return {"boxed_warning": "", "label_found": False, "matched_term": "", "matched_query": ""}

    base_url = "https://api.fda.gov/drug/label.json"
    result = {"boxed_warning": "", "label_found": False, "matched_term": "", "matched_query": ""}

    if not drug_name:
        return result

    terms_to_try = []
    base = _strip_salt_forms(drug_name)
    raw = _normalize_fda_term(drug_name)
    for t in (base, raw):
        if t and t not in terms_to_try:
            terms_to_try.append(t)

    terms_to_try = [t for t in terms_to_try if not _is_low_quality_fda_term(t)]
    if not terms_to_try:
        return result

    def _build_search_fields(clean_term: str) -> list[str]:
        return [
            f'openfda.generic_name.exact:"{clean_term}"',
            f'openfda.brand_name.exact:"{clean_term}"',
            f'openfda.substance_name.exact:"{clean_term}"',
            f'openfda.generic_name:"{clean_term}"',
            f'openfda.brand_name:"{clean_term}"',
            f'openfda.substance_name:"{clean_term}"',
        ]

    for term in terms_to_try:
        for search_query in _build_search_fields(term):
            try:
                data = _http_get(base_url, params={"search": search_query, "limit": 1}).json()
            except Exception:
                data = None

            if not (data and data.get("results")):
                continue

            label = data["results"][0]

            # Validate: openfda fields contain our term
            names = _extract_openfda_names_from_label(label)
            if term not in names:
                continue

            bw = " ".join(label.get("boxed_warning", []) or [])
            result["boxed_warning"] = _clean_whitespace(bw)[:500]
            result["label_found"] = True
            result["matched_term"] = term
            result["matched_query"] = search_query
            return result

    return result





def query_scholar(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Best‑effort Google Scholar scrape (fragile by nature).

    Returns list of {title, url, snippet}. If the layout changes, returns an
    error entry instead of raising.
    """
    if BeautifulSoup is None:
        raise ImportError("beautifulsoup4 and lxml are required for query_scholar")

    url = "https://scholar.google.com/scholar"
    params = {"q": query, "hl": "en"}
    try:
        r = _http_get(url, params=params)
        soup = BeautifulSoup(r.text, "lxml")
        out: List[Dict[str, Any]] = []
        for card in soup.select("div.gs_r.gs_or.gs_scl"):
            h3 = card.select_one("h3.gs_rt a")
            snip = card.select_one("div.gs_rs")
            if h3:
                out.append(
                    {
                        "title": h3.get_text(strip=True),
                        "url": h3.get("href"),
                        "snippet": (snip.get_text(" ", strip=True) if snip else ""),
                    }
                )
            if len(out) >= max_results:
                break
        return out or [{"error": "No results parsed (layout may have changed)."}]
    except Exception as e:
        return [{"error": f"Scholar fetch failed: {e}"}]


def query_arxiv(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search arXiv for preprints. Uses the `arxiv` package if available.

    Returns list of {title, authors, summary, pdf_url, published}.
    """
    if arxiv_py is None:
        raise ImportError("arxiv package not installed. pip install arxiv")
    _limiter.wait("arxiv")
    search = arxiv_py.Search(query=query, max_results=max_results,
                             sort_by=arxiv_py.SortCriterion.Relevance)
    client = getattr(arxiv_py, "Client", None)
    results_iter = client().results(search) if client else search.results()
    out = []
    for r in results_iter:
        published = getattr(r, "published", None)
        out.append({
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "summary": _clean_whitespace(r.summary or ""),
            "pdf_url": r.pdf_url,
            "published": published.isoformat() if hasattr(published, "isoformat") else str(published),
        })
    return out


# ---------------------------------------------------------------------------
# 2) Evidence extraction
# ---------------------------------------------------------------------------

def _find_doi(url: str):
    return re.findall(r'\b10\.\d{4,9}/[-.;()/:\w]+', url)

def extract_pdf_content(doi_or_url: str, *, max_pages: Optional[int] = None) -> Dict[str, Any]:
    """Download a PDF and return extracted plain text.

    Returns {text, meta:{pages, bytes}}. Attempts extraction even if the server
    omits a PDF content-type. Requires PyPDF2.
    """
    if PyPDF2 is None:
        raise ImportError("PyPDF2 is required: pip install pypdf2")

    try:
        email = os.environ["ENTREZ_EMAIL"]
    except Exception:
        email = NCBI_EMAIL

    if _find_doi(doi_or_url):
        doi = _find_doi(doi_or_url)[0]
        raw = None
        resp = _http_get(f"https://api.unpaywall.org/v2/{doi}?email={email}", stream=True)
        fields = resp.json()
        for loc in fields['oa_locations']:
            if 'url_for_pdf' in loc:
                try:
                    pdfresp = _http_get(loc["url_for_pdf"], stream=True)
                    raw = pdfresp.content
                    if raw:
                        break
                except Exception:
                    continue
    else:
        resp = _http_get(doi_or_url, stream=True)
        raw = resp.content
    with BytesIO(raw) as fh:
        reader = PyPDF2.PdfReader(fh)
        n_pages = len(reader.pages)
        pages_to_read = range(n_pages if max_pages is None else min(max_pages, n_pages))
        chunks: List[str] = []
        for i in pages_to_read:
            try:
                pg = reader.pages[i]
                txt = pg.extract_text() or ""
                chunks.append(txt)
            except Exception:
                continue
        text = _clean_whitespace("\n".join(chunks))
    result = {"text": text, "meta": {"pages": n_pages, "bytes": len(raw)}}
    return result


def extract_url_content(url: str) -> Dict[str, Any]:
    """Fetch a webpage and strip boilerplate.

    If readability-lxml is installed, use it; otherwise fall back to BeautifulSoup
    heuristics. Returns {text, title}.
    """
    if BeautifulSoup is None:
        raise ImportError("beautifulsoup4 and lxml are required for extract_url_content")

    r = _http_get(url)
    html = r.text
    title = None
    if ReadabilityDoc is not None:
        try:
            doc = ReadabilityDoc(html)
            title = doc.short_title()
            cleaned_html = doc.summary(html_partial=True)
            soup = BeautifulSoup(cleaned_html, "lxml")
            text = _clean_whitespace(soup.get_text(" "))
            return {"text": text, "title": title}
        except Exception:
            pass

    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.get_text(strip=True) if soup.title else None)
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    body = soup.body or soup
    paragraphs = [p.get_text(" ", strip=True) for p in body.find_all(["p", "h1", "h2", "h3", "li"]) if p.get_text(strip=True)]
    return {"text": _clean_whitespace("\n".join(paragraphs)), "title": title}


# ---------------------------------------------------------------------------
# 3) Clinical‑genetics reasoning
# ---------------------------------------------------------------------------

def query_clinvar(variant: str) -> Dict[str, Any]:
    """Retrieve ClinVar details for a variant query (rsID or HGVS).

    Fields: title, clinical_significance, review_status, conditions, gene,
    accessions (RCV/VCV), hgvs, pubmed_pmids, allele_frequencies (if available).
    Uses NCBI E-utilities (ClinVar XML) and attempts allele frequencies via
    Variation Services (for rsIDs).
    """
    if Entrez is None:
        raise ImportError("biopython is required: pip install biopython")

    # ESearch in ClinVar
    s = _entrez_call("esearch", db="clinvar", term=variant)
    s_rec = Entrez.read(s)
    s.close()
    ids = s_rec.get("IdList", [])
    if not ids:
        return {"error": "Variant not found in ClinVar."}

    # Use first match for simplicity (minimal action space)
    uid = ids[0]
    f = _entrez_call("efetch", db="clinvar", id=uid, retmode="xml")
    x = Entrez.read(f)
    f.close()

    # ClinVar XML is verbose; pull common fields
    # Handle different XML response formats
    if hasattr(x, 'get'):
        doc = x.get("ClinVarSet", x).get("ClinVarSet", x)  # tolerate shape
    else:
        # Handle ListElement responses - use esummary instead for cleaner parsing
        doc = None
    
    title = None
    significance = None
    review_status = None
    conditions: List[str] = []
    gene = None
    hgvs_list: List[str] = []
    pubmed_pmids: List[str] = []
    accessions: Dict[str, List[str]] = {"RCV": [], "VCV": []}

    # Walk nested structure defensively - use esummary with validation disabled
    try:
        # Use esummary for cleaner structured data
        summ = _entrez_call("esummary", db="clinvar", id=uid)
        summ_rec = Entrez.read(summ, validate=False)  # Disable DTD validation
        summ.close()
        
        if "DocumentSummarySet" in summ_rec:
            ds = summ_rec["DocumentSummarySet"]["DocumentSummary"][0]
            
            # Extract title
            title = ds.get("title")
            
            # Extract clinical significance - try multiple fields due to ClinVar schema changes
            significance = None
            review_status = None
            
            # Try germline_classification (newer format)
            germline = ds.get("germline_classification")
            if germline and isinstance(germline, dict):
                significance = germline.get("description") or germline.get("last_evaluated")
                review_status = germline.get("review_status")
            
            # Try clinical_significance (older format)
            if not significance:
                clin_sig = ds.get("clinical_significance")
                if isinstance(clin_sig, dict):
                    significance = clin_sig.get("description")
                    review_status = clin_sig.get("review_status")
                elif isinstance(clin_sig, str):
                    significance = clin_sig
            
            # Fallback to direct review_status field
            if not review_status:
                review_status = ds.get("review_status")
            
            # Extract conditions/traits - check both trait_set and germline_classification.trait_set
            conditions = []
            
            # Check top-level trait_set
            trait_set = ds.get("trait_set", [])
            if isinstance(trait_set, list):
                conditions.extend([t.get("trait_name") for t in trait_set if isinstance(t, dict) and t.get("trait_name")])
            
            # Check germline_classification.trait_set (newer location)
            if germline and isinstance(germline, dict):
                germline_traits = germline.get("trait_set", [])
                if isinstance(germline_traits, list):
                    conditions.extend([t.get("trait_name") for t in germline_traits if isinstance(t, dict) and t.get("trait_name")])
            
            # Remove duplicates
            conditions = list(set(filter(None, conditions)))
            
            # Extract genes
            genes = ds.get("genes", [])
            if isinstance(genes, list) and genes:
                gene_info = genes[0]
                if isinstance(gene_info, dict):
                    gene = gene_info.get("symbol")
                else:
                    gene = None
            else:
                gene = None
            
            # Extract accession
            acc = ds.get("accession")
            if acc:
                if str(acc).startswith("RCV"):
                    accessions["RCV"].append(str(acc))
                elif str(acc).startswith("VCV"):
                    accessions["VCV"].append(str(acc))
            
            # Extract HGVS expressions
            hgvs = ds.get("hgvs_expressions", {})
            if isinstance(hgvs, dict):
                for v in hgvs.values():
                    if isinstance(v, list):
                        hgvs_list.extend([str(i) for i in v])
                    elif v:
                        hgvs_list.append(str(v))
                        
    except Exception as e:
        # If esummary fails, significance will remain None
        pass

    # PubMed links via ELink
    try:
        el = _entrez_call("elink", dbfrom="clinvar", id=uid, linkname="clinvar_pubmed")
        elr = Entrez.read(el)
        el.close()
        for ls in elr:
            for db in ls.get("LinkSetDb", []):
                if db.get("DbTo") == "pubmed":
                    pubmed_pmids = [str(l["Id"]) for l in db.get("Link", [])]
    except Exception:
        pass

    # Extract allele frequencies from ClinVar variation_set (preferred) or Variation Services
    allele_freqs: Dict[str, float] = {}
    
    # Try extracting from ClinVar esummary first (more reliable)
    try:
        if "DocumentSummarySet" in summ_rec:
            ds = summ_rec["DocumentSummarySet"]["DocumentSummary"][0]
            variation_set = ds.get("variation_set", [])
            
            if isinstance(variation_set, list):
                for variation in variation_set:
                    if isinstance(variation, dict):
                        allele_freq_set = variation.get("allele_freq_set", [])
                        if isinstance(allele_freq_set, list):
                            for freq_data in allele_freq_set:
                                if isinstance(freq_data, dict):
                                    source = freq_data.get("source", "")
                                    value = freq_data.get("value", "")
                                    if value:
                                        try:
                                            freq_val = float(value)
                                            # Keep highest frequency across all sources/variants
                                            key = f"{source}_{variation.get('variation_name', 'unknown')}"
                                            allele_freqs[key] = max(freq_val, allele_freqs.get(key, 0.0))
                                        except (ValueError, TypeError):
                                            pass
    except Exception:
        pass
    
    # Fallback to Variation Services if no frequencies found and we have an rsID
    if not allele_freqs and re.match(r"^rs\d+$", variant, re.IGNORECASE):
        try:
            vs = _http_get(f"https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/{variant.lower()[2:]}")
            j = vs.json()
            # Aggregate observed allele frequencies if present
            for ann in j.get("primary_snapshot_data", {}).get("allele_annotations", []):
                for fset in ann.get("frequency", []):
                    for obs in fset.get("study_results", []):
                        alt = obs.get("allele" ) or obs.get("x_allele")
                        if "allele_freq" in obs:
                            freq_val = float(obs.get("allele_freq") or 0.0)
                            allele_freqs[f"vs_{alt}"] = max(freq_val, allele_freqs.get(f"vs_{alt}", 0.0))
        except Exception:
            pass

    return {
        "query": variant,
        "title": title,
        "clinical_significance": significance,
        "review_status": review_status,
        "conditions": conditions,
        "gene": gene,
        "accessions": accessions,
        "hgvs": sorted(set(hgvs_list)),
        "pubmed_pmids": pubmed_pmids,
        "allele_frequencies": allele_freqs or None,
    }


def query_omim(term: str, *, include: str = "geneMap,clinicalSynopsis,allelicVariant") -> Dict[str, Any]:
    """Query OMIM live API by MIM number or gene symbol.

    Requires OMIM_API_KEY. Returns a compact dict with gene symbol (if any),
    phenotype titles, inheritance, clinical synopsis highlights, allelic variants
    (IDs + simple names), and molecular mechanism if available.
    """
    api_key = os.getenv("OMIM_API_KEY")
    if not api_key:
        return {"error": "OMIM_API_KEY not set. Set it to use the live OMIM API."}

    base = "https://api.omim.org/api/entry"
    params = {"format": "json", "include": include, "apiKey": api_key}
    # Detect numeric MIM vs symbol/text
    if re.fullmatch(r"\d{6}", term):
        params["mimNumber"] = term
    else:
        params["search"] = term

    resp = _http_get(base, params=params)
    j = resp.json()
    entries = (j.get("omim", {}).get("entryList") or [])
    if not entries:
        return {"term": term, "results": []}

    results: List[Dict[str, Any]] = []
    for wrap in entries:
        e = wrap.get("entry", {})
        gene_symbols = []
        if e.get("geneMap") and e["geneMap"].get("geneMapMethods"):
            # prefer approved symbol if present
            sym = e["geneMap"].get("geneSymbols") or ""
            gene_symbols = [s.strip() for s in sym.split(",") if s.strip()]

        phenotypes = []
        for p in (e.get("geneMap", {}) or {}).get("phenotypeMapList", []) or []:
            phen = p.get("phenotypeMap", {})
            phenotypes.append(
                {
                    "phenotype": phen.get("phenotype") or phen.get("phenotypeMimNumber"),
                    "inheritance": phen.get("inheritance"),
                    "mim": phen.get("phenotypeMimNumber"),
                }
            )

        clin_syn = e.get("clinicalSynopsis") or {}
        mechanism = (e.get("textSectionList") or [{}])[0].get("textSection", {}).get("textSectionContent")

        allelic_variants = []
        for av in (e.get("allelicVariantList") or []):
            avx = av.get("allelicVariant", {})
            allelic_variants.append(
                {
                    "mim": avx.get("allelicVariantMimNumber"),
                    "name": avx.get("allelicVariantName"),
                    "dbsnps": avx.get("dbSnps"),
                }
            )

        results.append(
            {
                "mimNumber": e.get("mimNumber"),
                "titles": e.get("titles", {}),
                "gene_symbols": gene_symbols,
                "phenotypes": phenotypes,
                "inheritance": clin_syn.get("inheritance"),
                "clinicalSynopsis": {k: clin_syn.get(k) for k in ["inheritance", "phenotype", "molecularBasis"] if k in clin_syn},
                "molecular_mechanism": mechanism,
                "allelic_variants": allelic_variants,
            }
        )

    return {"term": term, "results": results}


# ---------------------------------------------------------------------------
# 4) Sequence analytics (BLAST)
# ---------------------------------------------------------------------------

def blast_sequence(
    sequence: str,
    *,
    program: str = "blastn",
    database: Optional[str] = None,
    hitlist_size: int = 10,
    entrez_query: Optional[str] = None,
    max_wait_s: int = 90,
    poll_interval_s: int = 15,
) -> Dict[str, Any]:
    """Remote BLAST via NCBI URL API with hard timeout + fail-soft JSON errors."""
    if not sequence or not re.fullmatch(r"[A-Za-z*\-\.\?\n\r]+", sequence):
        return {"error": "Invalid sequence string."}

    if database is None:
        database = "nt" if program.lower() == "blastn" else "nr"

    # URL API endpoint (Put/Get)
    blast_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

    # 1) Submit job (CMD=Put)
    params_put = {
        "CMD": "Put",
        "PROGRAM": program,
        "DATABASE": database,
        "QUERY": sequence,
        "HITLIST_SIZE": str(hitlist_size),
        "WORD_SIZE": 7,
        "EXPECT": 1000,
        "FILTER": "F",
        "FORMAT_TYPE": "XML",
        "TOOL": "ls_action_space",
    }
    if NCBI_EMAIL:
        params_put["EMAIL"] = NCBI_EMAIL
    if entrez_query:
        params_put["ENTREZ_QUERY"] = entrez_query
    # Speed hint for blastn (optional but helps a lot)
    if program.lower() == "blastn":
        params_put["MEGABLAST"] = "off"

    _limiter.wait("ncbi")
    r = _http_get(blast_url, params=params_put)
    txt = r.text

    m = re.search(r"RID = ([A-Z0-9]+)", txt)
    if not m:
        return {"error": "BLAST submission failed (no RID).", "response_snippet": txt[:300]}
    rid = m.group(1)

    # 2) Poll status (CMD=Get&FORMAT_OBJECT=SearchInfo)
    deadline = time.time() + max_wait_s
    status = None
    while time.time() < deadline:
        _limiter.wait("ncbi")
        s = _http_get(blast_url, params={"CMD": "Get", "RID": rid, "FORMAT_OBJECT": "SearchInfo"})
        stxt = s.text
        if "Status=READY" in stxt:
            status = "READY"
            break
        if "Status=FAILED" in stxt:
            return {"error": "BLAST job failed.", "rid": rid, "status": "FAILED"}
        if "Status=UNKNOWN" in stxt:
            return {"error": "BLAST job expired/unknown.", "rid": rid, "status": "UNKNOWN"}
        time.sleep(poll_interval_s)

    if status != "READY":
        return {"error": "BLAST timed out.", "rid": rid, "max_wait_s": max_wait_s}

    # 3) Fetch results (XML) and parse like before
    _limiter.wait("ncbi")
    xml_resp = _http_get(blast_url, params={"CMD": "Get", "RID": rid, "FORMAT_TYPE": "XML"})
    try:
        record = NCBIXML.read(BytesIO(xml_resp.content))
    except Exception as e:
        return {"error": f"BLAST XML parse failed: {type(e).__name__}: {e}", "rid": rid}

    hits = []
    for aln in getattr(record, "alignments", [])[:hitlist_size]:
        hsps = []
        for hsp in getattr(aln, "hsps", []):
            pct = (100.0 * hsp.identities / hsp.align_length) if getattr(hsp, "align_length", 0) else None
            hsps.append({
                "evalue": getattr(hsp, "expect", None),
                "score": getattr(hsp, "score", None),
                "identities": getattr(hsp, "identities", None),
                "align_len": getattr(hsp, "align_length", None),
                "pct_identity": pct,
            })
        hits.append({
            "hit_id": getattr(aln, "hit_id", None),
            "title": getattr(aln, "title", None),
            "length": getattr(aln, "length", None),
            "hsps": hsps,
        })

    return {"rid": rid, "query_length": getattr(record, "query_length", None), "hits": hits}



# ---------------------------------------------------------------------------
# 3b) NCBI Variation Services (open access; no keys required)
# ---------------------------------------------------------------------------

VARIATION_BASE = "https://api.ncbi.nlm.nih.gov/variation/v0"

def query_refsnp(rsid: str) -> Dict[str, Any]:
    """Fetch a dbSNP RefSNP JSON record (rsID) from NCBI Variation Services."""
    if not rsid:
        return {"error": "Empty rsID."}
    m = re.match(r"^rs(\d+)$", rsid.strip(), re.IGNORECASE)
    if not m:
        return {"error": "rsid must look like 'rs123'."}
    rid = m.group(1)
    _limiter.wait("ncbi")
    return _http_get(f"{VARIATION_BASE}/refsnp/{rid}").json()


def extract_refsnp_freqs(
    refsnp_json: Dict[str, Any],
    *,
    studies: Optional[Iterable[str]] = None
) -> List[Dict[str, Any]]:
    """Extract per-study allele frequencies from RefSNP JSON.

    Tolerates common schema variants from submitters:
      A) top-level allele_count / total_count (+ observation dict)
      B) study_results list with allele_freq or counts.
    """
    want = {s.lower() for s in studies} if studies else None
    out: List[Dict[str, Any]] = []

    annos = (refsnp_json.get("primary_snapshot_data") or {}).get("allele_annotations") or []
    for anno in annos:
        for fset in (anno.get("frequency") or []):
            study = (
                fset.get("study_name")
                or fset.get("study")
                or fset.get("source")
                or "unknown"
            )
            if want and study.lower() not in want:
                continue

            # Schema A: counts at top-level
            ac, tc = fset.get("allele_count"), fset.get("total_count")
            if ac is not None and tc:
                obs = fset.get("observation") or {}
                allele = obs.get("inserted_sequence") or obs.get("deleted_sequence")
                try:
                    out.append({
                        "study": study,
                        "allele": allele,
                        "af": float(ac) / float(tc),
                        "allele_count": int(ac),
                        "total_count": int(tc),
                    })
                except Exception:
                    pass
                continue

            # Schema B: per-population study_results
            sr = fset.get("study_results")
            if isinstance(sr, list):
                for res in sr:
                    allele = res.get("allele") or res.get("x_allele")
                    af = res.get("allele_freq")
                    if af is None:
                        ac = res.get("allele_count") or res.get("count")
                        tc = res.get("total_count") or res.get("allele_number") or res.get("an")
                        if ac is not None and tc:
                            try:
                                af = float(ac) / float(tc)
                            except Exception:
                                af = None
                    if af is not None:
                        out.append({
                            "study": study,
                            "population": res.get("population") or res.get("pop"),
                            "allele": allele,
                            "af": float(af),
                        })

    return out


def get_population_maf(
    rsid: str,
    *,
    prefer: Tuple[str, ...] = ("1000Genomes", "1000Genomes_30X", "ALFA")
) -> Optional[float]:
    """Convenience: return a MAF from preferred studies if present."""
    j = query_refsnp(rsid)
    if "error" in j:
        return None
    freqs = extract_refsnp_freqs(j, studies=prefer)
    if not freqs:
        return None
    afs = [f["af"] for f in freqs if isinstance(f.get("af"), (int, float))]
    return min(afs) if afs else None


def hgvs_to_contextuals(hgvs: str) -> Dict[str, Any]:
    """Convert HGVS -> contextual SPDI alleles (NCBI Overprecision Correction)."""
    if not hgvs:
        return {"error": "Empty HGVS."}
    _limiter.wait("ncbi")
    url = f"{VARIATION_BASE}/hgvs/{requests.utils.quote(hgvs, safe='')}/contextuals"
    return _http_get(url).json()


def vcf_to_contextuals(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    *,
    assembly: str = "GCF_000001405.40",  # GRCh38 default
) -> Dict[str, Any]:
    """Convert VCF coords -> contextual SPDI alleles (fail-soft)."""
    _limiter.wait("ncbi")
    url = f"{VARIATION_BASE}/vcf/{chrom}/{pos}/{ref}/{alt}/contextuals"
    try:
        return _http_get(url, params={"assembly": assembly}).json()
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        body = getattr(e.response, "text", "")[:300] if getattr(e, "response", None) else ""
        hint = (
            "Check that CHROM/POS/REF/ALT match the assembly. "
            "For example, vcf/11/5248224/A/AC is documented with assembly=GCF_000001405.25."
        )
        return {"error": f"Variation Services HTTP {status}", "hint": hint, "response_snippet": body}


# ---------------------------------------------------------------------------
# 5) ClinicalTrials.gov (live endpoint)
# ---------------------------------------------------------------------------

def query_clinicaltrials(
    expr: str,
    *,
    max_results: int = 50,
    fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Query ClinicalTrials.gov v2 API and project common study fields."""
    if fields is None:
        fields = [
            "NCTId",
            "BriefTitle",
            "OverallStatus",
            "StudyType",
            "Condition",
            "InterventionName",
            "PrimaryOutcomeMeasure",
            "EnrollmentCount",
            "StartDate",
            "CompletionDate",
            "Phase",
            "LocationCountry",
            "LeadSponsorName",
        ]

    base_url = "https://clinicaltrials.gov/api/v2/studies"
    page_size = max(1, min(max_results, 100))  # v2 returns paged results
    params = {
        "query.term": expr,         # v1 `expr` -> v2 `query.term`
        "format": "json",
        "pageSize": page_size,
        "countTotal": "true",       # include totalCount in response
    }

    def _project(s: Dict[str, Any]) -> Dict[str, Any]:
        ps = s.get("protocolSection", {})
        idm = ps.get("identificationModule", {}) or {}
        sm  = ps.get("statusModule", {}) or {}
        dm  = ps.get("designModule", {}) or {}
        cm  = ps.get("conditionsModule", {}) or {}
        aim = ps.get("armsInterventionsModule", {}) or {}
        om  = ps.get("outcomesModule", {}) or {}
        clm = ps.get("contactsLocationsModule", {}) or {}
        scm = ps.get("sponsorCollaboratorsModule", {}) or {}

        # helpers
        def _names(items, *keys):
            out = []
            for it in (items or []):
                if isinstance(it, dict):
                    for k in keys:
                        if it.get(k):
                            out.append(it[k])
                            break
            return out

        rec = {}
        for f in fields:
            if f == "NCTId":
                rec[f] = idm.get("nctId")
            elif f == "BriefTitle":
                rec[f] = idm.get("briefTitle")
            elif f == "OverallStatus":
                rec[f] = sm.get("overallStatus")
            elif f == "StudyType":
                rec[f] = dm.get("studyType")
            elif f == "Condition":
                rec[f] = cm.get("conditions")  # list[str]
            elif f == "InterventionName":
                rec[f] = _names(aim.get("interventions", []), "name", "interventionName")
            elif f == "PrimaryOutcomeMeasure":
                # v2 commonly uses 'name'; some records may still have 'measure'
                rec[f] = _names(om.get("primaryOutcomes", []), "name", "measure")
            elif f == "EnrollmentCount":
                rec[f] = (dm.get("enrollmentInfo") or {}).get("count")
            elif f == "StartDate":
                rec[f] = (sm.get("startDateStruct") or {}).get("date")
            elif f == "CompletionDate":
                rec[f] = (sm.get("completionDateStruct") or {}).get("date")
            elif f == "Phase":
                rec[f] = dm.get("phases") or dm.get("phase")
            elif f == "LocationCountry":
                rec[f] = sorted({
                    loc.get("country")
                    for loc in (clm.get("locations") or [])
                    if isinstance(loc, dict) and loc.get("country")
                })
            elif f == "LeadSponsorName":
                lead = scm.get("leadSponsor") or (scm.get("sponsors") or {}).get("leadSponsor")
                rec[f] = (lead or {}).get("name") or (lead or {}).get("fullName") or (lead or {}).get("agency")
            else:
                rec[f] = None
        return rec

    studies_out: List[Dict[str, Any]] = []
    next_token = None
    total_count = None

    while len(studies_out) < max_results:
        if next_token:
            params["pageToken"] = next_token
        r = _http_get(base_url, params=params)
        r.raise_for_status()
        payload = r.json() or {}
        if total_count is None:
            total_count = payload.get("totalCount")  # present when countTotal=true
        for s in payload.get("studies", []):
            studies_out.append(_project(s))
            if len(studies_out) >= max_results:
                break
        next_token = payload.get("nextPageToken")
        if not next_token:
            break

    return {"count": total_count, "studies": studies_out}



# ---------------------------------------------------------------------------
# 6) Persistent REPL (Python, with optional Bash/R via shebang)
# ---------------------------------------------------------------------------

_persistent_namespace: Dict[str, Any] = {}


def run_python_repl(code: str) -> Dict[str, Any]:
    """Execute Python code in a persistent namespace.

    Also supports **shebang** prefixes for quick Bash/R ops:
        #!bash\n<shell commands>
        #!r\n<single‑file R script>

    Returns {stdout, error}. CAUTION: This executes arbitrary code.
    """
    code = code.strip()

    # Strip Markdown fences if present
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", code).strip()

    # --- Bash mode ---
    if code.startswith("#!bash"):
        script = code.split("\n", 1)[1] if "\n" in code else ""
        return _run_shell(script)

    # --- R mode ---
    if re.match(r"^#!r(\b|\n)", code, flags=re.IGNORECASE):
        script = code.split("\n", 1)[1] if "\n" in code else ""
        return _run_r(script)

    # --- Python mode ---
    global _persistent_namespace
    stdout_buf = StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = stdout_buf
    sys.stderr = stdout_buf
    error: Optional[str] = None
    try:
        exec(code, _persistent_namespace)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return {"stdout": stdout_buf.getvalue(), "error": error}


def _run_shell(script: str) -> Dict[str, Any]:
    import subprocess, tempfile

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as tf:
        tf.write(script)
        path = tf.name
    try:
        proc = subprocess.run(["/bin/bash", path], capture_output=True, text=True)
        out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        return {"stdout": out, "error": None if proc.returncode == 0 else f"Exit {proc.returncode}"}
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def _run_r(script: str) -> Dict[str, Any]:
    import subprocess, tempfile

    if shutil.which("Rscript") is None:
        return {"stdout": "", "error": "Rscript not found in PATH."}
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".R") as tf:
        tf.write(script)
        path = tf.name
    try:
        proc = subprocess.run(["Rscript", path], capture_output=True, text=True)
        out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        return {"stdout": out, "error": None if proc.returncode == 0 else f"Exit {proc.returncode}"}
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 7) Local knowledge bases (CSV/TSV snapshots)
# ---------------------------------------------------------------------------

def query_local_clinvar(
    gene_symbol: str,
    tsv_path: str = "./data/clinvar_snapshot.tsv",
    *,
    cols: Optional[List[str]] = None,
    max_rows: int = 100,
) -> Dict[str, Any]:
    """Filter a local ClinVar TSV snapshot by gene symbol.

    Returns {rows:[...], columns:[...]}. Auto‑detects common column names.
    """
    if not os.path.exists(tsv_path):
        return {"error": f"ClinVar snapshot not found at '{tsv_path}'"}

    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    gene_cols = [c for c in df.columns if c.lower() in {"gene(s)", "genes", "genesymbol", "gene_symbol"}]
    if not gene_cols:
        return {"error": "Could not find gene symbol column in TSV."}
    gene_col = gene_cols[0]

    mask = df[gene_col].astype(str).str.contains(fr"\b{re.escape(gene_symbol)}\b", case=False, na=False)
    sub = df.loc[mask]
    if sub.empty:
        return {"rows": [], "columns": list(df.columns)}

    if cols is None:
        cols = [c for c in [
            "Name",
            gene_col,
            "Clinical significance (Last reviewed)",
            "ClinicalSignificance",
            "Review status",
            "ReviewStatus",
            "RCVaccession",
        ] if c in sub.columns]
        cols = cols or list(sub.columns)  # fallback

    out = sub[cols].head(max_rows)
    return {"rows": out.to_dict(orient="records"), "columns": list(out.columns)}


def query_local_omim(
    query: str,
    csv_path: str = "./data/omim_catalog.csv",
    *,
    max_rows: int = 200,
) -> Dict[str, Any]:
    """Filter a local OMIM CSV by gene symbol or phenotype keyword.

    Expects columns like 'Gene Symbols' and 'Phenotypes'. Returns {rows, columns}.
    """
    if not os.path.exists(csv_path):
        return {"error": f"OMIM catalog not found at '{csv_path}'"}

    df = pd.read_csv(csv_path, low_memory=False)
    cand_cols = {c.lower(): c for c in df.columns}
    gs = cand_cols.get("gene symbols") or cand_cols.get("gene_symbols")
    ph = cand_cols.get("phenotypes") or cand_cols.get("phenotype")
    if not gs or not ph:
        return {"error": "CSV must contain 'Gene Symbols' and 'Phenotypes' columns."}

    m = df[gs].astype(str).str.contains(query, case=False, na=False) | df[ph].astype(str).str.contains(query, case=False, na=False)
    sub = df.loc[m].head(max_rows)
    return {"rows": sub.to_dict(orient="records"), "columns": list(sub.columns)}


def query_local_faers(
    drug_name: str,
    csv_dir: str = "./data/faers",
    *,
    top_n: int = 20,
) -> Dict[str, Any]:
    """Summarize top MedDRA Preferred Terms for a drug from local FAERS CSVs.

    Expects per‑year CSVs with at least columns: 'drugname' and 'pt'. Returns
    {top_reactions:[{pt, count}], total_reports}.
    """
    if not os.path.isdir(csv_dir):
        return {"error": f"FAERS directory not found at '{csv_dir}'"}

    files = sorted(glob.glob(os.path.join(csv_dir, "faers_*.csv")))
    if not files:
        return {"error": f"No FAERS CSV files found in '{csv_dir}'"}

    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, usecols=["drugname", "pt"], on_bad_lines="skip", low_memory=False))
        except Exception:
            continue
    if not frames:
        return {"error": "No FAERS data could be read (check schema)."}

    df = pd.concat(frames, ignore_index=True)
    mask = df["drugname"].astype(str).str.contains(drug_name, case=False, na=False)
    sub = df.loc[mask]
    total = int(len(sub))
    if total == 0:
        return {"top_reactions": [], "total_reports": 0}

    counts = (
        sub["pt"].astype(str).str.strip().value_counts().head(top_n).reset_index()
    )
    counts.columns = ["pt", "count"]
    return {
        "top_reactions": counts.to_dict(orient="records"),
        "total_reports": total,
    }



def _build_params(base_params: Dict[str, Any]) -> Dict[str, Any]:
    """Add standard NCBI parameters."""
    params = dict(base_params)
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params

def _rate_limit():
    """Simple rate limiter for NCBI requests."""
    global _last_request_time
    now = time.monotonic()
    sleep_for = _ncbi_interval - (now - _last_request_time)
    if sleep_for > 0:
        time.sleep(sleep_for)
    _last_request_time = time.monotonic()


def query_ncbi_gene(term: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Query NCBI Gene database for gene information.

    Uses E-utilities (esearch + esummary) to retrieve gene data including:
    - Gene symbol and aliases
    - Full name and description
    - Summary (curated gene function description)
    - Chromosome location
    - Associated phenotypes and diseases (via MIM links)
    - Gene type and status

    Args:
        term: Gene symbol (e.g., "NPC1", "BRCA1") or gene name
        max_results: Maximum number of gene records to return (default 5)

    Returns:
        Dictionary with structure:
        {
            "term": str,
            "results": [
                {
                    "gene_id": str,
                    "symbol": str,
                    "name": str,
                    "aliases": [str],
                    "summary": str,
                    "chromosome": str,
                    "location": str,
                    "gene_type": str,
                    "mim_ids": [str],
                    "phenotypes": [
                        {
                            "phenotype": str,
                            "mim_id": str
                        }
                    ],
                    "organism": str
                }
            ]
        }

    Example:
        >>> result = query_ncbi_gene("NPC1")
        >>> print(result["results"][0]["summary"])
        "This gene encodes a protein containing..."
    """
    if not term or not term.strip():
        return {"error": "Empty search term provided.", "term": term, "results": []}

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # Step 1: Search for gene IDs
    search_url = f"{base_url}/esearch.fcgi"
    search_params = _build_params({
        "db": "gene",
        "term": f"{term}[Gene Name] OR {term}[Gene Symbol]",
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance"
    })

    # For human genes, add organism filter
    if not any(org in term.lower() for org in ["mouse", "mus", "rat", "drosophila"]):
        search_params["term"] = f"({term}[Gene Name] OR {term}[Gene Symbol]) AND Homo sapiens[Organism]"

    try:
        _rate_limit()
        search_resp = requests.get(search_url, params=search_params, timeout=DEFAULT_TIMEOUT)
        search_resp.raise_for_status()
        search_data = search_resp.json()

        id_list = search_data.get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return {"term": term, "results": [], "note": "No genes found matching the search term."}

        # Step 2: Get detailed summaries for each gene ID
        summary_url = f"{base_url}/esummary.fcgi"
        summary_params = _build_params({
            "db": "gene",
            "id": ",".join(id_list),
            "retmode": "json"
        })

        _rate_limit()
        summary_resp = requests.get(summary_url, params=summary_params, timeout=DEFAULT_TIMEOUT)
        summary_resp.raise_for_status()
        summary_data = summary_resp.json()

        results = []
        doc_sums = summary_data.get("result", {})

        for gene_id in id_list:
            gene_data = doc_sums.get(gene_id, {})

            if not gene_data or "error" in gene_data:
                continue

            # Extract basic information
            symbol = gene_data.get("name", "")
            name = gene_data.get("description", "")
            summary = gene_data.get("summary", "")
            chromosome = gene_data.get("chromosome", "")
            location = gene_data.get("maplocation", "")
            gene_type = gene_data.get("geneticSource", "")
            organism = gene_data.get("organism", {}).get("scientificname", "")

            # Extract aliases
            other_aliases = gene_data.get("otheraliases", "")
            aliases = [a.strip() for a in other_aliases.split(",") if a.strip()] if other_aliases else []

            # Extract MIM IDs and phenotypes from external references
            mim_ids = []
            phenotypes = []

            # Check for MIM in the gene record
            other_designations = gene_data.get("otherdesignations", "")

            # Look for phenotype information in genomicinfo
            genomic_info = gene_data.get("genomicinfo", [])
            if genomic_info and isinstance(genomic_info, list):
                for info in genomic_info:
                    if info.get("chrloc"):
                        location = f"{info.get('chraccver', '')}:{info.get('chrstart', '')}-{info.get('chrstop', '')}"
                        break

            # Build result entry
            result_entry = {
                "gene_id": gene_id,
                "symbol": symbol,
                "name": name,
                "aliases": aliases,
                "summary": summary,
                "chromosome": chromosome,
                "location": location,
                "gene_type": gene_type,
                "mim_ids": mim_ids,
                "phenotypes": phenotypes,
                "organism": organism
            }

            results.append(result_entry)

        # Step 3: Enrich with phenotype data from Gene2MIM links (optional enhancement)
        # This requires an additional efetch call, so we do it selectively
        if results and len(results) <= 3:
            results = _enrich_with_phenotypes(results, base_url)

        return {"term": term, "results": results}

    except requests.exceptions.Timeout:
        return {"error": "Request timed out.", "term": term, "results": []}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}", "term": term, "results": []}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "term": term, "results": []}


def _enrich_with_phenotypes(results: List[Dict], base_url: str) -> List[Dict]:
    """
    Enrich gene results with phenotype/disease associations.

    Uses efetch to get full gene records which include links to OMIM phenotypes.
    """
    gene_ids = [r["gene_id"] for r in results]

    fetch_url = f"{base_url}/efetch.fcgi"
    fetch_params = _build_params({
        "db": "gene",
        "id": ",".join(gene_ids),
        "rettype": "xml",
        "retmode": "xml"
    })

    try:
        _rate_limit()
        fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=DEFAULT_TIMEOUT)
        fetch_resp.raise_for_status()

        # Parse XML response
        root = ET.fromstring(fetch_resp.content)

        # Create lookup by gene ID
        result_lookup = {r["gene_id"]: r for r in results}

        # Process each Entrezgene entry
        for entrezgene in root.findall(".//Entrezgene"):
            # Get gene ID
            gene_id_elem = entrezgene.find(".//Gene-track_geneid")
            if gene_id_elem is None:
                continue
            gene_id = gene_id_elem.text

            if gene_id not in result_lookup:
                continue

            result = result_lookup[gene_id]
            phenotypes = []
            mim_ids = []

            # Look for MIM references in Dbtag elements
            for dbtag in entrezgene.findall(".//Dbtag"):
                db_elem = dbtag.find("Dbtag_db")
                if db_elem is not None and db_elem.text == "MIM":
                    tag_id = dbtag.find(".//Object-id_id")
                    if tag_id is not None:
                        mim_id = tag_id.text
                        if mim_id not in mim_ids:
                            mim_ids.append(mim_id)

            # Look for phenotype descriptions in Gene-commentary
            for commentary in entrezgene.findall(".//Gene-commentary"):
                heading = commentary.find("Gene-commentary_heading")
                text = commentary.find("Gene-commentary_text")

                if heading is not None and "phenotype" in heading.text.lower():
                    if text is not None and text.text:
                        phenotypes.append({
                            "phenotype": text.text,
                            "mim_id": None
                        })

                # Also check for disease associations in commentary labels
                label = commentary.find("Gene-commentary_label")
                if label is not None and label.text:
                    label_text = label.text.lower()
                    if any(term in label_text for term in ["disease", "disorder", "syndrome", "phenotype"]):
                        phenotypes.append({
                            "phenotype": label.text,
                            "mim_id": None
                        })

            # Look for phenotype info in Entrezgene_prot section comments
            for prot_comment in entrezgene.findall(".//Entrezgene_prot//Gene-commentary"):
                comment_text = prot_comment.find("Gene-commentary_text")
                if comment_text is not None and comment_text.text:
                    text = comment_text.text
                    # Check if it mentions disease/phenotype
                    if any(term in text.lower() for term in ["disease", "disorder", "mutation", "deficiency"]):
                        if len(text) < 500:  # Avoid very long texts
                            if not any(p["phenotype"] == text for p in phenotypes):
                                phenotypes.append({
                                    "phenotype": text,
                                    "mim_id": None
                                })

            result["mim_ids"] = mim_ids
            result["phenotypes"] = phenotypes[:10]  # Limit to 10 phenotypes

    except Exception:
        # If enrichment fails, return results without phenotypes
        pass

    return results


def query_ncbi_gene_summary(gene_symbol: str) -> str:
    """
    Convenience function to get just the gene summary.

    Args:
        gene_symbol: Gene symbol (e.g., "NPC1")

    Returns:
        Gene summary string, or empty string if not found
    """
    result = query_ncbi_gene(gene_symbol, max_results=1)
    if result.get("results"):
        return result["results"][0].get("summary", "")
    return ""


if __name__ == "__main__":
    pm = query_pubmed("Parkinsons")
    print(pm)
    print(query_clinicaltrials("Alzheimer's"))

    genetest = False
    if genetest:
        # Test with a well-known gene
        tst_genes = ["NPC1", "BRCA1", "TP53"]

        for gene in tst_genes:
            print(f"\n{'=' * 60}")
            print(f"Testing: {gene}")
            print("=" * 60)

            result = query_ncbi_gene(gene)

            if result.get("error"):
                print(f"Error: {result['error']}")
                continue

            for r in result.get("results", [])[:2]:
                print(f"\nGene ID: {r['gene_id']}")
                print(f"Symbol: {r['symbol']}")
                print(f"Name: {r['name']}")
                print(f"Aliases: {', '.join(r['aliases'][:5])}")
                print(f"Chromosome: {r['chromosome']}")
                print(f"Location: {r['location']}")

                summary = r.get('summary', '')
                if summary:
                    print(f"Summary: {summary[:200]}...")

                if r.get('phenotypes'):
                    print("Phenotypes:")
                    for p in r['phenotypes'][:3]:
                        print(f"  - {p['phenotype'][:80]}")

                if r.get('mim_ids'):
                    print(f"MIM IDs: {', '.join(r['mim_ids'][:5])}")

    extras = False
    if extras:
        # --- 1) RefSNP + frequency extraction (should NOT be empty if counts exist) ---
        rsid = "rs328"
        j = query_refsnp(rsid)
        freqs = extract_refsnp_freqs(j, studies=("1000Genomes", "1000Genomes_30X", "ALFA"))

        print(f"[refsnp] {rsid}: extracted {len(freqs)} freq rows")
        print(freqs[:5])

        maf = get_population_maf(rsid)
        print(f"[maf] {rsid}: {maf}")

        # --- 2) HGVS -> contextual SPDI ---
        hgvs = "NM_000518.4:c.27dupG"
        ctx = hgvs_to_contextuals(hgvs)
        print(f"[hgvs->contextuals] keys: {list(ctx.keys())[:8]}")

        # --- 3) VCF -> contextual SPDI (toy indel example) ---
        ctx2 = vcf_to_contextuals("11", 5248224, "A", "AC",
                                  assembly="GCF_000001405.25")  # may vary by assembly; just checks endpoint works
        print(f"[vcf->contextuals] keys: {list(ctx2.keys())[:8]}")

        # --- 4) Optional: BLAST organism filter smoke test
        # Known human HBB (beta-globin) coding-region snippet (works well as a demo query)
        seq = (
            "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAAC"
            "GTGGATGAAGTTGGTGGTGAGGCCCTGGGCAG"
        )

        r = blast_sequence(
            seq,
            program="blastn",
            database="nt",
            hitlist_size=5,
            entrez_query="txid9606[ORGN]",
            max_wait_s=60,
            poll_interval_s=15,
        )

        print(r.get("error") or f"OK: {len(r.get('hits', []))} hits")
