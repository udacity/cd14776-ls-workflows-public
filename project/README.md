# Drug Repositioning Agentic Workflow

## Project Overview

Build a multi-agent system that automates drug repurposing research. Given a target gene (e.g., TMEM97) and a disease (e.g., non-alcoholic steatohepatitis), your workflow will:

1. **Mine** biomedical databases for candidate compounds
2. **Route** candidates to appropriate enrichment strategies
3. **Enrich** candidates with literature evidence and safety data
4. **Score & Rank** candidates using iterative refinement
5. **Generate** a validation roadmap for top candidates

## Learning Objectives

This project demonstrates five agentic workflow patterns:

| Pattern | Agent | What You'll Implement |
|---------|-------|----------------------|
| Sequential Workflow | DataMiningAgent | ChEMBL → Open Targets → Merge pipeline |
| Prompt Chaining | LiteratureAgent | PubMed search → LLM assessment |
| LLM-based Routing | RoutingAgent | Dynamic strategy selection per candidate |
| Evaluator-Optimizer | EvaluationAgent | Score → Critique → Refine loop |
| Orchestration | Orchestrator | Coordinate all agents end-to-end |

## Setup

### Requirements

```bash
pip install openai requests biopython pandas chembl-webresource-client python-dotenv
```

### Environment Variables

Create a `.env` file:

```bash
# For Vocareum (if applicable)
UDACITY_OPENAI_API_KEY=your_key_here

# Optional
NCBI_EMAIL=your_email@example.com
```

## Running the Project

### Mock Mode (Recommended for Development)

No API keys required. Uses deterministic mock data:

```bash
USE_MOCK_DATA=true python scaffold.py
```

### Live Mode

Requires OpenAI API key. Queries real databases:

```bash
python scaffold.py
```

### Custom Target/Disease

```bash
TARGET=SIGMAR1 DISEASE=depression USE_MOCK_DATA=true python scaffold.py
```

## Project Structure

```
scaffold.py          # Main file - implement TODOs here
├── Configuration    # DO NOT MODIFY
├── Utilities        # DO NOT MODIFY  
├── Data Classes     # DO NOT MODIFY
├── Mock Data        # DO NOT MODIFY
├── API Functions    # DO NOT MODIFY
├── BaseAgent        # DO NOT MODIFY
└── YOUR CODE        # Implement TODOs 1-10
    ├── ActionPlanningAgent
    ├── RoutingAgent
    ├── DataMiningAgent
    ├── LiteratureAgent
    ├── EvaluationAgent
    └── DrugRepurposingOrchestrator
```

## Your Tasks

Complete **10 TODOs** across 6 agent classes:

| TODO | Agent | Task |
|------|-------|------|
| 1 | ActionPlanningAgent | Write planning prompt |
| 2 | RoutingAgent | Write routing prompt |
| 3 | DataMiningAgent | Implement sequential workflow |
| 4 | LiteratureAgent | Implement prompt chaining |
| 5 | LiteratureAgent | Write assessment prompt |
| 6 | EvaluationAgent | Implement evaluator-optimizer loop |
| 7 | EvaluationAgent | Implement scoring logic |
| 8 | EvaluationAgent | Write critique prompt |
| 9 | EvaluationAgent | Write refinement prompt + apply adjustments |
| 10 | Orchestrator | Implement step execution |

Each TODO includes detailed instructions and hints in the code comments.

## Key Concepts

### Prompt Engineering for JSON Output

Several TODOs require prompts that return structured JSON. Tips:
- Explicitly specify the JSON format you expect
- Use `self._call_llm_json(prompt, fallback)` — it handles parsing and retries
- The `fallback` dict is returned if parsing fails

### Evaluator-Optimizer Pattern

The core insight: **adjustments must persist across iterations**.

```
Iteration 1: Initial scoring from data
Iteration 2: Recalculate totals from adjusted scores (don't rescore!)
Iteration 3: Continue accumulating adjustments...
```

If you call `_score_candidates()` every iteration, you wipe out previous adjustments.

### Routing Strategies

The RoutingAgent chooses strategies based on candidate characteristics:

| Literature Strategy | When to Use |
|--------------------|-------------|
| target_focused | Known MOA, look for target evidence |
| disease_focused | Unknown MOA, look for disease evidence |
| broad_search | Early stage, cast wide net |

| Safety Strategy | When to Use |
|-----------------|-------------|
| comprehensive | Phase 3-4 drugs (FDA labels + FAERS) |
| faers_only | Phase 1-2 drugs |
| basic | Preclinical compounds |

## Expected Output

Successful execution produces:

1. **Console output**: Progress logs, top candidates, validation roadmap
2. **CSV file**: `candidates_{target}_{disease}.csv` with ranked candidates
3. **JSON file**: `audit_{target}_{disease}.json` with full audit trail

### Sample Console Output (Mock Mode)

```
============================================================
DRUG REPURPOSING AGENTIC WORKFLOW
Target: TMEM97 | Disease: non-alcoholic steatohepatitis
Mock mode: False
Max candidates to enrich: 5
============================================================


[Planning] ActionPlanningAgent creating plan...
  Objective: Identify and evaluate existing drugs that can be repurposed for the treatment of non-alcoholic steatohepatitis by targeting TMEM97.
  Steps: ['data_mining', 'enrichment', 'scoring', 'roadmap']

[Step 1] Query ChEMBL and Open Targets databases to identify compounds that interact with TMEM97 and have potential relevance to non-alcoholic steatohepatitis.
  Routed to: DataMiningAgent

[Step 2] Gather literature evidence and safety data on the identified compounds, focusing on their mechanisms of action, previous clinical trials, and any reported effects on liver health.
  Routed to: LiteratureAgent + SafetyAgent
  Enriching top 5 candidates...

[Step 3] Score and rank the identified compounds based on their efficacy, safety profiles, and relevance to TMEM97 in the context of non-alcoholic steatohepatitis.
  Routed to: EvaluationAgent

[Step 4] Generate a validation plan outlining the experimental approaches needed to test the top-ranked candidates in preclinical models of non-alcoholic steatohepatitis.
  Routed to: EvaluationAgent

============================================================
WORKFLOW COMPLETE
LLM calls: 23 | Candidates: 5 (5 enriched)
Parallel enrichment: True
============================================================

============================================================
TOP CANDIDATES
============================================================

1. PITOLISANT HYDROCHLORIDE (CHEMBL4164059)
   Score: 6.14 | Phase: 4 | pChEMBL: 8.15
   ...
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `EnvironmentError: Set either UDACITY_OPENAI_API_KEY...` | Use mock mode: `USE_MOCK_DATA=true` |
| Empty candidates list | Check TODO 3 — ensure all three steps execute |
| `candidates_enriched: 0` | Check TODO 10 — ensure `is_enriched = True` is set |
| Scores reset each iteration | Check TODO 6 — only call `_score_candidates` on iteration 1 |
| Loop never exits | Check TODO 8 — ensure approval phrase is detectable |

## Submission

Submit:
1. Completed `scaffold.py`
2. Output files from a successful mock-mode run:
   - `candidates_tmem97_non_alcoholic_steatohepatitis.csv`
   - `audit_tmem97_non_alcoholic_steatohepatitis.json`
3. Screenshot or text file of console output