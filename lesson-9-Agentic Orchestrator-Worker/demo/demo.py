"""
Module 9 Demo: Lab Experiment Scheduler - Orchestrator-Worker Pattern

This module implements an orchestrator-worker pattern where a central agent
decomposes experiment requests and delegates to specialized workers:
- EquipmentAgent: Handles equipment scheduling and availability
- ProtocolAgent: Creates detailed protocol timelines
- SafetyAgent: Assesses safety requirements and compliance

The orchestrator coordinates these workers and synthesizes results into
a comprehensive experiment schedule.

Learning Objective: Design a dynamic, multi-agent workflow using an orchestrator
to plan, delegate, and synthesize tasks performed by specialized worker agents.

"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent / ".env")


# =============================================================================
# EXPERIMENT REQUEST INPUT
# =============================================================================

EXPERIMENT_REQUEST = {
    "experiment_name": "CRISPR knockout validation in HEK293T cells",
    "principal_investigator": "Dr. Chen",
    "requested_date": "2025-01-27",
    "description": """
    Validate CRISPR knockout of TP53 gene in HEK293T cells:
    1. Transfection of Cas9 and gRNA
    2. Puromycin selection (3 days)
    3. Single-cell cloning
    4. Genomic DNA extraction and PCR
    5. Sanger sequencing
    6. Western blot to confirm knockout
    Timeline: 3 weeks
    """,
    "team_members": ["Dr. Chen", "Sarah (postdoc)", "Mike (grad student)"],
    "cell_line": "HEK293T",
    "target_gene": "TP53"
}


# =============================================================================
# WORKER AGENT BASE CLASS
# =============================================================================

class WorkerAgent(ABC):
    """Base class for specialized worker agents."""

    def __init__(self, name: str, llm_client: OpenAI):
        self.name = name
        self.llm = llm_client

    @abstractmethod
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute assigned task and return results."""
        pass

    def _call_llm_json(self, system_prompt: str, user_prompt: str) -> Dict:
        """Helper to call LLM and parse JSON response."""
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)


# =============================================================================
# PROTOCOL AGENT
# =============================================================================

class ProtocolAgent(WorkerAgent):
    """Creates detailed protocol timelines."""

    def __init__(self, llm_client: OpenAI):
        super().__init__("Protocol Agent", llm_client)

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed protocol timeline."""
        system_prompt = """You are a molecular biology protocol specialist.
Create detailed experimental protocols with realistic timing.

Return JSON with:
{
    "protocol_steps": [{"step_number": 1, "step_name": "...", "duration": "...", "start_day": 1, "end_day": 1, "critical_timing": true/false, "hands_on_time": "...", "reagents_needed": [...]}],
    "total_duration_days": number,
    "milestones": [{"name": "...", "day": number, "description": "..."}],
    "critical_path": [step numbers],
    "protocol_notes": "..."
}"""

        user_prompt = f"""Create protocol for: {task.get('experiment_description', '')}
Cell Line: {task.get('cell_line')} | Target: {task.get('target_gene')}
Timeline: {task.get('timeline_weeks', 3)} weeks"""

        try:
            return self._call_llm_json(system_prompt, user_prompt)
        except Exception:
            return self._default_protocol()

    def _default_protocol(self) -> Dict:
        """Fallback protocol if LLM fails."""
        return {
            "protocol_steps": [
                {"step_number": 1, "step_name": "Cell Preparation", "duration": "1 day", "start_day": 1, "end_day": 1, "critical_timing": False, "hands_on_time": "2 hours", "reagents_needed": ["DMEM", "FBS"]},
                {"step_number": 2, "step_name": "Transfection", "duration": "1 day", "start_day": 2, "end_day": 2, "critical_timing": True, "hands_on_time": "3 hours", "reagents_needed": ["Cas9 plasmid", "gRNA", "Lipofectamine"]},
                {"step_number": 3, "step_name": "Selection", "duration": "3 days", "start_day": 3, "end_day": 5, "critical_timing": True, "hands_on_time": "30 min/day", "reagents_needed": ["Puromycin"]},
                {"step_number": 4, "step_name": "Single-cell Cloning", "duration": "1 day", "start_day": 8, "end_day": 8, "critical_timing": True, "hands_on_time": "4 hours", "reagents_needed": ["96-well plates"]},
                {"step_number": 5, "step_name": "Clone Expansion", "duration": "6 days", "start_day": 9, "end_day": 14, "critical_timing": False, "hands_on_time": "1 hour/day", "reagents_needed": ["DMEM", "FBS"]},
                {"step_number": 6, "step_name": "DNA Extraction & PCR", "duration": "1 day", "start_day": 15, "end_day": 15, "critical_timing": False, "hands_on_time": "5 hours", "reagents_needed": ["DNA kit", "PCR primers"]},
                {"step_number": 7, "step_name": "Sequencing", "duration": "3 days", "start_day": 16, "end_day": 18, "critical_timing": False, "hands_on_time": "1 hour", "reagents_needed": []},
                {"step_number": 8, "step_name": "Western Blot", "duration": "2 days", "start_day": 17, "end_day": 18, "critical_timing": False, "hands_on_time": "8 hours", "reagents_needed": ["Anti-TP53 antibody", "ECL"]},
            ],
            "total_duration_days": 19,
            "milestones": [
                {"name": "Transfection complete", "day": 2, "description": "Cells transfected"},
                {"name": "Selection complete", "day": 5, "description": "Resistant pool established"},
                {"name": "Clones isolated", "day": 8, "description": "Single cells plated"},
                {"name": "Experiment complete", "day": 19, "description": "KO clones validated"}
            ],
            "critical_path": [2, 3, 4],
            "protocol_notes": "Critical timing for transfection and selection steps."
        }


# =============================================================================
# EQUIPMENT AGENT
# =============================================================================

class EquipmentAgent(WorkerAgent):
    """Handles equipment scheduling."""

    EQUIPMENT_CATALOG = {
        "tissue_culture_hood": {"name": "Biosafety Cabinet", "location": "Cell Culture Room", "availability": "high"},
        "electroporator": {"name": "Nucleofector", "location": "Equipment Room", "availability": "medium"},
        "cell_sorter": {"name": "BD FACSAria III", "location": "Flow Core", "availability": "low", "core_booking": True},
        "thermal_cycler": {"name": "PCR Thermal Cycler", "location": "Molecular Biology Bay", "availability": "high"},
        "western_blot_system": {"name": "Bio-Rad Western System", "location": "Protein Analysis Lab", "availability": "low"},
        "sequencer": {"name": "Sanger Sequencing", "location": "Genomics Core", "availability": "medium", "core_submission": True}
    }

    def __init__(self, llm_client: OpenAI):
        super().__init__("Equipment Agent", llm_client)

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create equipment booking schedule."""
        system_prompt = f"""You are a lab equipment scheduler.
Available equipment: {json.dumps(self.EQUIPMENT_CATALOG, indent=2)}

Return JSON with:
{{
    "equipment_needed": [{{"equipment_id": "...", "equipment_name": "...", "location": "...", "step": "...", "suggested_date": "YYYY-MM-DD", "suggested_time": "...", "booking_notes": "..."}}],
    "core_facility_submissions": [{{"facility": "...", "lead_time_days": number, "notes": "..."}}],
    "equipment_summary": "..."
}}"""

        protocol_steps = task.get("protocol_results", {}).get("protocol_steps", [])
        user_prompt = f"""Schedule equipment for experiment starting {task.get('requested_start_date', '2025-01-27')}.
Protocol steps: {json.dumps(protocol_steps, indent=2)}"""

        try:
            return self._call_llm_json(system_prompt, user_prompt)
        except Exception:
            return self._default_equipment(task)

    def _default_equipment(self, task: Dict) -> Dict:
        """Fallback equipment schedule."""
        start = datetime.strptime(task.get('requested_start_date', '2025-01-27'), '%Y-%m-%d')
        return {
            "equipment_needed": [
                {"equipment_id": "tissue_culture_hood", "equipment_name": "Biosafety Cabinet", "location": "Cell Culture Room", "step": "Cell culture", "suggested_date": start.strftime('%Y-%m-%d'), "suggested_time": "9:00 AM - 1:00 PM", "booking_notes": "Daily use"},
                {"equipment_id": "electroporator", "equipment_name": "Nucleofector", "location": "Equipment Room", "step": "Transfection", "suggested_date": (start + timedelta(days=1)).strftime('%Y-%m-%d'), "suggested_time": "10:00 AM - 1:00 PM", "booking_notes": "Book early"},
                {"equipment_id": "cell_sorter", "equipment_name": "BD FACSAria III", "location": "Flow Core", "step": "Single-cell cloning", "suggested_date": (start + timedelta(days=7)).strftime('%Y-%m-%d'), "suggested_time": "9:00 AM - 12:00 PM", "booking_notes": "Core facility booking"},
                {"equipment_id": "thermal_cycler", "equipment_name": "PCR Thermal Cycler", "location": "Molecular Biology Bay", "step": "PCR", "suggested_date": (start + timedelta(days=14)).strftime('%Y-%m-%d'), "suggested_time": "1:00 PM - 5:00 PM", "booking_notes": "Multiple units available"},
                {"equipment_id": "western_blot_system", "equipment_name": "Bio-Rad Western System", "location": "Protein Analysis Lab", "step": "Western blot", "suggested_date": (start + timedelta(days=16)).strftime('%Y-%m-%d'), "suggested_time": "10:00 AM - 5:00 PM", "booking_notes": "Book 1 week ahead"},
            ],
            "core_facility_submissions": [
                {"facility": "Flow Core", "lead_time_days": 5, "notes": "Submit by Day 1 for Day 8 sort"},
                {"facility": "Genomics Core", "lead_time_days": 3, "notes": "2-3 day turnaround"}
            ],
            "equipment_summary": "5 major bookings. Critical: FACSAria (5-day lead), Western blot (limited)."
        }


# =============================================================================
# SAFETY AGENT
# =============================================================================

class SafetyAgent(WorkerAgent):
    """Assesses safety requirements."""

    def __init__(self, llm_client: OpenAI):
        super().__init__("Safety Agent", llm_client)

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Assess safety requirements."""
        system_prompt = """You are a laboratory safety officer.
Assess BSL requirements, hazardous materials, PPE, and training needs.

Return JSON with:
{
    "biosafety_level": "BSL-1 or BSL-2",
    "hazardous_materials": [{"material": "...", "hazard_type": "...", "handling_requirements": "..."}],
    "ppe_requirements": ["..."],
    "training_requirements": [{"training": "...", "required_for": "...", "personnel": ["..."]}],
    "waste_disposal": [{"waste_type": "...", "disposal_method": "..."}],
    "compliance_checklist": [{"item": "...", "status": "Complete/Needed"}],
    "safety_summary": "..."
}"""

        user_prompt = f"""Safety assessment for: {task.get('experiment_description', '')}
Cell Line: {task.get('cell_line')} | Team: {task.get('team_members', [])}"""

        try:
            return self._call_llm_json(system_prompt, user_prompt)
        except Exception:
            return self._default_safety(task)

    def _default_safety(self, task: Dict) -> Dict:
        """Fallback safety assessment."""
        team = task.get('team_members', ['Researcher'])
        return {
            "biosafety_level": "BSL-2",
            "hazardous_materials": [
                {"material": "HEK293T cells", "hazard_type": "Biological", "handling_requirements": "BSL-2 cabinet"},
                {"material": "Puromycin", "hazard_type": "Chemical", "handling_requirements": "Gloves, avoid inhalation"}
            ],
            "ppe_requirements": ["Lab coat", "Nitrile gloves", "Safety glasses"],
            "training_requirements": [
                {"training": "BSL-2 Training", "required_for": "Cell culture", "personnel": team},
                {"training": "Flow Cytometry", "required_for": "Cell sorting", "personnel": [team[1] if len(team) > 1 else team[0]]}
            ],
            "waste_disposal": [
                {"waste_type": "Cell culture waste", "disposal_method": "Autoclave then biohazard bags"},
                {"waste_type": "Chemical waste", "disposal_method": "EHS pickup"}
            ],
            "compliance_checklist": [
                {"item": "IBC protocol approval", "status": "Complete"},
                {"item": "BSL-2 training", "status": "Complete"},
                {"item": "Western blot training (Mike)", "status": "Needed"}
            ],
            "safety_summary": "BSL-2 experiment with standard molecular biology safety practices."
        }


# =============================================================================
# EXPERIMENT ORCHESTRATOR
# =============================================================================

class ExperimentOrchestrator:
    """
    Central orchestrator that decomposes tasks and coordinates workers.

    This is the core of the orchestrator-worker pattern:
    1. Decompose complex request into subtasks
    2. Assign subtasks to specialized workers
    3. Execute in dependency order
    4. Synthesize results into final output
    """

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client
        self.protocol_agent = ProtocolAgent(llm_client)
        self.equipment_agent = EquipmentAgent(llm_client)
        self.safety_agent = SafetyAgent(llm_client)

    def decompose_request(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose request into tasks with dependencies."""
        base_task = {
            "experiment_description": request["description"],
            "cell_line": request["cell_line"],
            "target_gene": request["target_gene"],
            "requested_start_date": request["requested_date"],
            "timeline_weeks": 3,
            "team_members": request["team_members"]
        }

        return [
            {"worker": "protocol", "task": base_task, "dependencies": [], "priority": 1},
            {"worker": "equipment", "task": base_task, "dependencies": ["protocol"], "priority": 2},
            {"worker": "safety", "task": base_task, "dependencies": ["protocol", "equipment"], "priority": 3}
        ]

    def execute_plan(self, tasks: List[Dict]) -> Dict[str, Any]:
        """Execute tasks in dependency order, passing results between workers."""
        results = {}
        workers = {
            "protocol": self.protocol_agent,
            "equipment": self.equipment_agent,
            "safety": self.safety_agent
        }

        for task in sorted(tasks, key=lambda t: t["priority"]):
            worker_name = task["worker"]
            print(f"\n  [{worker_name.upper()} AGENT]")

            # Add dependency results to task
            for dep in task["dependencies"]:
                if dep in results:
                    task["task"][f"{dep}_results"] = results[dep]

            results[worker_name] = workers[worker_name].execute(task["task"])
            print(f"  + Completed")

        return results

    def synthesize_schedule(self, request: Dict, worker_results: Dict) -> Dict[str, Any]:
        """Synthesize worker outputs into final experiment schedule."""
        protocol = worker_results.get("protocol", {})
        equipment = worker_results.get("equipment", {})
        safety = worker_results.get("safety", {})

        start_date = datetime.strptime(request["requested_date"], "%Y-%m-%d")
        total_days = protocol.get("total_duration_days", 19)
        end_date = start_date + timedelta(days=total_days - 1)

        return {
            "experiment_name": request["experiment_name"],
            "principal_investigator": request["principal_investigator"],
            "team_members": request["team_members"],
            "schedule": {
                "start_date": start_date.strftime("%A, %B %d, %Y"),
                "end_date": end_date.strftime("%A, %B %d, %Y"),
                "total_days": total_days
            },
            "protocol_steps": protocol.get("protocol_steps", []),
            "milestones": protocol.get("milestones", []),
            "equipment_bookings": equipment.get("equipment_needed", []),
            "core_submissions": equipment.get("core_facility_submissions", []),
            "safety": {
                "biosafety_level": safety.get("biosafety_level", "BSL-2"),
                "ppe": safety.get("ppe_requirements", []),
                "hazards": safety.get("hazardous_materials", []),
                "training": safety.get("training_requirements", [])
            },
            "compliance_checklist": safety.get("compliance_checklist", []),
            "executive_summary": f"This {total_days}-day experiment is feasible within 3 weeks. Critical path: transfection -> selection -> cloning. Key actions: book FACSAria by Day 1, ensure training complete."
        }

    def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Full orchestration workflow."""
        print(f"\n{'='*70}")
        print("LAB EXPERIMENT SCHEDULER - ORCHESTRATOR-WORKER PATTERN")
        print(f"Experiment: {request['experiment_name']}")
        print(f"{'='*70}")

        # Phase 1: Decompose
        print("\n[PHASE 1: TASK DECOMPOSITION]")
        tasks = self.decompose_request(request)
        print(f"Created {len(tasks)} tasks:")
        for t in tasks:
            deps = ", ".join(t["dependencies"]) if t["dependencies"] else "none"
            print(f"  - {t['worker']}: depends on {deps}")

        # Phase 2: Execute
        print("\n[PHASE 2: WORKER EXECUTION]")
        worker_results = self.execute_plan(tasks)

        # Phase 3: Synthesize
        print("\n[PHASE 3: SYNTHESIS]")
        print("Combining worker outputs...")
        return self.synthesize_schedule(request, worker_results)


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_schedule(schedule: Dict[str, Any]) -> str:
    """Format the final schedule for display."""
    lines = []

    lines.append(f"\n{'='*70}")
    lines.append("FINAL EXPERIMENT SCHEDULE")
    lines.append("="*70)

    lines.append(f"\nExperiment: {schedule['experiment_name']}")
    lines.append(f"PI: {schedule['principal_investigator']}")
    lines.append(f"Team: {', '.join(schedule['team_members'])}")

    sched = schedule['schedule']
    lines.append(f"\nTimeline: {sched['start_date']} to {sched['end_date']} ({sched['total_days']} days)")

    # Protocol steps
    lines.append(f"\n{'─'*70}")
    lines.append("PROTOCOL STEPS")
    lines.append("─"*70)
    for step in schedule.get('protocol_steps', [])[:6]:
        critical = " [CRITICAL]" if step.get('critical_timing') else ""
        lines.append(f"Day {step['start_day']}-{step['end_day']}: {step['step_name']}{critical}")

    # Equipment
    lines.append(f"\n{'─'*70}")
    lines.append("EQUIPMENT BOOKINGS")
    lines.append("─"*70)
    for eq in schedule.get('equipment_bookings', [])[:5]:
        lines.append(f"  {eq.get('suggested_date', 'TBD')}: {eq.get('equipment_name')} @ {eq.get('location')}")

    # Core submissions
    if schedule.get('core_submissions'):
        lines.append("\nCore Facility Submissions:")
        for sub in schedule['core_submissions']:
            lines.append(f"  ! {sub['facility']}: {sub['notes']}")

    # Safety
    lines.append(f"\n{'─'*70}")
    lines.append("SAFETY REQUIREMENTS")
    lines.append("─"*70)
    safety = schedule.get('safety', {})
    lines.append(f"Biosafety Level: {safety.get('biosafety_level', 'N/A')}")
    lines.append(f"PPE: {', '.join(safety.get('ppe', []))}")

    # Compliance
    lines.append("\nCompliance Checklist:")
    for item in schedule.get('compliance_checklist', []):
        status = "+" if item.get('status') == 'Complete' else "!"
        lines.append(f"  [{status}] {item.get('item')}")

    # Milestones
    lines.append(f"\n{'─'*70}")
    lines.append("MILESTONES")
    lines.append("─"*70)
    for ms in schedule.get('milestones', []):
        lines.append(f"  Day {ms.get('day')}: {ms.get('name')}")

    # Summary
    lines.append(f"\n{'─'*70}")
    lines.append("EXECUTIVE SUMMARY")
    lines.append("─"*70)
    lines.append(schedule.get('executive_summary', ''))

    lines.append(f"\n{'='*70}")
    return "\n".join(lines)


def main():
    """Main entry point for the lab scheduler demo."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

    print("="*70)
    print("ORCHESTRATOR-WORKER PATTERN DEMO")
    print("="*70)
    print("\nThis demo shows the orchestrator-worker pattern:")
    print("  1. Orchestrator decomposes complex request into subtasks")
    print("  2. Tasks are assigned to specialized worker agents")
    print("  3. Workers execute in dependency order")
    print("  4. Orchestrator synthesizes results into final output")

    orchestrator = ExperimentOrchestrator(client)
    schedule = orchestrator.run(EXPERIMENT_REQUEST)
    print(format_schedule(schedule))


if __name__ == "__main__":
    main()
