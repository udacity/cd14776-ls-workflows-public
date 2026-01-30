"""
Module 2 Demo: Agent Anatomy - The Lab Manager Agent
=====================================================

This demo illustrates the fundamental components of a modern AI agent:
1. GOALS - System prompt defining purpose, priorities, constraints
2. TOOLS - External capabilities the agent can invoke
3. MEMORY - State tracking across the agent loop
4. REASONING - LLM-based planning and decision making
5. PERCEPTION - Understanding and parsing input

We demonstrate these through a Lab Manager agent that handles
researcher requests by reasoning through tool use.

Learning Objective:
    Define what constitutes a modern AI agent and identify its fundamental
    components and types.
"""

import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent.parent / ".env")


# =============================================================================
# SIMULATED LAB ENVIRONMENT (Tool backends)
# =============================================================================

EQUIPMENT_STATUS = {
    "thermal_cycler_1": {"status": "available", "next_reservation": None},
    "thermal_cycler_2": {"status": "in_use", "next_reservation": "14:00", "user": "Dr. Chen"},
    "sequencer": {"status": "available", "next_reservation": "tomorrow 09:00"},
    "centrifuge": {"status": "maintenance", "available_date": "2025-01-22"},
    "flow_cytometer": {"status": "available", "next_reservation": None},
}

INVENTORY = {
    "Taq_polymerase": {"quantity": 2, "unit": "vials", "status": "low", "reorder_threshold": 5},
    "dNTPs": {"quantity": 15, "unit": "vials", "status": "sufficient", "reorder_threshold": 5},
    "PCR_primers": {"quantity": 50, "unit": "tubes", "status": "sufficient", "reorder_threshold": 20},
    "extraction_kit": {"quantity": 3, "unit": "kits", "status": "low", "reorder_threshold": 5},
    "sequencing_reagents": {"quantity": 10, "unit": "kits", "status": "sufficient", "reorder_threshold": 3},
}

PROTOCOLS = {
    "standard_pcr": "1. Prepare master mix\n2. Add template DNA\n3. Run thermal cycler: 95°C 2min, 35x(95°C 30s, 55°C 30s, 72°C 1min), 72°C 5min",
    "dna_extraction": "1. Lyse cells\n2. Add binding buffer\n3. Spin through column\n4. Wash 2x\n5. Elute in 50µL TE buffer",
    "sequencing_prep": "1. Quantify DNA\n2. Normalize to 10ng/µL\n3. Prepare library\n4. Quality check\n5. Load on sequencer",
}


# =============================================================================
# LAB MANAGER AGENT
# =============================================================================

class LabManagerAgent:
    """
    A Lab Manager agent demonstrating the fundamental components of AI agents.

    COMPONENT ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │                     LabManagerAgent                         │
    ├─────────────────────────────────────────────────────────────┤
    │  GOALS (system_prompt)                                      │
    │    - Ensure lab safety and efficiency                       │
    │    - Help researchers with equipment and supplies           │
    │    - Proactively flag issues                                │
    ├─────────────────────────────────────────────────────────────┤
    │  TOOLS (self.tools)                                         │
    │    - check_equipment: Query equipment availability          │
    │    - check_inventory: Query supply levels                   │
    │    - get_protocol: Retrieve standard protocols              │
    │    - reserve_equipment: Make a reservation                  │
    ├─────────────────────────────────────────────────────────────┤
    │  MEMORY (self.conversation_history, self.lab_state)         │
    │    - Conversation context                                   │
    │    - Pending reservations                                   │
    │    - Flagged issues                                         │
    ├─────────────────────────────────────────────────────────────┤
    │  REASONING (LLM-based)                                      │
    │    - Decide which tools to use                              │
    │    - Synthesize information                                 │
    │    - Generate helpful responses                             │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

        # =====================================================================
        # COMPONENT 1: GOALS (System Prompt)
        # =====================================================================
        self.system_prompt = """You are Lab Manager Agent, an AI assistant for a molecular biology research lab.

YOUR GOALS (in priority order):
1. SAFETY FIRST - Never recommend actions that could compromise lab safety
2. EFFICIENCY - Help researchers plan experiments effectively
3. RESOURCE MANAGEMENT - Flag low supplies and equipment conflicts
4. HELPFULNESS - Provide clear, actionable guidance

YOUR CAPABILITIES:
You have access to tools to query lab systems. You should:
- Use tools to gather information before making recommendations
- Be proactive about flagging issues (low supplies, equipment conflicts)
- Provide specific, actionable answers

RESPONSE STYLE:
- Be concise but thorough
- If you use tools, briefly explain what you found
- If supplies are low, proactively mention it
- If equipment is unavailable, suggest alternatives"""

        # =====================================================================
        # COMPONENT 2: TOOLS
        # =====================================================================
        self.tools = {
            "check_equipment": {
                "function": self._tool_check_equipment,
                "description": "Check availability and status of lab equipment",
                "parameters": {"equipment_name": "Name of equipment to check"}
            },
            "check_inventory": {
                "function": self._tool_check_inventory,
                "description": "Check inventory levels of lab supplies",
                "parameters": {"item_name": "Name of supply item to check"}
            },
            "get_protocol": {
                "function": self._tool_get_protocol,
                "description": "Retrieve a standard lab protocol",
                "parameters": {"protocol_name": "Name of protocol to retrieve"}
            },
            "reserve_equipment": {
                "function": self._tool_reserve_equipment,
                "description": "Reserve equipment for a specific time",
                "parameters": {"equipment_name": "Equipment to reserve", "time": "Time slot"}
            }
        }

        # =====================================================================
        # COMPONENT 3: MEMORY
        # =====================================================================
        self.conversation_history: List[Dict] = []
        self.lab_state = {
            "pending_reservations": [],
            "flagged_issues": [],
            "recent_queries": []
        }

    # =========================================================================
    # TOOL IMPLEMENTATIONS
    # =========================================================================

    def _tool_check_equipment(self, equipment_name: str) -> Dict[str, Any]:
        """Tool: Check equipment availability."""
        # Fuzzy match equipment name
        equipment_name_lower = equipment_name.lower().replace(" ", "_").replace("-", "_")
        for key, value in EQUIPMENT_STATUS.items():
            if equipment_name_lower in key or key in equipment_name_lower:
                return {"equipment": key, **value}

        # Check for partial matches
        for key, value in EQUIPMENT_STATUS.items():
            if any(word in key for word in equipment_name_lower.split("_")):
                return {"equipment": key, **value}

        return {"error": f"Equipment '{equipment_name}' not found in lab inventory"}

    def _tool_check_inventory(self, item_name: str) -> Dict[str, Any]:
        """Tool: Check inventory levels."""
        item_name_lower = item_name.lower().replace(" ", "_").replace("-", "_")
        for key, value in INVENTORY.items():
            if item_name_lower in key.lower() or key.lower() in item_name_lower:
                return {"item": key, **value}

        # Check partial matches
        for key, value in INVENTORY.items():
            if any(word in key.lower() for word in item_name_lower.split("_") if len(word) > 2):
                return {"item": key, **value}

        return {"error": f"Item '{item_name}' not found in inventory"}

    def _tool_get_protocol(self, protocol_name: str) -> Dict[str, Any]:
        """Tool: Retrieve a standard protocol."""
        protocol_name_lower = protocol_name.lower().replace(" ", "_").replace("-", "_")
        for key, value in PROTOCOLS.items():
            if protocol_name_lower in key or key in protocol_name_lower:
                return {"protocol": key, "steps": value}

        # Check partial matches
        for key, value in PROTOCOLS.items():
            if any(word in key for word in protocol_name_lower.split("_") if len(word) > 2):
                return {"protocol": key, "steps": value}

        return {"error": f"Protocol '{protocol_name}' not found"}

    def _tool_reserve_equipment(self, equipment_name: str, time: str) -> Dict[str, Any]:
        """Tool: Reserve equipment."""
        equipment_info = self._tool_check_equipment(equipment_name)
        if "error" in equipment_info:
            return equipment_info

        if equipment_info.get("status") == "maintenance":
            return {"success": False, "reason": "Equipment is under maintenance"}

        # Simulate reservation
        reservation = {
            "equipment": equipment_info["equipment"],
            "time": time,
            "status": "confirmed"
        }
        self.lab_state["pending_reservations"].append(reservation)
        return {"success": True, "reservation": reservation}

    # =========================================================================
    # COMPONENT 4: REASONING (Agent Loop)
    # =========================================================================

    def _decide_tools_needed(self, user_request: str) -> List[Dict]:
        """Use LLM to decide which tools to call based on user request."""

        tools_description = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])

        prompt = f"""Given this user request, decide which tools to use to gather information.

USER REQUEST: {user_request}

AVAILABLE TOOLS:
{tools_description}

Respond with JSON containing a list of tool calls:
{{
    "tool_calls": [
        {{"tool": "tool_name", "args": {{"param": "value"}}}},
        ...
    ],
    "reasoning": "Brief explanation of why these tools are needed"
}}

If no tools are needed (e.g., general question), return empty tool_calls list.
Choose tools that will help you provide a complete answer."""

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a tool selection assistant. Analyze requests and select appropriate tools."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )

        result = json.loads(response.choices[0].message.content)
        return result.get("tool_calls", []), result.get("reasoning", "")

    def _execute_tools(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute the selected tools and collect observations."""
        observations = []
        for call in tool_calls:
            tool_name = call.get("tool")
            args = call.get("args", {})

            if tool_name in self.tools:
                tool_fn = self.tools[tool_name]["function"]
                try:
                    # Normalize argument names for flexibility
                    if tool_name == "check_equipment":
                        # Accept various key names
                        eq_name = args.get("equipment_name") or args.get("equipment") or args.get("name", "")
                        result = tool_fn(eq_name)
                    elif tool_name == "check_inventory":
                        item = args.get("item_name") or args.get("item") or args.get("name", "")
                        result = tool_fn(item)
                    elif tool_name == "get_protocol":
                        proto = args.get("protocol_name") or args.get("protocol") or args.get("name", "")
                        result = tool_fn(proto)
                    elif tool_name == "reserve_equipment":
                        eq_name = args.get("equipment_name") or args.get("equipment") or args.get("name", "")
                        time = args.get("time", "")
                        result = tool_fn(eq_name, time)
                    else:
                        result = tool_fn(**args) if isinstance(args, dict) else tool_fn(args)
                    observations.append({
                        "tool": tool_name,
                        "args": args,
                        "result": result
                    })
                except Exception as e:
                    observations.append({
                        "tool": tool_name,
                        "args": args,
                        "error": str(e)
                    })
            else:
                observations.append({
                    "tool": tool_name,
                    "error": f"Unknown tool: {tool_name}"
                })

        return observations

    def _synthesize_response(self, user_request: str, observations: List[Dict]) -> str:
        """Use LLM to synthesize a helpful response from observations."""

        observations_text = json.dumps(observations, indent=2) if observations else "No tools were used."

        prompt = f"""Based on the user's request and the information gathered, provide a helpful response.

USER REQUEST: {user_request}

INFORMATION GATHERED:
{observations_text}

LAB STATE:
- Pending reservations: {len(self.lab_state['pending_reservations'])}
- Flagged issues: {self.lab_state['flagged_issues']}

Provide a concise, helpful response that:
1. Directly answers the user's question
2. Mentions any relevant issues found (low supplies, unavailable equipment)
3. Suggests alternatives if something is unavailable
4. Is actionable and specific"""

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    def handle_request(self, user_request: str) -> Dict[str, Any]:
        """
        Main agent loop: Perceive -> Think -> Act -> Observe -> Respond

        This demonstrates the full agent cycle with explicit component usage.
        """
        print(f"\n{'─'*60}")
        print(f"[PERCEPTION] Received request: {user_request[:50]}...")

        # COMPONENT 3: Update memory
        self.conversation_history.append({"role": "user", "content": user_request})
        self.lab_state["recent_queries"].append(user_request)

        # COMPONENT 4: REASONING - Decide what tools to use
        print(f"\n[REASONING] Deciding which tools to use...")
        tool_calls, reasoning = self._decide_tools_needed(user_request)
        print(f"  Reasoning: {reasoning}")
        print(f"  Tools selected: {[t['tool'] for t in tool_calls] if tool_calls else 'None'}")

        # COMPONENT 2: TOOLS - Execute selected tools
        observations = []
        if tool_calls:
            print(f"\n[ACTION] Executing tools...")
            observations = self._execute_tools(tool_calls)
            for obs in observations:
                print(f"  Tool: {obs['tool']}")
                result = obs.get('result', obs.get('error', 'Unknown'))
                if isinstance(result, dict):
                    print(f"    Result: {json.dumps(result, indent=2)[:100]}...")
                else:
                    print(f"    Result: {result}")

        # Check for issues to flag (proactive behavior)
        for obs in observations:
            result = obs.get("result", {})
            if isinstance(result, dict):
                if result.get("status") == "low":
                    issue = f"Low inventory: {result.get('item', 'unknown')}"
                    if issue not in self.lab_state["flagged_issues"]:
                        self.lab_state["flagged_issues"].append(issue)
                        print(f"\n[MEMORY] Flagged issue: {issue}")
                if result.get("status") == "maintenance":
                    issue = f"Equipment maintenance: {result.get('equipment', 'unknown')}"
                    if issue not in self.lab_state["flagged_issues"]:
                        self.lab_state["flagged_issues"].append(issue)
                        print(f"\n[MEMORY] Flagged issue: {issue}")

        # COMPONENT 4: REASONING - Synthesize response
        print(f"\n[REASONING] Synthesizing response...")
        response = self._synthesize_response(user_request, observations)

        # COMPONENT 3: Update memory
        self.conversation_history.append({"role": "assistant", "content": response})

        print(f"\n[RESPONSE]")
        print(f"{'─'*60}")

        return {
            "response": response,
            "tools_used": [t["tool"] for t in tool_calls],
            "observations": observations,
            "reasoning": reasoning
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_agent_components(agent: LabManagerAgent) -> None:
    """Show the agent's component architecture."""
    print("\n" + "═" * 70)
    print("AGENT COMPONENT ARCHITECTURE")
    print("═" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                      LabManagerAgent                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ COMPONENT 1: GOALS (System Prompt)                          │   │
│  │   • Ensure lab safety and efficiency                        │   │
│  │   • Help researchers with equipment and supplies            │   │
│  │   • Proactively flag issues                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ COMPONENT 2: TOOLS                                          │   │
│  │   • check_equipment() - Query equipment availability        │   │
│  │   • check_inventory() - Query supply levels                 │   │
│  │   • get_protocol() - Retrieve standard protocols            │   │
│  │   • reserve_equipment() - Make reservations                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ COMPONENT 3: MEMORY                                         │   │
│  │   • Conversation history (context)                          │   │
│  │   • Lab state (reservations, flagged issues)                │   │
│  │   • Recent queries                                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ COMPONENT 4: REASONING (LLM)                                │   │
│  │   • Decide which tools to use                               │   │
│  │   • Synthesize information from tools                       │   │
│  │   • Generate helpful, contextual responses                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

AGENT LOOP: Perceive → Think → Act → Observe → Respond
""")


def demonstrate_agent_loop(agent: LabManagerAgent) -> None:
    """Demonstrate the agent loop with sample requests."""
    print("\n" + "═" * 70)
    print("AGENT LOOP DEMONSTRATION")
    print("═" * 70)

    requests = [
        "I need to run a PCR tomorrow. Is the thermal cycler available and do we have Taq polymerase?",
        "What's the protocol for DNA extraction?",
        "Can you reserve the sequencer for me at 2pm?",
    ]

    for i, request in enumerate(requests, 1):
        print(f"\n{'═'*70}")
        print(f"REQUEST {i}")
        print(f"{'═'*70}")

        result = agent.handle_request(request)
        print(result["response"])


def demonstrate_memory_persistence(agent: LabManagerAgent) -> None:
    """Show how memory persists across requests."""
    print("\n" + "═" * 70)
    print("MEMORY PERSISTENCE DEMONSTRATION")
    print("═" * 70)

    print(f"\nAgent Memory State:")
    print(f"  Conversation turns: {len(agent.conversation_history)}")
    print(f"  Pending reservations: {len(agent.lab_state['pending_reservations'])}")
    print(f"  Flagged issues: {agent.lab_state['flagged_issues']}")
    print(f"  Recent queries: {len(agent.lab_state['recent_queries'])}")

    if agent.lab_state["pending_reservations"]:
        print(f"\n  Reservations made this session:")
        for res in agent.lab_state["pending_reservations"]:
            print(f"    - {res['equipment']} at {res['time']}")


def main():
    """Main demo entry point."""
    print("=" * 70)
    print("Module 2 Demo: Agent Anatomy - The Lab Manager Agent")
    print("=" * 70)
    print("\nLearning Objective:")
    print("  Define what constitutes a modern AI agent and identify")
    print("  its fundamental components: Goals, Tools, Memory, Reasoning")

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UDACITY_OPENAI_API_KEY")

    if not api_key:
        print("\nError: OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

    # Create agent
    agent = LabManagerAgent(client)

    # Demo 1: Show component architecture
    demonstrate_agent_components(agent)

    # Demo 2: Show agent loop in action
    demonstrate_agent_loop(agent)

    # Demo 3: Show memory persistence
    demonstrate_memory_persistence(agent)

    print("\n" + "═" * 70)
    print("KEY TAKEAWAYS")
    print("═" * 70)
    print("""
1. AGENTS = Goals + Tools + Memory + Reasoning
   - Goals define WHAT the agent tries to achieve
   - Tools define WHAT the agent CAN DO
   - Memory tracks WHAT the agent KNOWS
   - Reasoning determines WHAT the agent DECIDES

2. The AGENT LOOP: Perceive → Think → Act → Observe → Respond
   - Each request triggers the full cycle
   - Tools are selected based on reasoning
   - Observations inform the final response

3. MEMORY enables CONTEXT
   - Conversation history provides context
   - State tracking enables proactive behavior
   - Issues flagged persist across requests

4. This is DIFFERENT from a simple LLM call:
   - Simple LLM: Input → Output (stateless)
   - Agent: Input → Reason → Tools → Observe → Output (stateful)
""")
    print("═" * 70)


if __name__ == "__main__":
    main()
