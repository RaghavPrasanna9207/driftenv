"""Environment implementation for DriftEnv."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import networkx as nx

from environment.action_parser import parse_action
from environment.graph_engine import GraphEngine
from environment.observation import build_observation
from environment.reward import (
    compute_r1,
    compute_r2,
    compute_r3,
    compute_r4,
    compute_r5,
)


class OpenEnv(ABC):
    """Minimal interface for interactive step-based environments."""

    @abstractmethod
    def reset(self) -> str:
        """Reset the environment and return the initial observation."""

    @abstractmethod
    def step(self, action_text: str) -> dict[str, Any]:
        """Advance the environment one step using an action string."""


class DriftEnv(OpenEnv):
    """DriftEnv implementation built on top of the local graph utilities.

    The environment tracks a hidden mutation schedule, exposes text
    observations for an LLM agent, and scores agent behavior with the five
    reward components defined in :mod:`environment.reward`.
    """

    MAX_TURNS = 12

    def __init__(self, episode_type: str, curriculum_stage: int, seed: int) -> None:
        """Initialize environment configuration without starting an episode.

        Args:
            episode_type: The scenario label for the episode.
            curriculum_stage: Curriculum difficulty stage.
            seed: Random seed used for deterministic mutation scheduling.
        """

        self.init(episode_type, curriculum_stage, seed)

    def init(self, episode_type: str, curriculum_stage: int, seed: int) -> None:
        """Set up internal state and supporting objects without resetting."""

        self.episode_type = episode_type
        self.curriculum_stage = curriculum_stage
        self.seed = seed
        self._rng = random.Random(seed)
        self.graph_engine = GraphEngine()
        self.reference_engine = GraphEngine()
        self.base_graph_path = (
            Path(__file__).resolve().parents[1] / "graph_templates" / "base_graph.json"
        )

        self.turn = 1
        self.done = False
        self.mutation_schedule: list[tuple[int, str, dict[str, Any]]] = []
        self.flagged_nodes: list[Any] = []
        self.proposed_edits: list[dict[str, Any]] = []
        self.tasks_resolved = False

        self.agent_belief_graph: nx.DiGraph = nx.DiGraph()
        self.mutation_history: list[dict[str, Any]] = []
        self.active_inconsistency_signals: list[str] = []
        self.mutated_nodes: list[Any] = []
        self.pending_ground_truth_delta: list[dict[str, Any]] = []
        self.repair_steps_taken = 0
        self.last_action_result: dict[str, Any] = {"success": False, "inconsistency_signals": []}
        self.latest_focus_nodes: list[Any] = ["P1"]

    def reset(self) -> str:
        """Start a fresh episode and return the initial observation string."""

        self.turn = 1
        self.done = False
        self.flagged_nodes = []
        self.proposed_edits = []
        self.tasks_resolved = False
        self.mutation_history = []
        self.active_inconsistency_signals = []
        self.mutated_nodes = []
        self.pending_ground_truth_delta = []
        self.repair_steps_taken = 0
        self.last_action_result = {"success": False, "inconsistency_signals": []}

        self.graph_engine.load_graph(str(self.base_graph_path))
        self.reference_engine.load_graph(str(self.base_graph_path))
        self.agent_belief_graph = self.reference_engine.graph.copy()
        self.mutation_schedule = self._generate_mutation_schedule()
        self.latest_focus_nodes = ["P1"]

        return self._build_observation([])

    def step(self, action_text: str) -> dict[str, Any]:
        """Process one agent action, apply due mutations, and compute rewards.

        Args:
            action_text: Raw text produced by the agent.

        Returns:
            A transition dictionary with keys ``observation``, ``reward``,
            ``done``, and ``info``.
        """

        if self.done:
            return {
                "observation": self._build_observation(["Episode already completed."]),
                "reward": 0.0,
                "done": True,
                "info": {
                    "reward_breakdown": {
                        "r1": 0.0,
                        "r2": 0.0,
                        "r3": 0.0,
                        "r4": 0.0,
                        "r5": 0.0,
                    }
                },
            }

        parsed_action = parse_action(action_text)
        self._apply_scheduled_mutations(self.turn)
        action_result = self._execute_action(parsed_action)
        self.last_action_result = action_result

        r1 = (
            compute_r1(
                self._normalize_action_for_r1(parsed_action),
                self._build_action_ground_truth(parsed_action),
            )
            if parsed_action.get("valid", False)
            else 0.0
        )
        r2 = self._compute_graph_accuracy_reward()
        r3 = compute_r3(self.flagged_nodes, self.mutated_nodes)
        r4 = compute_r4(self.proposed_edits, self.pending_ground_truth_delta)
        repair_steps_taken, minimum_required_steps = self._repair_efficiency_inputs()
        r5 = compute_r5(repair_steps_taken, minimum_required_steps)
        reward = (0.25 * r1) + (0.25 * r2) + (0.20 * r3) + (0.20 * r4) + (0.10 * r5)

        self.tasks_resolved = self._check_tasks_resolved()
        self.turn += 1
        self.done = self.tasks_resolved or self.turn > self.MAX_TURNS

        combined_signals = list(self.active_inconsistency_signals)
        combined_signals.extend(action_result.get("inconsistency_signals", []))
        observation = self._build_observation(combined_signals)

        return {
            "observation": observation,
            "reward": reward,
            "done": self.done,
            "info": {
                "reward_breakdown": {
                    "r1": r1,
                    "r2": r2,
                    "r3": r3,
                    "r4": r4,
                    "r5": r5,
                },
                "parsed_action": parsed_action,
                "action_result": action_result,
                "reward_inputs": {
                    "mutated_nodes": list(self.mutated_nodes),
                    "flagged_nodes": list(self.flagged_nodes),
                    "proposed_edits_count": len(self.proposed_edits),
                    "ground_truth_delta_count": len(self.pending_ground_truth_delta),
                    "repair_steps_taken": repair_steps_taken,
                    "minimum_required_steps": minimum_required_steps,
                    "graph_reward_nodes": sorted(
                        (str(node_id) for node_id in self._graph_reward_node_set())
                    ),
                },
                "turn": self.turn - 1,
                "tasks_resolved": self.tasks_resolved,
            },
        }

    def _apply_scheduled_mutations(self, turn: int) -> None:
        """Apply all hidden mutations scheduled for the specified turn."""

        due_mutations = [entry for entry in self.mutation_schedule if entry[0] == turn]
        if not due_mutations:
            return

        remaining_schedule = [entry for entry in self.mutation_schedule if entry[0] != turn]
        self.mutation_schedule = remaining_schedule

        for scheduled_turn, mutation_type, params in due_mutations:
            before_state = self._capture_graph_state(self.graph_engine.graph)
            change_record = self.graph_engine.apply_mutation(mutation_type, params)
            after_state = self._capture_graph_state(self.graph_engine.graph)
            ground_truth_delta = self.graph_engine.get_ground_truth_delta(before_state, after_state)

            self.mutation_history.append(
                {
                    "turn": scheduled_turn,
                    "mutation_type": mutation_type,
                    "params": dict(params),
                    "change_record": change_record,
                    "ground_truth_delta": ground_truth_delta,
                }
            )
            self.pending_ground_truth_delta.extend(ground_truth_delta)

            mutated_node = change_record.get("removed_node", {}).get("id")
            if mutated_node is not None:
                self.mutated_nodes.append(mutated_node)
                self.latest_focus_nodes = [mutated_node]

            self.active_inconsistency_signals.extend(
                self._signals_from_mutation(mutation_type, change_record)
            )

    def _execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Route an action to an EA or repair handler and return its outcome."""

        if not action.get("valid", False):
            return {
                "success": False,
                "inconsistency_signals": [action.get("error_message", "Invalid action.")],
            }

        action_type = action["action_type"]
        params = action.get("params", {})

        if action_type in {
            "draft_reply",
            "reschedule_meeting",
            "delegate_task",
            "escalate",
            "decline_meeting",
        }:
            return self._handle_ea_action(action_type, params)

        return self._handle_repair_action(action_type, params)

    def _handle_ea_action(self, action_type: str, params: dict[str, Any]) -> dict[str, Any]:
        """Handle one executive-assistant action."""

        required_fields = {
            "draft_reply": {"recipient_id", "message"},
            "reschedule_meeting": {"meeting_id", "new_time"},
            "delegate_task": {"task_id", "assignee_id"},
            "escalate": {"issue_id", "target_id"},
            "decline_meeting": {"meeting_id", "reason"},
        }
        missing_fields = sorted(
            field for field in required_fields.get(action_type, set()) if field not in params
        )
        if missing_fields:
            return {
                "success": False,
                "inconsistency_signals": [
                    f"Action {action_type} is missing required params: {', '.join(missing_fields)}."
                ],
            }

        referenced_nodes = self._extract_referenced_nodes(params)
        invalid_nodes = [
            node_id for node_id in referenced_nodes if node_id not in self.graph_engine.graph
        ]
        signals: list[str] = []
        if invalid_nodes:
            signals.append(
                f"Action references unknown or missing graph nodes: {', '.join(map(str, invalid_nodes))}."
            )

        if self.active_inconsistency_signals:
            signals.append(
                "Background graph inconsistencies remain unresolved while handling the EA task."
            )

        success = not invalid_nodes and not missing_fields
        return {"success": success, "inconsistency_signals": signals}

    def _handle_repair_action(self, action_type: str, params: dict[str, Any]) -> dict[str, Any]:
        """Handle one repair-oriented action."""

        self.repair_steps_taken += 1
        signals: list[str] = []
        success = True

        if action_type == "flag_inconsistency":
            flagged = self._coerce_node_list(params)
            if not flagged:
                success = False
                signals.append("flag_inconsistency requires 'node_id' or 'node_ids'.")
            else:
                self.flagged_nodes.extend(flagged)
                self.latest_focus_nodes = flagged
                for node_id in flagged:
                    if node_id not in self.graph_engine.graph:
                        signals.append(f"Flagged node {node_id} is currently absent from the graph.")

        elif action_type == "propose_node_removal":
            node_id = params.get("node_id")
            if node_id is None:
                success = False
                signals.append("propose_node_removal requires 'node_id'.")
            else:
                self.flagged_nodes.append(node_id)
                self.latest_focus_nodes = [node_id]
                self._apply_node_removal_to_belief_graph(node_id)
                signals.append(f"Proposed removing node {node_id} from the agent belief graph.")

        elif action_type == "propose_edge_update":
            source = params.get("source")
            old_target = params.get("old_target")
            new_target = params.get("new_target")
            if source is None or old_target is None or new_target is None:
                success = False
                signals.append(
                    "propose_edge_update requires 'source', 'old_target', and 'new_target'."
                )
            else:
                self.flagged_nodes.extend(
                    node_id for node_id in (source, old_target, new_target) if node_id is not None
                )
                self.latest_focus_nodes = [source, old_target, new_target]
                self._apply_edge_update_to_belief_graph(source, old_target, new_target)
                signals.append(
                    f"Proposed rewiring edge {source}->{old_target} to {source}->{new_target}."
                )

        elif action_type == "propose_attribute_update":
            node_id = params.get("node_id")
            attribute_name = params.get("attribute_name")
            new_value = params.get("new_value")
            if node_id is None or attribute_name is None:
                success = False
                signals.append(
                    "propose_attribute_update requires 'node_id' and 'attribute_name'."
                )
            else:
                self.flagged_nodes.append(node_id)
                self.latest_focus_nodes = [node_id]
                self._apply_attribute_update_to_belief_graph(node_id, attribute_name, new_value)
                signals.append(
                    f"Proposed updating attribute {attribute_name} on node {node_id}."
                )

        elif action_type == "request_clarification":
            question = params.get("question", "No clarification question provided.")
            signals.append(f"Clarification requested: {question}")

        else:
            success = False
            signals.append(f"Unhandled repair action type: {action_type}.")

        return {"success": success, "inconsistency_signals": signals}

    def _generate_mutation_schedule(self) -> list[tuple[int, str, dict[str, Any]]]:
        """Create the hidden mutation schedule for the next episode."""

        nodes_by_priority = [
            node_id
            for node_id in sorted(self.graph_engine.graph.nodes(), key=str)
            if self.graph_engine.graph.degree(node_id) > 0 and node_id not in {"P1", "P10"}
        ]
        if not nodes_by_priority:
            nodes_by_priority = sorted(self.graph_engine.graph.nodes(), key=str)

        if self.curriculum_stage <= 1:
            scheduled_turn = self._rng.randint(5, self.MAX_TURNS)
            node_id = self._rng.choice(nodes_by_priority)
            return [(scheduled_turn, "node_removal", {"node_id": node_id})]

        mutation_count = min(max(1, self.curriculum_stage), 3)
        used_nodes: set[Any] = set()
        schedule: list[tuple[int, str, dict[str, Any]]] = []
        available_turns = list(range(5, self.MAX_TURNS + 1))
        self._rng.shuffle(available_turns)
        for index in range(mutation_count):
            turn = available_turns[index]
            remaining_nodes = [node_id for node_id in nodes_by_priority if node_id not in used_nodes]
            if not remaining_nodes:
                remaining_nodes = nodes_by_priority
            node_id = self._rng.choice(remaining_nodes)
            used_nodes.add(node_id)
            schedule.append((turn, "node_removal", {"node_id": node_id}))
        return sorted(schedule, key=lambda item: item[0])

    def _build_observation(self, extra_signals: list[str]) -> str:
        """Build the current prompt observation for the agent."""

        centers = self._resolve_observation_centers()
        subgraph = self.graph_engine.get_subgraph(centers, hops=2)
        if subgraph.number_of_nodes() == 0:
            fallback_centers = [node_id for node_id in ["P1", "P10"] if node_id in self.graph_engine.graph]
            subgraph = self.graph_engine.get_subgraph(fallback_centers, hops=2)

        subgraph_text = self.graph_engine.serialize_subgraph(subgraph)
        signals = list(dict.fromkeys(self.active_inconsistency_signals + extra_signals))

        return build_observation(
            subgraph_text=subgraph_text,
            task_description=self._build_task_description(),
            inconsistency_signals=signals,
            mutation_history=[self._format_mutation_history_entry(entry) for entry in self.mutation_history],
            active_policy_excerpt=self._build_policy_excerpt(),
            turn_number=min(self.turn, self.MAX_TURNS),
        )

    def _build_task_description(self) -> str:
        """Return the high-level task description for the active episode."""

        episode_descriptions = {
            "manager_departure": (
                "Act as an executive assistant that keeps the org graph coherent while "
                "handling communication and delegation tasks."
            ),
            "policy_injection": (
                "Act as an executive assistant that follows policy constraints and "
                "repairs graph inconsistencies when they appear."
            ),
        }
        base_description = episode_descriptions.get(
            self.episode_type,
            "Act as an executive assistant operating over an organizational graph. "
            "Complete the task, detect graph drift, and propose repairs when needed.",
        )

        if self.pending_ground_truth_delta:
            return (
                f"{base_description} Hidden mutations may have introduced inconsistencies. "
                "Inspect the observation, flag likely drift, and keep your internal graph "
                "hypothesis aligned with the live graph."
            )
        return base_description

    def _build_policy_excerpt(self) -> str:
        """Return a compact policy excerpt for the current curriculum stage."""

        base_excerpt = (
            "Prefer precise, schema-valid actions. Reference only stakeholders or nodes "
            "supported by the observed graph context. When you detect a graph mismatch, "
            "flag the inconsistency before proposing a repair."
        )
        if self.curriculum_stage <= 1:
            return (
                f"{base_excerpt} Stage 1 focuses on identifying and reasoning about a single "
                "structural mutation that may appear after several turns."
            )
        return (
            f"{base_excerpt} Higher curriculum stages may involve multiple mutations across "
            "different turns."
        )

    def _build_action_ground_truth(self, parsed_action: dict[str, Any]) -> dict[str, Any]:
        """Construct a lightweight ground-truth schema for action-validity scoring."""

        action_type = parsed_action.get("action_type")
        required_fields_map = {
            "draft_reply": ["recipient_id", "message"],
            "reschedule_meeting": ["meeting_id", "new_time"],
            "delegate_task": ["task_id", "assignee_id"],
            "escalate": ["issue_id", "target_id"],
            "decline_meeting": ["meeting_id", "reason"],
            "flag_inconsistency": ["node_id"],
            "propose_node_removal": ["node_id"],
            "propose_edge_update": ["source", "old_target", "new_target"],
            "propose_attribute_update": ["node_id", "attribute_name", "new_value"],
            "request_clarification": ["question"],
        }
        if action_type == "flag_inconsistency" and "node_ids" in parsed_action.get("params", {}):
            required_fields_map["flag_inconsistency"] = ["node_ids"]

        focus_nodes = self._resolve_observation_centers()
        if not focus_nodes:
            focus_nodes = ["P1"]
        subgraph = self.graph_engine.get_subgraph(focus_nodes, hops=2)
        valid_nodes = list(subgraph.nodes()) if subgraph.number_of_nodes() > 0 else list(self.graph_engine.graph.nodes())[:15]
        valid_edges = list(subgraph.edges()) if subgraph.number_of_edges() > 0 else list(self.graph_engine.graph.edges())[:20]

        return {
            "stakeholders": focus_nodes,
            "required_fields": required_fields_map.get(action_type, []),
            "valid_nodes": valid_nodes,
            "valid_edges": valid_edges,
        }

    def _normalize_action_for_r1(self, parsed_action: dict[str, Any]) -> dict[str, Any]:
        """Normalize a parsed action into the shape expected by ``compute_r1``."""

        params = parsed_action.get("params", {})
        stakeholders = []
        explicit_stakeholders = params.get("stakeholders")
        if isinstance(explicit_stakeholders, list):
            stakeholders.extend(explicit_stakeholders)
        elif explicit_stakeholders is not None:
            stakeholders.append(explicit_stakeholders)
        stakeholders.extend(self._extract_referenced_nodes(params))

        route = params.get("route")
        if route is None:
            route_candidates = [params.get("source"), params.get("old_target"), params.get("new_target")]
            route = [item for item in route_candidates if item is not None]

        return {
            "stakeholders": stakeholders,
            "fields": params if isinstance(params, dict) else {},
            "route": route if isinstance(route, list) else [],
        }

    def _resolve_observation_centers(self) -> list[Any]:
        """Determine which nodes should anchor the current observation subgraph."""

        candidates: list[Any] = []
        candidates.extend(self.flagged_nodes)
        candidates.extend(self.latest_focus_nodes)
        candidates.extend(self.mutated_nodes)

        existing = [node_id for node_id in candidates if node_id in self.graph_engine.graph]
        if existing:
            return list(dict.fromkeys(existing))

        for entry in reversed(self.mutation_history):
            removed_edges = entry.get("change_record", {}).get("removed_edges", [])
            for edge in removed_edges:
                for endpoint in (edge.get("source"), edge.get("target")):
                    if endpoint in self.graph_engine.graph:
                        candidates.append(endpoint)

        existing = [node_id for node_id in candidates if node_id in self.graph_engine.graph]
        if existing:
            return list(dict.fromkeys(existing))

        return [node_id for node_id in ["P1", "P10"] if node_id in self.graph_engine.graph]

    def _signals_from_mutation(
        self, mutation_type: str, change_record: dict[str, Any]
    ) -> list[str]:
        """Generate observation-facing inconsistency signals from a mutation."""

        if mutation_type == "node_removal":
            removed_node = change_record.get("removed_node", {}).get("id", "unknown")
            removed_edges = change_record.get("removed_edges", [])
            return [
                f"Potential inconsistency detected: node {removed_node} disappeared from the live graph.",
                f"Related graph evidence: {len(removed_edges)} incident edges were removed.",
            ]
        if mutation_type == "edge_rewiring":
            removed_edge = change_record.get("removed_edge", {})
            added_edge = change_record.get("added_edge", {})
            return [
                "Potential inconsistency detected: an existing relationship changed targets.",
                (
                    "Related graph evidence: "
                    f"{removed_edge.get('source')}->{removed_edge.get('target')} "
                    f"became {added_edge.get('source')}->{added_edge.get('target')}."
                ),
            ]
        return ["Potential inconsistency detected in the live graph."]

    def _format_mutation_history_entry(self, entry: dict[str, Any]) -> str:
        """Render one mutation-history entry into a concise text line."""

        change_record = entry.get("change_record", {})
        if entry.get("mutation_type") == "node_removal":
            removed_node = change_record.get("removed_node", {}).get("id", "unknown")
            return f"turn={entry.get('turn')} | node_removal | node_id={removed_node}"
        if entry.get("mutation_type") == "edge_rewiring":
            removed_edge = change_record.get("removed_edge", {})
            added_edge = change_record.get("added_edge", {})
            return (
                f"turn={entry.get('turn')} | edge_rewiring | "
                f"{removed_edge.get('source')}->{removed_edge.get('target')} -> "
                f"{added_edge.get('source')}->{added_edge.get('target')}"
            )
        return f"turn={entry.get('turn')} | mutation={entry.get('mutation_type')}"

    def _check_tasks_resolved(self) -> bool:
        """Determine whether the current episode's outstanding tasks are resolved."""

        if not self.mutation_history:
            return False
        graph_match = self._compute_graph_accuracy_reward() == 1.0
        flag_score = compute_r3(self.flagged_nodes, self.mutated_nodes)
        repair_score = compute_r4(self.proposed_edits, self.pending_ground_truth_delta)
        return graph_match and flag_score > 0.0 and repair_score > 0.0

    def _compute_graph_accuracy_reward(self) -> float:
        """Compute graph accuracy on the mutation-affected region when available."""

        reward_nodes = self._graph_reward_node_set()
        if not reward_nodes:
            return compute_r2(self.agent_belief_graph, self.graph_engine.graph)

        agent_nodes = [node_id for node_id in reward_nodes if node_id in self.agent_belief_graph]
        truth_nodes = [node_id for node_id in reward_nodes if node_id in self.graph_engine.graph]
        agent_view = self.agent_belief_graph.subgraph(agent_nodes).copy()
        truth_view = self.graph_engine.graph.subgraph(truth_nodes).copy()
        return compute_r2(agent_view, truth_view)

    def _graph_reward_node_set(self) -> set[Any]:
        """Return nodes whose local graph structure should drive graph reward."""

        if not self.mutation_history:
            return set()

        reward_nodes: set[Any] = set()
        for entry in self.mutation_history:
            change_record = entry.get("change_record", {})
            removed_node = change_record.get("removed_node", {}).get("id")
            if removed_node is not None:
                reward_nodes.add(removed_node)

            for edge_key in ("removed_edge", "added_edge", "previous_new_edge"):
                edge = change_record.get(edge_key)
                if isinstance(edge, dict):
                    source = edge.get("source")
                    target = edge.get("target")
                    if source is not None:
                        reward_nodes.add(source)
                    if target is not None:
                        reward_nodes.add(target)

            for edge in change_record.get("removed_edges", []):
                source = edge.get("source")
                target = edge.get("target")
                if source is not None:
                    reward_nodes.add(source)
                if target is not None:
                    reward_nodes.add(target)

            for delta in entry.get("ground_truth_delta", []):
                node_id = delta.get("id")
                source = delta.get("source")
                target = delta.get("target")
                if node_id is not None:
                    reward_nodes.add(node_id)
                if source is not None:
                    reward_nodes.add(source)
                if target is not None:
                    reward_nodes.add(target)

        return reward_nodes

    def _repair_efficiency_inputs(self) -> tuple[int, int]:
        """Return inputs for the repair-efficiency reward component."""

        if not self.mutation_history:
            return self.repair_steps_taken, 0

        first_mutation_turn = min(int(entry.get("turn", self.turn)) for entry in self.mutation_history)
        unresolved_turns = max(0, self.turn - first_mutation_turn + 1)
        effective_repair_steps = max(self.repair_steps_taken, unresolved_turns)
        minimum_required_steps = max(1, len(self.mutated_nodes) * 2)
        return effective_repair_steps, minimum_required_steps

    def _apply_node_removal_to_belief_graph(self, node_id: Any) -> None:
        """Apply a node-removal proposal to the agent belief graph."""

        if node_id in self.agent_belief_graph:
            incoming = list(self.agent_belief_graph.in_edges(node_id, data=True))
            outgoing = list(self.agent_belief_graph.out_edges(node_id, data=True))
            for source, target, data in incoming + outgoing:
                self.proposed_edits.append(
                    {
                        "op": "remove_edge",
                        "source": source,
                        "target": target,
                        "type": data.get("type"),
                        "attributes": {
                            key: value for key, value in data.items() if key != "type"
                        },
                    }
                )

            node_data = self.agent_belief_graph.nodes[node_id]
            self.proposed_edits.append(
                {
                    "op": "remove_node",
                    "id": node_id,
                    "type": node_data.get("type"),
                    "attributes": dict(node_data.get("attributes", {})),
                }
            )
            self.agent_belief_graph.remove_node(node_id)
            return

        self.proposed_edits.append({"op": "remove_node", "id": node_id, "type": None, "attributes": {}})

    def _apply_edge_update_to_belief_graph(
        self, source: Any, old_target: Any, new_target: Any
    ) -> None:
        """Apply an edge-rewiring proposal to the agent belief graph."""

        previous_data = (
            dict(self.agent_belief_graph.edges[source, old_target])
            if self.agent_belief_graph.has_edge(source, old_target)
            else {}
        )
        if self.agent_belief_graph.has_edge(source, old_target):
            self.agent_belief_graph.remove_edge(source, old_target)

        if source not in self.agent_belief_graph:
            self.agent_belief_graph.add_node(source, type=None, attributes={})
        if new_target not in self.agent_belief_graph:
            self.agent_belief_graph.add_node(new_target, type=None, attributes={})
        self.agent_belief_graph.add_edge(source, new_target, **previous_data)

        self.proposed_edits.append(
            {
                "op": "remove_edge",
                "source": source,
                "target": old_target,
                "type": previous_data.get("type"),
                "attributes": {
                    key: value for key, value in previous_data.items() if key != "type"
                },
            }
        )
        self.proposed_edits.append(
            {
                "op": "add_edge",
                "source": source,
                "target": new_target,
                "type": previous_data.get("type"),
                "attributes": {
                    key: value for key, value in previous_data.items() if key != "type"
                },
            }
        )

    def _apply_attribute_update_to_belief_graph(
        self, node_id: Any, attribute_name: Any, new_value: Any
    ) -> None:
        """Apply a node-attribute proposal to the agent belief graph."""

        if node_id not in self.agent_belief_graph:
            self.agent_belief_graph.add_node(node_id, type=None, attributes={})

        node_data = self.agent_belief_graph.nodes[node_id]
        before_record = {
            "type": node_data.get("type"),
            "attributes": dict(node_data.get("attributes", {})),
        }
        attributes = dict(node_data.get("attributes", {}))
        attributes[str(attribute_name)] = new_value
        node_data["attributes"] = attributes
        after_record = {
            "type": node_data.get("type"),
            "attributes": dict(node_data.get("attributes", {})),
        }
        self.proposed_edits.append(
            {
                "op": "update_node",
                "id": node_id,
                "before": before_record,
                "after": after_record,
            }
        )

    def _capture_graph_state(self, graph: nx.DiGraph) -> dict[str, Any]:
        """Capture a graph into the normalized state format used by GraphEngine."""

        nodes = []
        for node_id, data in sorted(graph.nodes(data=True), key=lambda item: str(item[0])):
            nodes.append(
                {
                    "id": node_id,
                    "type": data.get("type"),
                    "attributes": dict(data.get("attributes", {})),
                }
            )

        edges = []
        for source, target, data in sorted(
            graph.edges(data=True),
            key=lambda item: (str(item[0]), str(item[1])),
        ):
            edge_record = {
                "source": source,
                "target": target,
                "type": data.get("type"),
            }
            edge_record.update({key: value for key, value in data.items() if key != "type"})
            edges.append(edge_record)

        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _extract_referenced_nodes(params: dict[str, Any]) -> list[Any]:
        """Extract likely graph node references from action params."""

        referenced: list[Any] = []
        for key, value in params.items():
            if key.endswith("_id") and value is not None:
                referenced.append(value)
            elif key.endswith("_ids") and isinstance(value, list):
                referenced.extend(item for item in value if item is not None)
        return referenced

    @staticmethod
    def _coerce_node_list(params: dict[str, Any]) -> list[Any]:
        """Extract one or more node IDs from a repair action payload."""

        if "node_ids" in params and isinstance(params["node_ids"], list):
            return [node_id for node_id in params["node_ids"] if node_id is not None]
        if "node_id" in params and params["node_id"] is not None:
            return [params["node_id"]]
        return []
