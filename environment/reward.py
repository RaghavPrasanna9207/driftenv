"""Reward functions for the DriftEnv reinforcement-learning environment."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import networkx as nx


ACTION_FAILURE_PENALTIES = {
    "invalid_json": -1.0,
    "unsupported_action": -0.75,
    "wrong_parameters": -0.5,
}

# Strengthens contrast between valid actions and “safe” but wrong ones during RL.
R2_ENV_DOMINANCE_SCALE = 0.3  # applied to raw r2 before it enters the weighted sum
R1_IMPORTANCE_SCALE = 1.08  # multiplicative on r1 in [0,1] before weighting; capped below
FORMAT_BONUS_VALID = 0.05
EXPLORATION_BONUS = 0.05

# Parameters whose values are graph node ids (issue_id, meeting_id, task_id, etc. are not validated).
NODE_REFERENCE_PARAM_KEYS = frozenset(
    {
        "node_id",
        "source",
        "old_target",
        "new_target",
        "target_id",
        "recipient_id",
        "assignee_id",
        "from_node_id",
        "to_node_id",
    }
)


def compute_graph_node_reference_penalty(params: dict[str, Any], valid_graph_node_ids: set[str]) -> float:
    """Strong negative when graph-bound fields reference a node that does not exist in the live graph."""

    if not valid_graph_node_ids or not isinstance(params, dict):
        return 0.0
    invalid_count = 0
    for key, value in params.items():
        if key in NODE_REFERENCE_PARAM_KEYS and value is not None:
            if str(value) not in valid_graph_node_ids:
                invalid_count += 1
        if key == "node_ids" and isinstance(value, list):
            for item in value:
                if str(item) not in valid_graph_node_ids:
                    invalid_count += 1
    if invalid_count == 0:
        return 0.0
    return max(-0.75, -0.4 * min(invalid_count, 2))


def compute_action_failure_penalty(validation_status: str) -> float:
    """Return the penalty for a structurally invalid action.

    The environment uses a stronger penalty for malformed JSON than for a
    syntactically valid but unsupported or under-specified action so the
    rollout can still learn from distinct failure modes.
    """

    return ACTION_FAILURE_PENALTIES.get(validation_status, -1.0)


def compute_r1(action: dict[str, Any], ground_truth: dict[str, Any]) -> float:
    """Score action correctness against a ground-truth action specification.

    This reward is intended for high-level action validation in DriftEnv. The
    function evaluates applicable aspects of an agent action and returns the
    arithmetic mean of their scores, clamped to the inclusive range
    ``[0.0, 1.0]``:

    1. Stakeholder correctness:
       The function looks for stakeholder identifiers in the action under the
       keys ``"stakeholders"``, ``"referenced_stakeholders"``,
       ``"stakeholder_ids"``, or ``"targets"``. The same keys are checked in the
       ground truth. A set-based F1 score is used when explicit stakeholder
       targets are available. When no explicit stakeholder targets are
       available, actions can still receive partial credit when referenced
       stakeholders are valid nodes in the allowed graph context.
    2. Required field completion:
       The function looks for required field names in the ground truth under the
       keys ``"required_fields"`` or ``"expected_fields"``. It also accepts a
       mapping under ``"required_field_values"`` or ``"expected_field_values"``,
       in which case the mapping keys are treated as the required fields. A
       required field is counted as complete when it is present either at the
       top level of ``action`` or inside ``action["fields"]`` and its value is
       not empty.
    3. Route validity:
       The function looks for a proposed route in ``action`` under the keys
       ``"route"``, ``"routing_path"``, ``"path"``, ``"nodes"``, or
       ``"route_nodes"``. Valid routing constraints are read from the ground
       truth using ``"valid_nodes"``, ``"allowed_nodes"``, ``"route_nodes"``,
       ``"nodes"``, ``"valid_edges"``, ``"allowed_edges"``, ``"route_edges"``,
       or ``"edges"``. Routing is only scored when the action actually supplies
       a route-like field. When valid nodes are provided, the node portion of
       the score is the fraction of visited route nodes that are allowed. When
       valid directed edges are provided, the edge portion of the score is the
       fraction of consecutive route hops that are allowed. If both node and
       edge constraints are present, the routing score is their average.

    Args:
        action: Agent-produced action data. The function is forgiving about
            schema details and treats missing sections as zero credit for the
            affected sub-score rather than raising an exception.
        ground_truth: Reference action specification describing the correct
            stakeholders, required fields, and valid routing constraints.

    Returns:
        A float in ``[0.0, 1.0]``. A value of ``1.0`` indicates a perfect match
        under the inferred schema; ``0.0`` indicates the action did not satisfy
        any scored requirement.

    Edge Cases:
        - If both the action and ground truth omit a scored section, that
          section is skipped rather than receiving automatic credit.
        - If the ground truth requires stakeholders, fields, or routing but the
          action omits that section, the corresponding sub-score is ``0.0``.
        - If none of the three sections are applicable, the function returns
          ``0.0``.
        - Empty strings, ``None``, empty containers, and missing keys are all
          treated as incomplete field values.
        - If route constraints are provided but the route contains only one
          node, the edge portion is treated as fully satisfied because there are
          no hops to validate.
        - If either input is not a dictionary, the function returns ``0.0``.
    """

    if not isinstance(action, dict) or not isinstance(ground_truth, dict):
        return 0.0

    def pick(mapping: dict[str, Any], candidate_keys: tuple[str, ...]) -> Any:
        for key in candidate_keys:
            if key in mapping:
                return mapping[key]
        return None

    def is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False

    def to_set(value: Any) -> set[Any]:
        if value is None:
            return set()
        if isinstance(value, dict):
            return set(value.keys())
        if isinstance(value, (list, tuple, set)):
            return set(value)
        return {value}

    def to_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, set):
            return list(value)
        return [value]

    component_scores: list[float] = []

    valid_nodes = to_set(
        pick(ground_truth, ("valid_nodes", "allowed_nodes", "route_nodes", "nodes"))
    )
    valid_edges_raw = pick(
        ground_truth,
        ("valid_edges", "allowed_edges", "route_edges", "edges"),
    )
    valid_edges: set[tuple[Any, Any]] = set()
    for edge in to_list(valid_edges_raw):
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            valid_edges.add((edge[0], edge[1]))

    action_stakeholder_value = pick(
        action,
        ("stakeholders", "referenced_stakeholders", "stakeholder_ids", "targets"),
    )
    truth_stakeholder_value = pick(
        ground_truth,
        ("stakeholders", "referenced_stakeholders", "stakeholder_ids", "targets"),
    )
    stakeholder_applicable = (
        action_stakeholder_value is not None or truth_stakeholder_value is not None
    )
    if stakeholder_applicable:
        action_stakeholders = to_set(action_stakeholder_value)
        truth_stakeholders = to_set(truth_stakeholder_value)
        stakeholder_parts: list[float] = []

        if truth_stakeholders:
            stakeholder_overlap = len(action_stakeholders & truth_stakeholders)
            if not action_stakeholders:
                stakeholder_parts.append(0.0)
            else:
                stakeholder_precision = stakeholder_overlap / len(action_stakeholders)
                stakeholder_recall = stakeholder_overlap / len(truth_stakeholders)
                stakeholder_denominator = stakeholder_precision + stakeholder_recall
                stakeholder_parts.append(
                    0.0
                    if stakeholder_denominator == 0.0
                    else (2.0 * stakeholder_precision * stakeholder_recall)
                    / stakeholder_denominator
                )

        if not truth_stakeholders and action_stakeholders and valid_nodes:
            valid_stakeholder_hits = sum(
                1 for stakeholder in action_stakeholders if stakeholder in valid_nodes
            )
            stakeholder_parts.append(valid_stakeholder_hits / len(action_stakeholders))

        stakeholder_score = (
            sum(stakeholder_parts) / len(stakeholder_parts) if stakeholder_parts else 0.0
        )
        component_scores.append(stakeholder_score)

    required_field_values = pick(
        ground_truth,
        ("required_field_values", "expected_field_values"),
    )
    required_fields_raw = pick(ground_truth, ("required_fields", "expected_fields"))
    if isinstance(required_field_values, dict):
        required_fields = list(required_field_values.keys())
    else:
        required_fields = to_list(required_fields_raw)
    action_fields = action.get("fields")
    if not isinstance(action_fields, dict):
        action_fields = {}
    if required_fields:
        completed_fields = 0
        for field_name in required_fields:
            if field_name in action and not is_missing(action[field_name]):
                completed_fields += 1
            elif field_name in action_fields and not is_missing(action_fields[field_name]):
                completed_fields += 1
        field_score = completed_fields / len(required_fields)
        component_scores.append(field_score)

    route_value = pick(action, ("route", "routing_path", "path", "nodes", "route_nodes"))
    route_nodes = to_list(route_value)
    routing_is_applicable = route_value is not None
    if routing_is_applicable:
        routing_components: list[float] = []
        if valid_nodes:
            if not route_nodes:
                routing_components.append(0.0)
            else:
                valid_node_hits = sum(1 for node in route_nodes if node in valid_nodes)
                routing_components.append(valid_node_hits / len(route_nodes))
        if valid_edges:
            if not route_nodes:
                routing_components.append(0.0)
            elif len(route_nodes) == 1:
                routing_components.append(1.0)
            else:
                traversed_edges = list(zip(route_nodes[:-1], route_nodes[1:]))
                valid_edge_hits = sum(1 for edge in traversed_edges if edge in valid_edges)
                routing_components.append(valid_edge_hits / len(traversed_edges))
        if routing_components:
            routing_score = sum(routing_components) / len(routing_components)
            component_scores.append(routing_score)

    if not component_scores:
        return 0.0

    total_score = sum(component_scores) / len(component_scores)
    return max(0.0, min(1.0, total_score))


def compute_r2(agent_graph: nx.DiGraph, ground_truth_graph: nx.DiGraph) -> float:
    """Compute the mean node-F1 and edge-F1 between two directed graphs.

    Nodes are compared by node identifier only, and edges are compared as
    directed ``(source, target)`` pairs. Node and edge attributes are ignored.
    The final reward is the arithmetic mean of the node F1 score and the edge
    F1 score.

    Args:
        agent_graph: The graph state proposed by the agent.
        ground_truth_graph: The reference graph state.

    Returns:
        A float in ``[0.0, 1.0]`` where:
        - ``1.0`` means both the node set and edge set match perfectly.
        - ``0.0`` means there is no overlap in both scored components.

    Edge Cases:
        - If both graphs have no nodes, the node F1 score is defined as ``1.0``
          because the prediction matches the target exactly.
        - If both graphs have no edges, the edge F1 score is defined as ``1.0``
          for the same reason.
        - If one side is empty and the other is not, the relevant F1 score is
          ``0.0``.
        - Direction matters for edges in the underlying ``nx.DiGraph``.
    """

    def f1_score(predicted: set[Any], expected: set[Any]) -> float:
        if not predicted and not expected:
            return 1.0
        if not predicted or not expected:
            return 0.0
        true_positives = len(predicted & expected)
        if true_positives == 0:
            return 0.0
        precision = true_positives / len(predicted)
        recall = true_positives / len(expected)
        return (2.0 * precision * recall) / (precision + recall)

    agent_nodes = set(agent_graph.nodes())
    truth_nodes = set(ground_truth_graph.nodes())
    agent_edges = set(agent_graph.edges())
    truth_edges = set(ground_truth_graph.edges())

    node_f1 = f1_score(agent_nodes, truth_nodes)
    edge_f1 = f1_score(agent_edges, truth_edges)
    return (node_f1 + edge_f1) / 2.0


def compute_r3(flagged_nodes: list[Any], mutated_nodes: list[Any]) -> float:
    """Compute precision for mutation-flagging decisions.

    The score is defined as ``TP / (TP + FP)``, where:

    - ``TP`` is the number of flagged nodes that were actually mutated.
    - ``FP`` is the number of flagged nodes that were not mutated.

    This is a precision-only reward and does not penalize false negatives. Node
    identifiers are deduplicated before scoring so repeated entries do not
    inflate or deflate the result.

    Args:
        flagged_nodes: Nodes predicted by the agent as mutated or suspicious.
        mutated_nodes: Nodes that were actually mutated in the environment.

    Returns:
        A float in ``[0.0, 1.0]`` representing precision.

    Edge Cases:
        - If ``TP`` and ``FP`` are both zero, the function returns ``0.0``.
          This is intentional and differs from some metric conventions that
          would return ``1.0`` for an empty prediction.
        - Duplicate node identifiers are ignored.
        - Unhashable entries are converted to stable string representations for
          comparison so the function remains usable with loosely structured
          inputs.
    """

    def canonicalize(value: Any) -> Hashable:
        if isinstance(value, dict):
            return tuple(
                (repr(key), canonicalize(subvalue))
                for key, subvalue in sorted(value.items(), key=lambda item: repr(item[0]))
            )
        if isinstance(value, (list, tuple)):
            return tuple(canonicalize(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted((canonicalize(item) for item in value), key=repr))
        if isinstance(value, Hashable):
            return value
        return repr(value)

    flagged_set = {canonicalize(node) for node in flagged_nodes}
    mutated_set = {canonicalize(node) for node in mutated_nodes}

    if not mutated_set:
        return 0.0
    if not flagged_set:
        return 0.0

    true_positives = len(flagged_set & mutated_set)
    if true_positives == 0:
        return 0.0

    precision = true_positives / len(flagged_set) if flagged_set else 0.0
    recall = true_positives / len(mutated_set)
    if precision + recall < 1e-12:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def compute_r4(proposed_edits: list[Any], ground_truth_delta: list[Any]) -> float:
    """Score repair quality: strong positive when repairs match the delta, negative when wrong.

    Uses repair-set F1 with a 0.6 max positive, plus a spurious penalty, clamped
    to ``[-0.5, 0.6]`` so r4 is not a constant negative during RL.
    """

    def canonicalize(value: Any) -> Hashable:
        if isinstance(value, dict):
            return tuple(
                (repr(key), canonicalize(subvalue))
                for key, subvalue in sorted(value.items(), key=lambda item: repr(item[0]))
            )
        if isinstance(value, (list, tuple)):
            return tuple(canonicalize(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted((canonicalize(item) for item in value), key=repr))
        if isinstance(value, Hashable):
            return value
        return repr(value)

    proposed_set = {canonicalize(edit) for edit in proposed_edits}
    truth_set = {canonicalize(edit) for edit in ground_truth_delta}

    if not truth_set and not proposed_set:
        return 0.0
    if not truth_set and proposed_set:
        return -min(0.45, 0.2 + 0.12 * max(0, len(proposed_set) - 1))
    if truth_set and not proposed_set:
        return -0.35

    true_pos = len(proposed_set & truth_set)
    false_pos = len(proposed_set - truth_set)
    if true_pos == 0:
        return -0.45

    precision = true_pos / len(proposed_set) if proposed_set else 0.0
    recall = true_pos / len(truth_set)
    f1 = (
        0.0
        if (precision + recall) < 1e-12
        else (2.0 * precision * recall) / (precision + recall)
    )
    positive = 0.6 * f1
    spurious_penalty = 0.45 * (false_pos / max(len(proposed_set), 1))
    raw = positive - spurious_penalty
    return max(-0.5, min(0.6, raw))


def compute_r5(repair_steps_taken: int, minimum_required_steps: int) -> float:
    """Reward repair efficiency relative to the minimum viable step count.

    The score is defined as:

    ``1 / (1 + max(0, repair_steps_taken - minimum_required_steps))``

    This gives full credit when the agent uses no more than the minimum
    required number of repair steps, and smoothly decays as the agent takes
    additional unnecessary steps.

    Args:
        repair_steps_taken: Total repair actions performed by the agent.
        minimum_required_steps: The minimum number of steps needed to complete
            the repair.

    Returns:
        A float in the interval ``(0.0, 1.0]``.

    Edge Cases:
        - If ``repair_steps_taken`` is less than ``minimum_required_steps``, the
          reward is still ``1.0`` because the formula only penalizes extra
          steps.
        - If either argument is negative, the formula is applied as written
          rather than raising an exception.
        - Very large step overruns asymptotically approach ``0.0`` but never
          become negative.
    """

    return 1.0 / (1.0 + max(0, repair_steps_taken - minimum_required_steps))


def compute_repetition_penalty(repeat_streak: int) -> float:
    """Penalty grows after the first repeat; 3+ consecutive same action_type is heavily penalized."""

    if repeat_streak < 2:
        return 0.0
    if repeat_streak == 2:
        return -0.1
    if repeat_streak == 3:
        return -0.3
    if repeat_streak == 4:
        return -0.5
    return -0.75


def build_format_bonus(valid: bool) -> float:
    """Small bonus for schema-valid actions (stabilizes early RL)."""

    return FORMAT_BONUS_VALID if valid else 0.0


def build_exploration_bonus(
    last_valid_action_type: str | None,
    current_action_type: str | None,
) -> float:
    if last_valid_action_type is None or current_action_type is None:
        return 0.0
    return EXPLORATION_BONUS if current_action_type != last_valid_action_type else 0.0


def action_repeat_key(validation_status: str, action_type: str, valid: bool) -> str:
    """Key used to detect repeated behavior including invalid/parse failures."""

    if not valid:
        return f"invalid:{validation_status}"
    return f"ok:{action_type}"


def assemble_step_reward(
    r1: float,
    r2: float,
    r3: float,
    r4: float,
    r5: float,
    *,
    format_bonus: float,
    repetition_penalty: float,
    invalid_node_penalty: float = 0.0,
    exploration_bonus: float = 0.0,
) -> float:
    """Blend r1..r5 with r3 weighted higher for mutation detection."""

    r1_adj = min(1.0, r1 * R1_IMPORTANCE_SCALE)
    r2_adj = r2 * R2_ENV_DOMINANCE_SCALE
    return (
        0.28 * r1_adj
        + 0.09 * r2_adj
        + 0.25 * r3
        + 0.19 * r4
        + 0.10 * r5
        + float(format_bonus)
        + float(repetition_penalty)
        + float(invalid_node_penalty)
        + float(exploration_bonus)
    )


def compute_total_reward(r1: float, r2: float, r3: float, r4: float, r5: float) -> float:
    """Legacy weighted sum without repetition or format terms (kept for callers/tests)."""

    return 0.25 * r1 + 0.25 * r2 + 0.20 * r3 + 0.20 * r4 + 0.10 * r5
