"""Reward functions for the DriftEnv reinforcement-learning environment."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import networkx as nx


def compute_r1(action: dict[str, Any], ground_truth: dict[str, Any]) -> float:
    """Score action correctness against a ground-truth action specification.

    This reward is intended for high-level action validation in DriftEnv. The
    function evaluates three independent aspects of an agent action and returns
    the arithmetic mean of their scores, clamped to the inclusive range
    ``[0.0, 1.0]``:

    1. Stakeholder correctness:
       The function looks for stakeholder identifiers in the action under the
       keys ``"stakeholders"``, ``"referenced_stakeholders"``,
       ``"stakeholder_ids"``, or ``"targets"``. The same keys are checked in the
       ground truth. A set-based F1 score is used so that missing stakeholders
       and extra stakeholders are both penalized.
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
       or ``"edges"``. When valid nodes are provided, the node portion of the
       score is the fraction of visited route nodes that are allowed. When valid
       directed edges are provided, the edge portion of the score is the
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
          section receives full credit because there is nothing to validate.
        - If the ground truth requires stakeholders, fields, or routing but the
          action omits that section, the corresponding sub-score is ``0.0``.
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

    action_stakeholders = to_set(
        pick(
            action,
            ("stakeholders", "referenced_stakeholders", "stakeholder_ids", "targets"),
        )
    )
    truth_stakeholders = to_set(
        pick(
            ground_truth,
            ("stakeholders", "referenced_stakeholders", "stakeholder_ids", "targets"),
        )
    )
    stakeholder_overlap = len(action_stakeholders & truth_stakeholders)
    if not action_stakeholders and not truth_stakeholders:
        stakeholder_score = 1.0
    elif not action_stakeholders or not truth_stakeholders:
        stakeholder_score = 0.0
    else:
        stakeholder_precision = stakeholder_overlap / len(action_stakeholders)
        stakeholder_recall = stakeholder_overlap / len(truth_stakeholders)
        stakeholder_denominator = stakeholder_precision + stakeholder_recall
        stakeholder_score = (
            0.0
            if stakeholder_denominator == 0.0
            else (2.0 * stakeholder_precision * stakeholder_recall)
            / stakeholder_denominator
        )

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
    if not required_fields:
        field_score = 1.0
    else:
        completed_fields = 0
        for field_name in required_fields:
            if field_name in action and not is_missing(action[field_name]):
                completed_fields += 1
            elif field_name in action_fields and not is_missing(action_fields[field_name]):
                completed_fields += 1
        field_score = completed_fields / len(required_fields)

    route_nodes = to_list(
        pick(action, ("route", "routing_path", "path", "nodes", "route_nodes"))
    )
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
    else:
        routing_score = 1.0

    total_score = (stakeholder_score + field_score + routing_score) / 3.0
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

    true_positives = len(flagged_set & mutated_set)
    false_positives = len(flagged_set - mutated_set)
    if true_positives == 0 and false_positives == 0:
        return 0.0
    return true_positives / (true_positives + false_positives)


def compute_r4(proposed_edits: list[Any], ground_truth_delta: list[Any]) -> float:
    """Score repair quality against the expected change set.

    Proposed repairs are compared against the required repair delta using
    canonicalized, deduplicated edit entries. Each unique correct edit receives
    ``+1.0`` and each unique incorrect edit receives ``-0.5``. The raw score is
    then normalized by the number of unique ground-truth repairs so the result
    behaves like a proportion of required work completed. Finally, the value is
    clamped to ``[-0.5, 1.0]``.

    Args:
        proposed_edits: Repairs suggested or applied by the agent. Entries may
            be strings, tuples, dictionaries, or other Python values.
        ground_truth_delta: The set of repairs that were actually needed.

    Returns:
        A float in ``[-0.5, 1.0]``.
        - ``1.0`` means all required repairs were proposed with no extra edits.
        - Values between ``0.0`` and ``1.0`` indicate partial correctness.
        - Negative values indicate the agent proposed more harmful or spurious
          edits than useful ones.

    Edge Cases:
        - Duplicate edit entries are ignored so repeated items cannot game the
          reward.
        - If no repair is required and no edit is proposed, the function returns
          ``1.0`` because the empty repair plan is correct.
        - If no repair is required but edits are still proposed, the agent is
          penalized and the result is clamped at ``-0.5`` on the low end.
        - Unhashable edit structures are converted into stable canonical tuples
          before comparison.
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
        return 1.0

    correct_repairs = len(proposed_set & truth_set)
    incorrect_repairs = len(proposed_set - truth_set)
    denominator = len(truth_set) if truth_set else 1
    raw_score = (correct_repairs - 0.5 * incorrect_repairs) / denominator
    return max(-0.5, min(1.0, raw_score))


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


def compute_total_reward(r1: float, r2: float, r3: float, r4: float, r5: float) -> float:
    """Compute the weighted DriftEnv reward from component rewards.

    Args:
        r1: Action-validity reward.
        r2: Graph-state similarity reward.
        r3: Mutation-flagging precision reward.
        r4: Repair-quality reward.
        r5: Repair-efficiency reward.

    Returns:
        The weighted sum ``0.25 * r1 + 0.25 * r2 + 0.20 * r3 + 0.20 * r4 +
        0.10 * r5``.

    Edge Cases:
        - The function does not clamp the final value. If callers pass values
          outside their expected ranges, the weighted sum will reflect that.
        - The function performs no type coercion beyond normal Python numeric
          semantics.
    """

    return 0.25 * r1 + 0.25 * r2 + 0.20 * r3 + 0.20 * r4 + 0.10 * r5
