import networkx as nx
import pytest

from environment.reward import (
    compute_r1,
    compute_r2,
    compute_r3,
    compute_r4,
    compute_r5,
)


# This scenario represents a fully correct action with matching stakeholders,
# complete required fields, and a valid route.
def test_compute_r1_best_case() -> None:
    action = {
        "stakeholders": ["ops", "security"],
        "fields": {"ticket_id": "INC-1", "severity": "high"},
        "route": ["ingest", "triage", "repair"],
    }
    ground_truth = {
        "stakeholders": ["ops", "security"],
        "required_fields": ["ticket_id", "severity"],
        "valid_nodes": ["ingest", "triage", "repair"],
        "valid_edges": [("ingest", "triage"), ("triage", "repair")],
    }

    assert compute_r1(action, ground_truth) == pytest.approx(1.0)


# This scenario represents an action that misses all required stakeholders,
# fields, and routing requirements, producing the minimum score.
def test_compute_r1_worst_case() -> None:
    action = {}
    ground_truth = {
        "stakeholders": ["ops"],
        "required_fields": ["ticket_id"],
        "valid_nodes": ["ingest"],
        "valid_edges": [("ingest", "repair")],
    }

    assert compute_r1(action, ground_truth) == pytest.approx(0.0)


# This scenario represents a partially correct action with one missing field,
# incomplete stakeholder coverage, and an invalid hop in the route.
def test_compute_r1_partial_case() -> None:
    action = {
        "stakeholders": ["alice"],
        "fields": {"ticket_id": "INC-9"},
        "route": ["n1", "bad", "n3"],
    }
    ground_truth = {
        "stakeholders": ["alice", "bob"],
        "required_fields": ["ticket_id", "severity"],
        "valid_nodes": ["n1", "n2", "n3"],
        "valid_edges": [("n1", "n2"), ("n2", "n3")],
    }

    assert compute_r1(action, ground_truth) == pytest.approx(0.5)


# This scenario represents two identical directed graphs, which should yield
# perfect node and edge F1 scores.
def test_compute_r2_best_case() -> None:
    agent_graph = nx.DiGraph([("a", "b"), ("b", "c")])
    ground_truth_graph = nx.DiGraph([("a", "b"), ("b", "c")])

    assert compute_r2(agent_graph, ground_truth_graph) == pytest.approx(1.0)


# This scenario represents graphs with no overlapping nodes or edges, which
# should produce zero node F1 and zero edge F1.
def test_compute_r2_worst_case() -> None:
    agent_graph = nx.DiGraph([("a", "b")])
    ground_truth_graph = nx.DiGraph([("x", "y")])

    assert compute_r2(agent_graph, ground_truth_graph) == pytest.approx(0.0)


# This scenario represents a partial graph match where one node overlaps but
# the directed edges do not match.
def test_compute_r2_partial_case() -> None:
    agent_graph = nx.DiGraph([("a", "b")])
    ground_truth_graph = nx.DiGraph([("a", "c")])

    assert compute_r2(agent_graph, ground_truth_graph) == pytest.approx(0.25)


# This scenario represents perfectly precise mutation flagging where every
# flagged node is truly mutated.
def test_compute_r3_best_case() -> None:
    flagged_nodes = ["n1", "n2"]
    mutated_nodes = ["n1", "n2"]

    assert compute_r3(flagged_nodes, mutated_nodes) == pytest.approx(1.0)


# This scenario represents all flagged nodes being false positives, so the
# precision-based reward should be zero.
def test_compute_r3_worst_case() -> None:
    flagged_nodes = ["n1", "n2"]
    mutated_nodes = ["n3", "n4"]

    assert compute_r3(flagged_nodes, mutated_nodes) == pytest.approx(0.0)


# This scenario represents the critical empty-input edge case where both TP and
# FP are zero and the function must still return 0.0.
def test_compute_r3_empty_edge_case() -> None:
    assert compute_r3([], []) == pytest.approx(0.0)


# This scenario represents proposing exactly the required repairs with no extra
# edits, which should receive the maximum repair score.
def test_compute_r4_best_case() -> None:
    proposed_edits = ["fix-node-a", "repair-edge-b"]
    ground_truth_delta = ["fix-node-a", "repair-edge-b"]

    assert compute_r4(proposed_edits, ground_truth_delta) == pytest.approx(1.0)


# This scenario represents only incorrect repairs when no repair was actually
# needed, which should clamp to the minimum score.
def test_compute_r4_worst_case() -> None:
    proposed_edits = ["bad-fix-1", "bad-fix-2"]
    ground_truth_delta: list[str] = []

    assert compute_r4(proposed_edits, ground_truth_delta) == pytest.approx(-0.5)


# This scenario represents one correct repair and one incorrect extra repair,
# leading to a partially positive normalized score.
def test_compute_r4_partial_case() -> None:
    proposed_edits = ["repair-a", "extra-fix"]
    ground_truth_delta = ["repair-a", "repair-b"]

    assert compute_r4(proposed_edits, ground_truth_delta) == pytest.approx(0.25)


# This scenario represents taking exactly the minimum required number of steps,
# which should receive the maximum efficiency reward.
def test_compute_r5_best_case() -> None:
    assert compute_r5(3, 3) == pytest.approx(1.0)


# This scenario represents a large number of unnecessary extra steps, producing
# a very small efficiency reward.
def test_compute_r5_worst_case() -> None:
    assert compute_r5(100, 0) == pytest.approx(1.0 / 101.0)


# This scenario represents a modest overrun above the minimum required steps,
# which should reduce the reward without driving it to zero.
def test_compute_r5_partial_case() -> None:
    assert compute_r5(5, 3) == pytest.approx(1.0 / 3.0)
