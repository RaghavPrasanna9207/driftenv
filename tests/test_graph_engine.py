from pathlib import Path

from environment.graph_engine import GraphEngine


BASE_GRAPH_PATH = Path(__file__).resolve().parents[1] / "graph_templates" / "base_graph.json"


# This scenario verifies that loading the real base graph populates the expected
# node count and preserves known node metadata.
def test_load_graph_loads_base_graph_with_30_nodes() -> None:
    engine = GraphEngine()

    engine.load_graph(str(BASE_GRAPH_PATH))

    assert engine.graph.number_of_nodes() == 30
    assert engine.graph.nodes["P1"]["type"] == "Person"


# This scenario verifies that removing a node deletes the node itself along with
# all incoming and outgoing edges connected to it.
def test_apply_mutation_node_removal_removes_node_and_incident_edges() -> None:
    engine = GraphEngine()
    engine.load_graph(str(BASE_GRAPH_PATH))

    prior_edges = list(engine.graph.in_edges("P1")) + list(engine.graph.out_edges("P1"))
    change = engine.apply_mutation("node_removal", {"node_id": "P1"})

    assert "P1" not in engine.graph
    assert all("P1" not in edge for edge in engine.graph.edges())
    assert len(change["removed_edges"]) == len(prior_edges)


# This scenario verifies that rewiring an existing edge removes the old target
# and creates the new directed edge from the same source.
def test_apply_mutation_edge_rewiring_changes_edge_target() -> None:
    engine = GraphEngine()
    engine.load_graph(str(BASE_GRAPH_PATH))

    change = engine.apply_mutation(
        "edge_rewiring",
        {"source": "P1", "old_target": "P2", "new_target": "P7"},
    )

    assert not engine.graph.has_edge("P1", "P2")
    assert engine.graph.has_edge("P1", "P7")
    assert change["added_edge"]["target"] == "P7"


# This scenario verifies that subgraph extraction enforces the 15-node hard cap
# even after making the neighborhood around a real graph node artificially dense.
def test_get_subgraph_caps_dense_neighborhood_at_15_nodes() -> None:
    engine = GraphEngine()
    engine.load_graph(str(BASE_GRAPH_PATH))

    for index in range(20):
        node_id = f"X{index}"
        engine.graph.add_node(node_id, type="Synthetic", attributes={"name": node_id})
        engine.graph.add_edge("P1", node_id, type="linked_to")
        engine.graph.add_edge(node_id, "P1", type="linked_to")

    subgraph = engine.get_subgraph(["P1"], hops=1)

    assert subgraph.number_of_nodes() <= 15
    assert "P1" in subgraph
