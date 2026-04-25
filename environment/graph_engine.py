"""Graph state management for DriftEnv."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx


class GraphEngine:
    """Manage a directed graph representing the DriftEnv world state.

    The engine is intentionally limited to graph loading, inspection,
    mutation, diffing, and serialization. It does not contain any
    reinforcement-learning logic.
    """

    def __init__(self) -> None:
        """Initialize an empty directed graph engine."""

        self.graph: nx.DiGraph = nx.DiGraph()

    def load_graph(self, filepath: str) -> None:
        """Load graph data from a JSON file into the internal ``DiGraph``.

        The expected JSON structure is:

        ``{"nodes": [{"id": ..., "type": ..., "attributes": {...}}],``
        `` "edges": [{"source": ..., "target": ..., "type": ...}]}``

        Node entries may include additional top-level keys beyond ``id``,
        ``type``, and ``attributes``; those keys are preserved as node
        attributes. Edge entries may also include additional keys beyond
        ``source``, ``target``, and ``type``; those are preserved as edge
        attributes.

        Args:
            filepath: Path to the JSON file to load.

        Returns:
            None. The internal graph is replaced with the loaded graph.

        Raises:
            FileNotFoundError: If ``filepath`` does not exist.
            ValueError: If the file content is not valid JSON graph data.

        Edge Cases:
            - Missing ``nodes`` or ``edges`` sections are treated as empty lists.
            - Duplicate node IDs overwrite the earlier node definition because
              ``networkx.DiGraph`` stores one node per ID.
            - Duplicate directed edges overwrite the earlier edge definition
              because ``networkx.DiGraph`` stores at most one edge per
              ``(source, target)`` pair.
            - Nodes referenced by edges are added automatically by NetworkX if
              they were not declared explicitly in the node list.
        """

        path = Path(filepath)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise ValueError("Graph JSON must contain a top-level object.")

        nodes = payload.get("nodes", [])
        edges = payload.get("edges", [])
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("'nodes' and 'edges' must be lists.")

        graph = nx.DiGraph()

        for node_entry in nodes:
            if not isinstance(node_entry, dict):
                raise ValueError("Each node entry must be an object.")
            node_id = node_entry.get("id")
            if node_id is None:
                raise ValueError("Each node entry must include an 'id'.")

            node_type = node_entry.get("type")
            raw_attributes = node_entry.get("attributes", {})
            if raw_attributes is None:
                raw_attributes = {}
            if not isinstance(raw_attributes, dict):
                raise ValueError("Node 'attributes' must be a dictionary.")

            extra_attributes = {
                key: value
                for key, value in node_entry.items()
                if key not in {"id", "type", "attributes"}
            }
            graph.add_node(
                node_id,
                type=node_type,
                attributes=dict(raw_attributes),
                **extra_attributes,
            )

        for edge_entry in edges:
            if not isinstance(edge_entry, dict):
                raise ValueError("Each edge entry must be an object.")
            source = edge_entry.get("source")
            target = edge_entry.get("target")
            if source is None or target is None:
                raise ValueError("Each edge entry must include 'source' and 'target'.")

            edge_type = edge_entry.get("type")
            extra_attributes = {
                key: value
                for key, value in edge_entry.items()
                if key not in {"source", "target", "type"}
            }
            graph.add_edge(source, target, type=edge_type, **extra_attributes)

        self.graph = graph

    def get_subgraph(self, center_node_ids: list[Any], hops: int = 2) -> nx.DiGraph:
        """Return the bounded N-hop neighborhood around one or more center nodes.

        Neighborhood distance is computed on an undirected view of the graph so
        both predecessors and successors are considered reachable. If more than
        15 nodes fall within the requested radius, the subgraph is truncated to
        the 15 closest nodes by hop distance. Ties are broken deterministically
        by stringified node ID.

        Args:
            center_node_ids: Node IDs from which to expand the neighborhood.
            hops: Maximum hop distance from any center node. Defaults to ``2``.

        Returns:
            A new ``nx.DiGraph`` containing the induced subgraph over the
            selected neighborhood nodes.

        Raises:
            ValueError: If ``hops`` is negative.

        Edge Cases:
            - Nonexistent center node IDs are ignored.
            - If none of the provided center nodes exist in the graph, an empty
              graph is returned.
            - If more than 15 valid center nodes are supplied with ``hops=0``,
              only the 15 lexicographically earliest IDs are kept.
            - The returned graph is a copy, so mutating it does not alter the
              engine's internal graph.
        """

        if hops < 0:
            raise ValueError("'hops' must be non-negative.")

        valid_centers = [node_id for node_id in center_node_ids if node_id in self.graph]
        if not valid_centers:
            return nx.DiGraph()

        undirected = self.graph.to_undirected(as_view=True)
        distances: dict[Any, int] = {}
        queue: deque[tuple[Any, int]] = deque()

        for node_id in valid_centers:
            if node_id not in distances:
                distances[node_id] = 0
                queue.append((node_id, 0))

        while queue:
            current_node, current_distance = queue.popleft()
            if current_distance >= hops:
                continue
            for neighbor in undirected.neighbors(current_node):
                next_distance = current_distance + 1
                if neighbor not in distances or next_distance < distances[neighbor]:
                    distances[neighbor] = next_distance
                    queue.append((neighbor, next_distance))

        selected_nodes = [
            node_id
            for node_id, distance in sorted(
                distances.items(),
                key=lambda item: (item[1], str(item[0])),
            )
            if distance <= hops
        ][:15]

        return self.graph.subgraph(selected_nodes).copy()

    def apply_mutation(self, mutation_type: str, params: dict[str, Any]) -> dict[str, Any]:
        """Apply a supported mutation to the internal graph and describe it.

        Supported mutation types are:

        - ``"node_removal"``: Removes a node and all incident edges.
          Required params: ``{"node_id": ...}``
        - ``"edge_rewiring"``: Redirects an existing edge from
          ``(source, old_target)`` to ``(source, new_target)`` while preserving
          the original edge attributes unless overrides are supplied.
          Required params: ``{"source": ..., "old_target": ..., "new_target": ...}``

        Args:
            mutation_type: Name of the mutation to apply.
            params: Mutation parameters for the selected mutation type.

        Returns:
            A dictionary describing the mutation and the affected graph
            elements. The return format is designed to be easy to compare or
            transform into reward inputs.

        Raises:
            ValueError: If ``mutation_type`` is unsupported or required params
                are missing.
            KeyError: If the referenced node or edge does not exist.

        Edge Cases:
            - Removing a node also removes all incoming and outgoing edges; the
              returned change record includes those removed edges.
            - Rewiring an edge preserves its existing edge data unless
              ``params`` includes ``edge_type`` or ``edge_attributes`` to
              override them.
            - If the rewired target edge already exists, its data is replaced by
              the new edge attributes because ``nx.DiGraph`` stores one edge per
              directed pair.
        """

        if mutation_type == "node_removal":
            node_id = params.get("node_id")
            if node_id is None:
                raise ValueError("'node_removal' requires a 'node_id' parameter.")
            if node_id not in self.graph:
                raise KeyError(f"Node {node_id!r} does not exist.")

            node_data = dict(self.graph.nodes[node_id])
            removed_edges = [
                {
                    "source": source,
                    "target": target,
                    "type": data.get("type"),
                    "attributes": {
                        key: value for key, value in data.items() if key != "type"
                    },
                }
                for source, target, data in self.graph.in_edges(node_id, data=True)
            ] + [
                {
                    "source": source,
                    "target": target,
                    "type": data.get("type"),
                    "attributes": {
                        key: value for key, value in data.items() if key != "type"
                    },
                }
                for source, target, data in self.graph.out_edges(node_id, data=True)
            ]

            self.graph.remove_node(node_id)
            return {
                "mutation_type": "node_removal",
                "removed_node": {
                    "id": node_id,
                    "type": node_data.get("type"),
                    "attributes": dict(node_data.get("attributes", {})),
                },
                "removed_edges": sorted(
                    removed_edges,
                    key=lambda edge: (str(edge["source"]), str(edge["target"]), str(edge["type"])),
                ),
            }

        if mutation_type == "edge_rewiring":
            source = params.get("source")
            old_target = params.get("old_target")
            new_target = params.get("new_target")
            if source is None or old_target is None or new_target is None:
                raise ValueError(
                    "'edge_rewiring' requires 'source', 'old_target', and 'new_target'."
                )
            if source not in self.graph:
                raise KeyError(f"Source node {source!r} does not exist.")
            if old_target not in self.graph:
                raise KeyError(f"Old target node {old_target!r} does not exist.")
            if new_target not in self.graph:
                raise KeyError(f"New target node {new_target!r} does not exist.")
            if not self.graph.has_edge(source, old_target):
                raise KeyError(f"Edge ({source!r}, {old_target!r}) does not exist.")

            original_data = dict(self.graph.edges[source, old_target])
            new_data = dict(original_data)
            if "edge_type" in params:
                new_data["type"] = params["edge_type"]
            if "edge_attributes" in params:
                edge_attributes = params["edge_attributes"]
                if not isinstance(edge_attributes, dict):
                    raise ValueError("'edge_attributes' must be a dictionary when provided.")
                new_data.update(edge_attributes)

            replaced_existing_edge = self.graph.has_edge(source, new_target)
            previous_new_edge_data = (
                dict(self.graph.edges[source, new_target]) if replaced_existing_edge else None
            )

            self.graph.remove_edge(source, old_target)
            self.graph.add_edge(source, new_target, **new_data)

            return {
                "mutation_type": "edge_rewiring",
                "removed_edge": {
                    "source": source,
                    "target": old_target,
                    "type": original_data.get("type"),
                    "attributes": {
                        key: value for key, value in original_data.items() if key != "type"
                    },
                },
                "added_edge": {
                    "source": source,
                    "target": new_target,
                    "type": new_data.get("type"),
                    "attributes": {
                        key: value for key, value in new_data.items() if key != "type"
                    },
                },
                "replaced_existing_edge": replaced_existing_edge,
                "previous_new_edge": (
                    None
                    if previous_new_edge_data is None
                    else {
                        "source": source,
                        "target": new_target,
                        "type": previous_new_edge_data.get("type"),
                        "attributes": {
                            key: value
                            for key, value in previous_new_edge_data.items()
                            if key != "type"
                        },
                    }
                ),
            }

        raise ValueError(f"Unsupported mutation type: {mutation_type!r}")

    def get_ground_truth_delta(
        self, before_state: dict[str, Any], after_state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Compute a deterministic list of graph changes between two states.

        The expected state format matches the JSON graph structure used by
        :meth:`load_graph`: a top-level dictionary with ``"nodes"`` and
        ``"edges"`` lists. The returned change records are plain dictionaries
        so they can be fed directly into ``compute_r4`` or compared against
        agent-proposed edits.

        Change operations emitted by this method are:

        - ``remove_node``
        - ``add_node``
        - ``update_node``
        - ``remove_edge``
        - ``add_edge``
        - ``update_edge``

        Args:
            before_state: Graph state before a mutation or repair.
            after_state: Graph state after a mutation or repair.

        Returns:
            A sorted list of change dictionaries describing how to transform
            ``before_state`` into ``after_state``.

        Raises:
            ValueError: If either state is not a valid graph-state dictionary.

        Edge Cases:
            - If there are no differences between the states, an empty list is
              returned.
            - Node comparison is keyed by node ID.
            - Edge comparison is keyed by ``(source, target)`` because the
              internal graph uses ``nx.DiGraph`` rather than a multigraph.
            - Changes are returned in deterministic sorted order to keep reward
              comparisons stable across runs.
        """

        before_nodes, before_edges = self._normalize_state(before_state)
        after_nodes, after_edges = self._normalize_state(after_state)
        changes: list[dict[str, Any]] = []

        for node_id in sorted(before_nodes.keys() - after_nodes.keys(), key=str):
            node_record = before_nodes[node_id]
            changes.append(
                {
                    "op": "remove_node",
                    "id": node_id,
                    "type": node_record.get("type"),
                    "attributes": dict(node_record.get("attributes", {})),
                }
            )

        for node_id in sorted(after_nodes.keys() - before_nodes.keys(), key=str):
            node_record = after_nodes[node_id]
            changes.append(
                {
                    "op": "add_node",
                    "id": node_id,
                    "type": node_record.get("type"),
                    "attributes": dict(node_record.get("attributes", {})),
                }
            )

        for node_id in sorted(before_nodes.keys() & after_nodes.keys(), key=str):
            if before_nodes[node_id] != after_nodes[node_id]:
                changes.append(
                    {
                        "op": "update_node",
                        "id": node_id,
                        "before": before_nodes[node_id],
                        "after": after_nodes[node_id],
                    }
                )

        for edge_key in sorted(before_edges.keys() - after_edges.keys(), key=self._edge_sort_key):
            source, target = edge_key
            edge_record = before_edges[edge_key]
            changes.append(
                {
                    "op": "remove_edge",
                    "source": source,
                    "target": target,
                    "type": edge_record.get("type"),
                    "attributes": dict(edge_record.get("attributes", {})),
                }
            )

        for edge_key in sorted(after_edges.keys() - before_edges.keys(), key=self._edge_sort_key):
            source, target = edge_key
            edge_record = after_edges[edge_key]
            changes.append(
                {
                    "op": "add_edge",
                    "source": source,
                    "target": target,
                    "type": edge_record.get("type"),
                    "attributes": dict(edge_record.get("attributes", {})),
                }
            )

        for edge_key in sorted(
            before_edges.keys() & after_edges.keys(),
            key=self._edge_sort_key,
        ):
            if before_edges[edge_key] != after_edges[edge_key]:
                source, target = edge_key
                changes.append(
                    {
                        "op": "update_edge",
                        "source": source,
                        "target": target,
                        "before": before_edges[edge_key],
                        "after": after_edges[edge_key],
                    }
                )

        return changes

    def serialize_subgraph(self, subgraph: nx.DiGraph) -> str:
        """Serialize a subgraph into compact structured text.

        The output format is:

        ``NODE_LIST``
        ``- id=<...> | type=<...> | attr1=value1; attr2=value2``
        ``EDGE_LIST``
        ``- <source> -> <target> | type=<...>``

        Each node includes its ID, type, and a small set of key attributes.
        The method attempts to stay within roughly 500 tokens by limiting the
        overall character budget and appending a truncation note if needed.

        Args:
            subgraph: The directed graph to serialize.

        Returns:
            A structured text summary suitable for prompts, debugging, or
            downstream inspection.

        Edge Cases:
            - Empty subgraphs still include both section headers.
            - Missing node types or edge types are rendered as ``unknown``.
            - If the content would exceed the approximate size budget, trailing
              node or edge lines are omitted and a truncation marker is added.
        """

        max_chars = 3200
        node_lines = ["NODE_LIST"]
        edge_lines = ["EDGE_LIST"]

        for node_id, data in sorted(subgraph.nodes(data=True), key=lambda item: str(item[0])):
            node_type = data.get("type", "unknown")
            attributes = data.get("attributes", {})
            if not isinstance(attributes, dict):
                attributes = {}
            key_attributes = self._select_key_attributes(attributes)
            if key_attributes:
                attribute_text = "; ".join(
                    f"{key}={value}" for key, value in key_attributes.items()
                )
            else:
                attribute_text = "no_key_attributes"
            node_lines.append(
                f"- id={node_id} | type={node_type if node_type is not None else 'unknown'}"
                f" | {attribute_text}"
            )

        for source, target, data in sorted(
            subgraph.edges(data=True),
            key=lambda item: (str(item[0]), str(item[1]), str(item[2].get("type"))),
        ):
            edge_type = data.get("type", "unknown")
            edge_lines.append(f"- {source} -> {target} | type={edge_type}")

        lines: list[str] = []
        truncated = False
        for line in node_lines + edge_lines:
            candidate = "\n".join(lines + [line])
            if len(candidate) > max_chars:
                truncated = True
                break
            lines.append(line)

        if truncated:
            truncation_note = "... TRUNCATED_FOR_LENGTH"
            candidate = "\n".join(lines + [truncation_note])
            if len(candidate) <= max_chars:
                lines.append(truncation_note)

        return "\n".join(lines)

    @staticmethod
    def _normalize_state(
        state: dict[str, Any]
    ) -> tuple[dict[Any, dict[str, Any]], dict[tuple[Any, Any], dict[str, Any]]]:
        """Normalize a graph-state dictionary into comparable node and edge maps."""

        if not isinstance(state, dict):
            raise ValueError("Graph state must be a dictionary.")

        raw_nodes = state.get("nodes", [])
        raw_edges = state.get("edges", [])
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise ValueError("Graph state 'nodes' and 'edges' must be lists.")

        nodes: dict[Any, dict[str, Any]] = {}
        edges: dict[tuple[Any, Any], dict[str, Any]] = {}

        for node_entry in raw_nodes:
            if not isinstance(node_entry, dict) or "id" not in node_entry:
                raise ValueError("Each node entry must be a dictionary with an 'id'.")
            node_id = node_entry["id"]
            raw_attributes = node_entry.get("attributes", {})
            if raw_attributes is None:
                raw_attributes = {}
            if not isinstance(raw_attributes, dict):
                raise ValueError("Node 'attributes' must be a dictionary.")
            nodes[node_id] = {
                "type": node_entry.get("type"),
                "attributes": dict(raw_attributes),
            }

        for edge_entry in raw_edges:
            if not isinstance(edge_entry, dict):
                raise ValueError("Each edge entry must be a dictionary.")
            if "source" not in edge_entry or "target" not in edge_entry:
                raise ValueError("Each edge entry must contain 'source' and 'target'.")
            extra_attributes = {
                key: value
                for key, value in edge_entry.items()
                if key not in {"source", "target", "type"}
            }
            edges[(edge_entry["source"], edge_entry["target"])] = {
                "type": edge_entry.get("type"),
                "attributes": extra_attributes,
            }

        return nodes, edges

    @staticmethod
    def _edge_sort_key(edge_key: tuple[Any, Any]) -> tuple[str, str]:
        """Return a deterministic sort key for directed edge identifiers."""

        source, target = edge_key
        return (str(source), str(target))

    @staticmethod
    def _select_key_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
        """Choose a small, stable subset of node attributes for serialization."""

        preferred_order = (
            "name",
            "role",
            "status",
            "owner_team",
            "org_level",
            "timezone",
            "communication_pref",
            "manager_id",
            "version",
            "api_schema_version",
            "tool_id",
            "oo_status",
        )
        selected: dict[str, Any] = {}

        for key in preferred_order:
            if key in attributes:
                selected[key] = attributes[key]
            if len(selected) >= 4:
                return selected

        for key in sorted(attributes.keys(), key=str):
            if key not in selected:
                selected[key] = attributes[key]
            if len(selected) >= 4:
                break

        return selected
