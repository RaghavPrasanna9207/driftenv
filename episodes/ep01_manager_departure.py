"""Episode definition for DriftEnv Episode 01: Manager Departure."""

from __future__ import annotations

import random
from typing import Any


EPISODE_ID = "EP-01"

MARCUS_NODE_ID = "P1"
MARCUS_NAME = "Marcus"


def get_initial_tasks() -> list[dict[str, Any]]:
    """Return the initial task list for the manager-departure scenario."""

    return [
        {
            "task_id": "TASK-EP01-1",
            "action_type": "draft_reply",
            "route_to": MARCUS_NODE_ID,
            "recipient_name": MARCUS_NAME,
            "prompt": "Route a status update to Marcus about the current sprint blockers.",
        },
        {
            "task_id": "TASK-EP01-2",
            "action_type": "delegate_task",
            "route_to": MARCUS_NODE_ID,
            "assignee_name": MARCUS_NAME,
            "prompt": "Route a follow-up approval task to Marcus for review.",
        },
    ]


def get_mutation_schedule(seed: int) -> list[tuple[int, str, dict[str, Any]]]:
    """Return the hidden mutation schedule for Episode 01."""

    rng = random.Random(seed)
    scheduled_turn = rng.randint(5, 8)
    return [(scheduled_turn, "node_removal", {"node_id": MARCUS_NODE_ID})]


def get_episode_description() -> str:
    """Return a short description of what the agent experiences in Episode 01."""

    return (
        "The agent begins with two ordinary tasks that both require routing work to "
        "Marcus, the engineering manager. Between turns 5 and 8, the live graph "
        "silently mutates and removes Marcus from the org graph. After that removal, "
        "any action that still routes to Marcus should fail with ENTITY_NOT_FOUND."
    )
