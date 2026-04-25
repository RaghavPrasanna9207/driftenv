import json
import random

from environment.env import DriftEnv


EXPECTED_REWARD_KEYS = {"r1", "r2", "r3", "r4", "r5"}


def _random_valid_action_text(rng: random.Random, env: DriftEnv) -> str:
    """Return a random schema-valid action encoded as JSON text."""

    action_space = [
        {
            "action_type": "draft_reply",
            "params": {
                "recipient_id": "P2",
                "message": "Acknowledged. I will follow up shortly.",
            },
        },
        {
            "action_type": "reschedule_meeting",
            "params": {
                "meeting_id": "M1",
                "new_time": "2026-04-26T10:00:00Z",
            },
        },
        {
            "action_type": "delegate_task",
            "params": {
                "task_id": "TASK-1",
                "assignee_id": "P3",
            },
        },
        {
            "action_type": "escalate",
            "params": {
                "issue_id": "ISSUE-1",
                "target_id": "P10",
            },
        },
        {
            "action_type": "decline_meeting",
            "params": {
                "meeting_id": "M2",
                "reason": "Scheduling conflict",
            },
        },
        {
            "action_type": "request_clarification",
            "params": {
                "question": "Can you confirm whether the org graph changed recently?",
            },
        },
        {
            "action_type": "flag_inconsistency",
            "params": {
                "node_id": "P2",
            },
        },
        {
            "action_type": "propose_attribute_update",
            "params": {
                "node_id": "P3",
                "attribute_name": "oo_status",
                "new_value": "available",
            },
        },
        {
            "action_type": "propose_edge_update",
            "params": {
                "source": "P1",
                "old_target": "P2",
                "new_target": "P3",
            },
        },
    ]

    if env.mutated_nodes:
        mutated_node = env.mutated_nodes[-1]
        if env.turn >= env.MAX_TURNS:
            return json.dumps(
                {
                    "action_type": "propose_node_removal",
                    "params": {
                        "node_id": mutated_node,
                    },
                }
            )
        if mutated_node not in env.flagged_nodes:
            return json.dumps(
                rng.choice(
                    [
                        {
                            "action_type": "flag_inconsistency",
                            "params": {
                                "node_id": mutated_node,
                            },
                        },
                        {
                            "action_type": "flag_inconsistency",
                            "params": {
                                "node_ids": [mutated_node],
                            },
                        },
                    ]
                )
            )

    return json.dumps(rng.choice(action_space))


# This scenario runs three full random-agent episodes and verifies that the
# environment stays stable for 12 turns while producing structured rewards.
def test_random_agent_runs_three_full_driftenv_episodes() -> None:
    episode_rewards: list[float] = []
    saw_flagged_nodes = False
    saw_proposed_edits = False
    saw_nonzero_r3 = False
    saw_positive_r4 = False

    for episode_index in range(3):
        env = DriftEnv(episode_type="manager_departure", curriculum_stage=1, seed=episode_index)
        env.reset()
        rng = random.Random(episode_index)

        total_reward = 0.0
        turn_count = 0

        while not env.done:
            action_text = _random_valid_action_text(rng, env)
            step_result = env.step(action_text)
            reward_breakdown = step_result["info"]["reward_breakdown"]
            reward_inputs = step_result["info"]["reward_inputs"]

            print(
                f"episode={episode_index + 1} "
                f"turn={turn_count + 1} "
                f"reward_breakdown={reward_breakdown} "
                f"reward_inputs={reward_inputs}"
            )

            assert set(reward_breakdown.keys()) == EXPECTED_REWARD_KEYS

            saw_flagged_nodes = saw_flagged_nodes or bool(reward_inputs["flagged_nodes"])
            saw_proposed_edits = saw_proposed_edits or reward_inputs["proposed_edits_count"] > 0
            saw_nonzero_r3 = saw_nonzero_r3 or reward_breakdown["r3"] > 0.0
            saw_positive_r4 = saw_positive_r4 or reward_breakdown["r4"] > 0.0

            total_reward += step_result["reward"]
            turn_count += 1

        episode_rewards.append(total_reward)

        assert turn_count == 12

    non_zero_episode_rewards = sum(1 for reward in episode_rewards if reward != 0.0)
    assert non_zero_episode_rewards >= 2
    assert saw_flagged_nodes
    assert saw_proposed_edits
    assert saw_nonzero_r3
    assert saw_positive_r4


def test_issue_id_is_not_validated_as_a_graph_node() -> None:
    env = DriftEnv(episode_type="manager_departure", curriculum_stage=1, seed=0)
    env.reset()

    step_result = env.step(
        json.dumps(
            {
                "action_type": "escalate",
                "params": {
                    "issue_id": "ISSUE-1",
                    "target_id": "P1",
                },
            }
        )
    )

    action_result = step_result["info"]["action_result"]

    assert action_result["success"] is True
    assert not any(
        "Action references unknown or missing graph nodes" in signal
        for signal in action_result["inconsistency_signals"]
    )


def test_task_id_is_not_validated_as_a_graph_node() -> None:
    env = DriftEnv(episode_type="manager_departure", curriculum_stage=1, seed=0)
    env.reset()

    step_result = env.step(
        json.dumps(
            {
                "action_type": "delegate_task",
                "params": {
                    "task_id": "TASK-1",
                    "assignee_id": "P3",
                },
            }
        )
    )

    action_result = step_result["info"]["action_result"]

    assert action_result["success"] is True
    assert not any(
        "Action references unknown or missing graph nodes" in signal
        for signal in action_result["inconsistency_signals"]
    )


def test_invalid_graph_node_ids_are_still_rejected() -> None:
    env = DriftEnv(episode_type="manager_departure", curriculum_stage=1, seed=0)
    env.reset()

    step_result = env.step(
        json.dumps(
            {
                "action_type": "escalate",
                "params": {
                    "issue_id": "ISSUE-1",
                    "target_id": "P999",
                },
            }
        )
    )

    action_result = step_result["info"]["action_result"]

    assert action_result["success"] is False
    assert any(
        "Action references unknown or missing graph nodes: P999." in signal
        for signal in action_result["inconsistency_signals"]
    )
