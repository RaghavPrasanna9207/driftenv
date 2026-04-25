import json

import pytest
from fastapi.testclient import TestClient

from environment.action_parser import build_action_schema_text, parse_action
from server import app


SUPPORTED_ACTION_TYPES = {
    "draft_reply",
    "reschedule_meeting",
    "delegate_task",
    "escalate",
    "decline_meeting",
    "flag_inconsistency",
    "propose_node_removal",
    "propose_edge_update",
    "propose_attribute_update",
    "request_clarification",
}


def _reset_episode(client: TestClient) -> None:
    response = client.post(
        "/reset",
        json={
            "episode_type": "manager_departure",
            "curriculum_stage": 1,
            "seed": 0,
        },
    )
    assert response.status_code == 200


def test_action_schema_lists_supported_actions() -> None:
    schema_text = build_action_schema_text()

    for action_type in SUPPORTED_ACTION_TYPES:
        assert action_type in schema_text
    assert "ADD_NODE" in schema_text
    assert "REPAIR_GRAPH" in schema_text
    assert "LOG_INCONSISTENCY" in schema_text


def test_parse_action_reports_unsupported_action() -> None:
    parsed = parse_action(
        json.dumps(
            {
                "action_type": "ADD_NODE",
                "params": {"node_id": "P2"},
            }
        )
    )

    assert parsed["valid"] is False
    assert parsed["validation_status"] == "unsupported_action"
    assert parsed["action_type"] == "ADD_NODE"


def test_parse_action_reports_wrong_parameters() -> None:
    parsed = parse_action(
        json.dumps(
            {
                "action_type": "draft_reply",
                "params": {"recipient_id": "P2"},
            }
        )
    )

    assert parsed["valid"] is False
    assert parsed["validation_status"] == "wrong_parameters"
    assert "message" in parsed["error_message"]


def test_step_accepts_raw_invalid_json_and_returns_penalty() -> None:
    client = TestClient(app)
    _reset_episode(client)

    response = client.post("/step", json={"action": "hallucinated text"})

    assert response.status_code == 200
    payload = response.json()
    breakdown = payload["info"]["reward_breakdown"]

    assert payload["reward"] == pytest.approx(-1.0)
    assert payload["done"] is False
    assert breakdown["validation_penalty"] == pytest.approx(-1.0)
    assert breakdown["reason"] == "invalid_json"
    assert {"r1", "r2", "r3", "r4", "r5"}.issubset(breakdown.keys())
    assert breakdown["r1"] == pytest.approx(0.0)
    assert breakdown["r2"] == pytest.approx(0.0)
    assert breakdown["r3"] == pytest.approx(0.0)
    assert breakdown["r4"] == pytest.approx(0.0)
    assert breakdown["r5"] == pytest.approx(0.0)


def test_step_accepts_unsupported_action_and_returns_distinct_penalty() -> None:
    client = TestClient(app)
    _reset_episode(client)

    response = client.post(
        "/step",
        json={
            "action": json.dumps(
                {
                    "action_type": "ADD_NODE",
                    "params": {"node_id": "P99"},
                }
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    breakdown = payload["info"]["reward_breakdown"]

    assert payload["reward"] == pytest.approx(-0.75)
    assert payload["done"] is False
    assert breakdown["validation_penalty"] == pytest.approx(-0.75)
    assert breakdown["reason"] == "unsupported_action"
    assert {"r1", "r2", "r3", "r4", "r5"}.issubset(breakdown.keys())
