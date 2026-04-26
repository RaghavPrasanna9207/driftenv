"""Action parsing utilities for DriftEnv."""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any

ACTION_DEFINITIONS: dict[str, dict[str, Any]] = {
    "draft_reply": {
        "category": "ea",
        "description": "Draft a reply to a recipient.",
        "required_params": ("recipient_id", "message"),
    },
    "reschedule_meeting": {
        "category": "ea",
        "description": "Propose a new meeting time.",
        "required_params": ("meeting_id", "new_time"),
    },
    "delegate_task": {
        "category": "ea",
        "description": "Delegate a task to another person.",
        "required_params": ("task_id", "assignee_id"),
    },
    "escalate": {
        "category": "ea",
        "description": "Escalate an issue to a target stakeholder.",
        "required_params": ("issue_id", "target_id"),
    },
    "decline_meeting": {
        "category": "ea",
        "description": "Decline a meeting with a reason.",
        "required_params": ("meeting_id", "reason"),
    },
    "flag_inconsistency": {
        "category": "repair",
        "description": "Flag one or more suspicious graph nodes.",
        "required_any_of": (("node_id",), ("node_ids",)),
    },
    "propose_node_removal": {
        "category": "repair",
        "description": "Propose removing a node from the belief graph.",
        "required_params": ("node_id",),
    },
    "propose_edge_update": {
        "category": "repair",
        "description": "Propose rewiring an edge to a new target.",
        "required_params": ("source", "old_target", "new_target"),
    },
    "propose_attribute_update": {
        "category": "repair",
        "description": "Propose updating a node attribute.",
        "required_params": ("node_id", "attribute_name", "new_value"),
    },
    "request_clarification": {
        "category": "repair",
        "description": "Request clarification about the current state.",
        "required_params": ("question",),
    },
}

EA_ACTION_TYPES = {
    action_type
    for action_type, definition in ACTION_DEFINITIONS.items()
    if definition["category"] == "ea"
}

REPAIR_ACTION_TYPES = {
    action_type
    for action_type, definition in ACTION_DEFINITIONS.items()
    if definition["category"] == "repair"
}

VALID_ACTION_TYPES = set(ACTION_DEFINITIONS)
SUPPORTED_ACTION_TYPES = tuple(ACTION_DEFINITIONS.keys())


def parse_action(raw_text: str) -> dict[str, Any]:
    """Parse and validate the last valid JSON action object in an LLM response.

    The function scans the input text for JSON objects, attempts to decode each
    candidate object, and keeps the last successfully decoded top-level JSON
    object. That object is then validated against the DriftEnv action schema.

    Expected JSON shape:

    ``{"action_type": "<allowed_action>", "params": {...}}``

    The returned dictionary always contains the keys ``action_type``,
    ``params``, ``valid``, and ``error_message`` so callers can handle success
    and failure uniformly.

    Args:
        raw_text: Raw text emitted by an LLM. The text may contain prose,
            multiple JSON snippets, markdown fences, or malformed JSON before
            the final valid object.

        Returns:
                A normalized result dictionary with:
                - ``action_type``: The validated action type, or ``"parse_error"``
                    when no valid JSON object can be extracted.
                - ``params``: The parsed params object when available, otherwise an
                    empty dictionary.
                - ``valid``: ``True`` when both parsing and schema validation succeed,
                    else ``False``.
                - ``validation_status``: ``"valid"``, ``"invalid_json"``,
                    ``"unsupported_action"``, or ``"wrong_parameters"``.
                - ``error_message``: ``None`` on success, otherwise a human-readable
                    explanation of the parsing or validation failure.

    Edge Cases:
        - If no valid JSON object is found anywhere in the text, the function
          returns ``valid=False`` and ``action_type='parse_error'``.
        - If the last valid JSON value is not an object, it is ignored and the
          scan continues looking for a valid JSON object.
        - If the JSON object is parsed successfully but fails schema
          validation, the parsed ``action_type`` is returned when possible and
          ``valid`` is set to ``False``.
        - Extra keys are preserved in the parsed JSON object only indirectly;
          the returned structure is normalized to the required fields.
    """

    parsed_object = _extract_last_json_object(raw_text)
    if parsed_object is None:
        return {
            "action_type": "parse_error",
            "params": {},
            "valid": False,
            "validation_status": "invalid_json",
            "error_message": "No valid JSON object found in raw_text.",
            "missing_params": [],
            "supported_action_types": list(SUPPORTED_ACTION_TYPES),
        }

    return _validate_action_object(parsed_object)


def build_action_schema_text() -> str:
    """Render a strict, unambiguous action contract for the policy."""

    lines = [
        "The only top-level keys are: \"action_type\" (string) and \"params\" (object).",
        "Do NOT use prose, section titles, or explanatory text as parameter keys. Use only the",
        "exact field names listed below for each action_type's params object.",
        "",
        "flag_inconsistency: params must contain exactly one of the following, not both:",
        '  * \"node_id\": <string> — single node identifier that exists in the current graph, OR',
        '  * \"node_ids\": <list of strings> — each string must be a node id that exists in the current graph.',
        "",
        "Required params for other action types (params must use exactly these key names):",
    ]

    for action_type in SUPPORTED_ACTION_TYPES:
        if action_type == "flag_inconsistency":
            continue
        definition = ACTION_DEFINITIONS[action_type]
        if "required_params" in definition:
            names = ", ".join(definition["required_params"])
            lines.append(f"- {action_type}: {names}")
        else:
            required_any_of = definition.get("required_any_of", ())
            for option in required_any_of:
                alts = ", ".join(option)
                lines.append(
                    f"- {action_type}: provide one of these key sets: {alts} (separate key names, not a combined label)."
                )

    lines.extend(
        [
            "",
            "Return exactly one JSON object with 'action_type' and 'params'.",
            "Disallowed action_type string values: ADD_NODE, ADD_EDGE, REPAIR_GRAPH, LOG_INCONSISTENCY.",
            "Full list of allowed action_type values:",
            ", ".join(SUPPORTED_ACTION_TYPES),
        ]
    )
    return "\n".join(lines)


def _extract_last_json_object(raw_text: str) -> dict[str, Any] | None:
    """Extract the last decodable top-level JSON object from mixed text."""

    last_object: dict[str, Any] | None = None
    object_start: int | None = None
    depth = 0
    in_string = False
    escape_next = False

    for index, character in enumerate(raw_text):
        if escape_next:
            escape_next = False
            continue

        if character == "\\" and in_string:
            escape_next = True
            continue

        if character == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if character == "{":
            if depth == 0:
                object_start = index
            depth += 1
            continue

        if character == "}" and depth > 0:
            depth -= 1
            if depth == 0 and object_start is not None:
                candidate_text = raw_text[object_start : index + 1]
                try:
                    candidate = json.loads(candidate_text)
                except JSONDecodeError:
                    object_start = None
                    continue
                if isinstance(candidate, dict):
                    last_object = candidate
                object_start = None

    return last_object


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _missing_params_for_action(action_type: str, params: dict[str, Any]) -> list[str]:
    definition = ACTION_DEFINITIONS[action_type]
    if "required_params" in definition:
        return [
            param_name
            for param_name in definition["required_params"]
            if param_name not in params or _is_missing(params[param_name])
        ]

    required_any_of = definition.get("required_any_of", ())
    for option in required_any_of:
        if all(param_name in params and not _is_missing(params[param_name]) for param_name in option):
            return []

    flattened_options = ["/".join(option) for option in required_any_of]
    return [" or ".join(flattened_options) if flattened_options else "params"]


def _validate_action_object(action_object: dict[str, Any]) -> dict[str, Any]:
    """Validate a parsed action object and normalize the result payload."""

    action_type = action_object.get("action_type")
    params = action_object.get("params")

    if not isinstance(action_type, str):
        return {
            "action_type": "parse_error",
            "params": {},
            "valid": False,
            "validation_status": "invalid_json",
            "error_message": "Parsed JSON object is missing a string 'action_type'.",
            "missing_params": [],
            "supported_action_types": list(SUPPORTED_ACTION_TYPES),
        }

    if action_type not in ACTION_DEFINITIONS:
        return {
            "action_type": action_type,
            "params": params if isinstance(params, dict) else {},
            "valid": False,
            "validation_status": "unsupported_action",
            "error_message": (
                f"Unsupported action_type: {action_type}. Allowed action_types: "
                f"{', '.join(SUPPORTED_ACTION_TYPES)}."
            ),
            "missing_params": [],
            "supported_action_types": list(SUPPORTED_ACTION_TYPES),
        }

    if not isinstance(params, dict):
        return {
            "action_type": action_type,
            "params": {},
            "valid": False,
            "validation_status": "wrong_parameters",
            "error_message": "Parsed JSON object must contain a 'params' object.",
            "missing_params": ["params"],
            "supported_action_types": list(SUPPORTED_ACTION_TYPES),
        }

    missing_params = _missing_params_for_action(action_type, params)
    if missing_params:
        return {
            "action_type": action_type,
            "params": params,
            "valid": False,
            "validation_status": "wrong_parameters",
            "error_message": (
                f"Parsed action_type {action_type} is missing required params: "
                f"{', '.join(missing_params)}."
            ),
            "missing_params": missing_params,
            "supported_action_types": list(SUPPORTED_ACTION_TYPES),
        }

    return {
        "action_type": action_type,
        "params": params,
        "valid": True,
        "validation_status": "valid",
        "error_message": None,
        "missing_params": [],
        "supported_action_types": list(SUPPORTED_ACTION_TYPES),
    }
