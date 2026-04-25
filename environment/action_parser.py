"""Action parsing utilities for DriftEnv."""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any


EA_ACTION_TYPES = {
    "draft_reply",
    "reschedule_meeting",
    "delegate_task",
    "escalate",
    "decline_meeting",
}

REPAIR_ACTION_TYPES = {
    "flag_inconsistency",
    "propose_node_removal",
    "propose_edge_update",
    "propose_attribute_update",
    "request_clarification",
}

VALID_ACTION_TYPES = EA_ACTION_TYPES | REPAIR_ACTION_TYPES


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
            "error_message": "No valid JSON object found in raw_text.",
        }

    return _validate_action_object(parsed_object)


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


def _validate_action_object(action_object: dict[str, Any]) -> dict[str, Any]:
    """Validate a parsed action object and normalize the result payload."""

    action_type = action_object.get("action_type")
    params = action_object.get("params")

    if not isinstance(action_type, str):
        return {
            "action_type": "parse_error",
            "params": {},
            "valid": False,
            "error_message": "Parsed JSON object is missing a string 'action_type'.",
        }

    if action_type not in VALID_ACTION_TYPES:
        return {
            "action_type": action_type,
            "params": params if isinstance(params, dict) else {},
            "valid": False,
            "error_message": f"Unsupported action_type: {action_type}.",
        }

    if not isinstance(params, dict):
        return {
            "action_type": action_type,
            "params": {},
            "valid": False,
            "error_message": "Parsed JSON object must contain a 'params' object.",
        }

    return {
        "action_type": action_type,
        "params": params,
        "valid": True,
        "error_message": None,
    }
