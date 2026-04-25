"""Observation-building utilities for DriftEnv."""

from __future__ import annotations

from typing import Any


def estimate_token_count(text: str) -> int:
    """Approximate token usage for a piece of text.

    The estimate is intentionally simple and deterministic, using the heuristic
    ``len(text.split()) * 1.3``.

    Args:
        text: Input text whose approximate token count should be estimated.

    Returns:
        An integer approximation of token count.
    """

    return int(len(text.split()) * 1.3)


def build_observation(
    subgraph_text: str,
    task_description: str,
    inconsistency_signals: list[Any],
    mutation_history: list[Any],
    active_policy_excerpt: str,
    turn_number: int,
) -> str:
    """Build the formatted DriftEnv LLM observation prompt.

    The resulting prompt includes all six requested input fields under stable
    section headers so an LLM can reliably parse the observation. The function
    also tries to keep the result under approximately 1200 tokens by trimming
    lower-priority sections if necessary.

    Args:
        subgraph_text: Serialized subgraph context, typically produced by the
            graph engine.
        task_description: Natural-language task or objective for the current
            step.
        inconsistency_signals: List of detected warning signs, anomalies, or
            repair cues.
        mutation_history: List describing prior mutations or repair actions.
        active_policy_excerpt: Relevant policy text that should guide the LLM.
        turn_number: Current environment turn number.

    Returns:
        A single formatted string suitable for use as an LLM prompt.
    """

    max_tokens = 1200
    section_order = [
        ("TURN", str(turn_number)),
        ("TASK_DESCRIPTION", _normalize_text(task_description)),
        ("ACTIVE_POLICY_EXCERPT", _normalize_text(active_policy_excerpt)),
        ("SUBGRAPH", _normalize_text(subgraph_text)),
        ("INCONSISTENCY_SIGNALS", _format_list_section(inconsistency_signals)),
        ("MUTATION_HISTORY", _format_list_section(mutation_history)),
    ]

    rendered_sections = {
        name: _render_section(name, content) for name, content in section_order
    }
    prompt = "\n\n".join(rendered_sections[name] for name, _ in section_order)
    if estimate_token_count(prompt) <= max_tokens:
        return prompt

    trimming_plan = [
        ("MUTATION_HISTORY", 180),
        ("INCONSISTENCY_SIGNALS", 180),
        ("ACTIVE_POLICY_EXCERPT", 260),
        ("SUBGRAPH", 420),
        ("TASK_DESCRIPTION", 180),
    ]

    section_content = {name: content for name, content in section_order}
    for section_name, word_budget in trimming_plan:
        section_content[section_name] = _trim_text(section_content[section_name], word_budget)
        rendered_sections[section_name] = _render_section(
            section_name,
            section_content[section_name],
        )
        prompt = "\n\n".join(rendered_sections[name] for name, _ in section_order)
        if estimate_token_count(prompt) <= max_tokens:
            return prompt

    compact_sections = []
    for name, _ in section_order:
        compact_sections.append(
            _render_section(name, _trim_text(section_content[name], 120))
        )
    compact_prompt = "\n\n".join(compact_sections)
    if estimate_token_count(compact_prompt) <= max_tokens:
        return compact_prompt

    final_words = compact_prompt.split()
    final_budget_words = max(1, int(max_tokens / 1.3))
    truncated = " ".join(final_words[:final_budget_words])
    if len(final_words) > final_budget_words:
        return truncated + " ...[TRUNCATED]"
    return truncated


def _render_section(header: str, content: str) -> str:
    """Render one observation section with a stable header."""

    normalized_content = content.strip() or "None"
    return f"## {header}\n{normalized_content}"


def _normalize_text(value: Any) -> str:
    """Convert observation content into a clean string."""

    if value is None:
        return "None"
    text = str(value).strip()
    return text if text else "None"


def _format_list_section(items: list[Any]) -> str:
    """Format a list into a bullet-style section body."""

    if not items:
        return "None"
    lines = []
    for item in items:
        text = _normalize_text(item)
        lines.append(f"- {text}")
    return "\n".join(lines)


def _trim_text(text: str, max_words: int) -> str:
    """Trim text to a rough word budget while preserving readability."""

    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ...[TRUNCATED]"
