"""Inference helpers for comparing DriftEnv model behavior."""

from __future__ import annotations

import json
import textwrap
from itertools import zip_longest
from typing import Any

import torch

from environment.env import DriftEnv


MAX_NEW_TOKENS = 192
SIDE_BY_SIDE_WIDTH = 72


def format_turn(
    turn_num: int,
    action: str,
    signals: list[str],
    reward_breakdown: dict[str, Any],
) -> str:
    """Format a single turn summary for plain-text display."""

    clean_action = (action or "").strip() or "<empty>"
    signal_text = "; ".join(signals) if signals else "None"
    total_reward = float(
        reward_breakdown.get(
            "total_reward",
            sum(float(reward_breakdown.get(key, 0.0)) for key in ("r1", "r2", "r3", "r4", "r5")),
        )
    )
    breakdown_text = ", ".join(
        f"{key}={float(reward_breakdown.get(key, 0.0)):.3f}"
        for key in ("r1", "r2", "r3", "r4", "r5")
    )

    wrapped_action = textwrap.fill(
        clean_action,
        width=SIDE_BY_SIDE_WIDTH - 8,
        initial_indent="Action: ",
        subsequent_indent="        ",
    )
    wrapped_signals = textwrap.fill(
        signal_text,
        width=SIDE_BY_SIDE_WIDTH - 9,
        initial_indent="Signals: ",
        subsequent_indent="         ",
    )

    return "\n".join(
        [
            f"Turn {turn_num}",
            wrapped_action,
            wrapped_signals,
            f"Reward: total={total_reward:.3f} | {breakdown_text}",
        ]
    )


def generate_comparison(
    episode_type: str,
    baseline_model,
    trained_model,
    tokenizer,
) -> str:
    """Run the same episode twice and return a side-by-side comparison string."""

    baseline_trace = _run_episode(
        episode_type=episode_type,
        model=baseline_model,
        tokenizer=tokenizer,
        label="Baseline",
        seed=42,
    )
    trained_trace = _run_episode(
        episode_type=episode_type,
        model=trained_model,
        tokenizer=tokenizer,
        label="Trained",
        seed=42,
    )

    all_turns = max(len(baseline_trace["turns"]), len(trained_trace["turns"]))
    header = [
        f"DriftEnv Comparison",
        f"Episode: {episode_type}",
        f"Seed: 42",
        "",
        _side_by_side_line("BASELINE MODEL", "TRAINED MODEL"),
        _side_by_side_line("-" * 14, "-" * 13),
    ]

    body: list[str] = []
    for turn_index in range(all_turns):
        baseline_turn = baseline_trace["turns"][turn_index] if turn_index < len(baseline_trace["turns"]) else None
        trained_turn = trained_trace["turns"][turn_index] if turn_index < len(trained_trace["turns"]) else None

        left_block = (
            format_turn(
                baseline_turn["turn_num"],
                baseline_turn["action"],
                baseline_turn["signals"],
                baseline_turn["reward_breakdown"],
            )
            if baseline_turn is not None
            else "No turn data"
        )
        right_block = (
            format_turn(
                trained_turn["turn_num"],
                trained_turn["action"],
                trained_turn["signals"],
                trained_turn["reward_breakdown"],
            )
            if trained_turn is not None
            else "No turn data"
        )

        body.append(_side_by_side_block(left_block, right_block))
        body.append("-" * ((SIDE_BY_SIDE_WIDTH * 2) + 3))

    footer = [
        f"Baseline total reward: {baseline_trace['total_reward']:.3f}",
        f"Trained total reward:  {trained_trace['total_reward']:.3f}",
        f"Baseline done: {baseline_trace['done']}",
        f"Trained done:  {trained_trace['done']}",
    ]

    return "\n".join(header + body + footer)


def _run_episode(
    episode_type: str,
    model,
    tokenizer,
    label: str,
    seed: int,
) -> dict[str, Any]:
    """Run one deterministic episode rollout for a single model."""

    del label
    env = DriftEnv(episode_type=episode_type, curriculum_stage=1, seed=seed)
    observation = env.reset()

    turns: list[dict[str, Any]] = []
    total_reward = 0.0

    while not env.done:
        action_text = _generate_action_text(model, tokenizer, observation)
        step_result = env.step(action_text)
        info = step_result.get("info", {})
        action_result = info.get("action_result", {})

        turns.append(
            {
                "turn_num": info.get("turn", len(turns) + 1),
                "action": action_text,
                "signals": list(action_result.get("inconsistency_signals", [])),
                "reward_breakdown": {
                    **dict(info.get("reward_breakdown", {})),
                    "total_reward": float(step_result.get("reward", 0.0)),
                },
                "reward": float(step_result.get("reward", 0.0)),
            }
        )

        total_reward += float(step_result.get("reward", 0.0))
        observation = step_result["observation"]

    return {
        "turns": turns,
        "total_reward": total_reward,
        "done": env.done,
    }


def _generate_action_text(model, tokenizer, observation: str) -> str:
    """Generate one action string from a model given the current observation."""

    prompt = observation
    model_inputs = tokenizer(prompt, return_tensors="pt")
    model_device = _resolve_model_device(model)
    model_inputs = {
        key: value.to(model_device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }

    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "pad_token_id": _resolve_pad_token_id(tokenizer),
    }

    with torch.no_grad():
        output_ids = model.generate(**model_inputs, **generation_kwargs)

    prompt_length = model_inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if not generated_text:
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    return _extract_last_json_like_text(generated_text)


def _extract_last_json_like_text(text: str) -> str:
    """Return the last JSON object substring when present, otherwise raw text."""

    start_index: int | None = None
    depth = 0
    in_string = False
    escape_next = False
    last_json_candidate: str | None = None

    for index, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if in_string and char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_index is not None:
                candidate = text[start_index : index + 1]
                try:
                    json.loads(candidate)
                except json.JSONDecodeError:
                    start_index = None
                    continue
                last_json_candidate = candidate
                start_index = None

    return (last_json_candidate or text).strip()


def _resolve_model_device(model) -> torch.device:
    """Resolve the device that should receive tokenizer tensors."""

    if hasattr(model, "device"):
        return torch.device(model.device)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def _resolve_pad_token_id(tokenizer) -> int | None:
    """Choose a pad token id that works for generation."""

    if getattr(tokenizer, "pad_token_id", None) is not None:
        return tokenizer.pad_token_id
    return getattr(tokenizer, "eos_token_id", None)


def _side_by_side_block(left: str, right: str) -> str:
    """Render two multi-line blocks side by side with fixed-width columns."""

    left_lines = left.splitlines()
    right_lines = right.splitlines()
    combined_lines = [
        _side_by_side_line(left_line, right_line)
        for left_line, right_line in zip_longest(left_lines, right_lines, fillvalue="")
    ]
    return "\n".join(combined_lines)


def _side_by_side_line(left: str, right: str) -> str:
    """Render a single side-by-side line."""

    return f"{left:<{SIDE_BY_SIDE_WIDTH}} | {right}"
