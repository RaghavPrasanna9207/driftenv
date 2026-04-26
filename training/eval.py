"""Evaluate baseline vs trained policies on DriftEnv and save metrics."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable

from environment.env import DriftEnv
from training.inference import _generate_action_text


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for running policy evaluation rollouts."""

    parser = argparse.ArgumentParser(description="Evaluate DriftEnv policies.")
    parser.add_argument("--episode-type", default="manager_departure")
    parser.add_argument("--curriculum-stage", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default="artifacts/eval",
        help="Directory for metrics and per-episode JSON outputs.",
    )
    parser.add_argument(
        "--baseline-model",
        default="",
        help="Optional Hugging Face model id/path for baseline. Empty -> random policy.",
    )
    parser.add_argument(
        "--trained-model",
        default="",
        help="Optional Hugging Face model id/path for trained policy. Empty -> heuristic policy.",
    )
    return parser.parse_args()


def _random_action(_observation: str, rng: random.Random) -> str:
    """Return a random but schema-valid action JSON string."""

    actions = [
        {
            "action_type": "request_clarification",
            "params": {"question": "Could you confirm the intended recipient?"},
        },
        {
            "action_type": "flag_inconsistency",
            "params": {"node_id": rng.choice(["P1", "P2", "P3", "P4"])},
        },
        {
            "action_type": "draft_reply",
            "params": {"recipient_id": "P2", "message": "Acknowledged. I will follow up."},
        },
    ]
    return json.dumps(rng.choice(actions))


def _heuristic_action(observation: str, rng: random.Random) -> str:
    """Return a deterministic-ish heuristic action from observation hints."""

    del rng
    if "Potential inconsistency detected" in observation:
        return json.dumps({"action_type": "flag_inconsistency", "params": {"node_id": "P1"}})
    return json.dumps(
        {
            "action_type": "draft_reply",
            "params": {"recipient_id": "P2", "message": "Sending a status update as requested."},
        }
    )


def _load_model_policy(model_name_or_path: str) -> Callable[[str, random.Random], str]:
    """Load a HF model and return a policy callable."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    def policy(observation: str, rng: random.Random) -> str:
        del rng
        return _generate_action_text(model=model, tokenizer=tokenizer, observation=observation)

    return policy


def _run_policy(
    policy: Callable[[str, random.Random], str],
    episode_type: str,
    curriculum_stage: int,
    episodes: int,
) -> dict[str, Any]:
    """Run one policy across seeded episodes and return aggregate metrics."""

    rng = random.Random(2026)
    per_episode: list[dict[str, Any]] = []

    for seed in range(episodes):
        env = DriftEnv(episode_type=episode_type, curriculum_stage=curriculum_stage, seed=seed)
        observation = env.reset()
        total_reward = 0.0
        invalid_actions = 0
        turns = 0

        while not env.done:
            turns += 1
            action = policy(observation, rng)
            step_result = env.step(action)
            total_reward += float(step_result.get("reward", 0.0))
            info = dict(step_result.get("info", {}))
            validation = dict(info.get("validation_result", {}))
            if validation.get("validation_status") not in (None, "valid"):
                invalid_actions += 1
            observation = str(step_result.get("observation", ""))

        per_episode.append(
            {
                "seed": seed,
                "total_reward": total_reward,
                "turns": turns,
                "tasks_resolved": bool(env.tasks_resolved),
                "invalid_actions": invalid_actions,
            }
        )

    count = len(per_episode)
    rewards = [row["total_reward"] for row in per_episode]
    successes = [1.0 if row["tasks_resolved"] else 0.0 for row in per_episode]
    invalid_counts = [float(row["invalid_actions"]) for row in per_episode]
    turn_counts = [float(row["turns"]) for row in per_episode]
    return {
        "episodes": count,
        "mean_total_reward": sum(rewards) / max(count, 1),
        "success_rate": sum(successes) / max(count, 1),
        "mean_invalid_actions": sum(invalid_counts) / max(count, 1),
        "mean_turns": sum(turn_counts) / max(count, 1),
        "per_episode": per_episode,
    }


def main() -> None:
    """Run baseline/trained evaluation and write artifacts."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_policy = (
        _load_model_policy(args.baseline_model)
        if args.baseline_model.strip()
        else _random_action
    )
    trained_policy = (
        _load_model_policy(args.trained_model)
        if args.trained_model.strip()
        else _heuristic_action
    )

    baseline_metrics = _run_policy(
        policy=baseline_policy,
        episode_type=args.episode_type,
        curriculum_stage=args.curriculum_stage,
        episodes=args.episodes,
    )
    trained_metrics = _run_policy(
        policy=trained_policy,
        episode_type=args.episode_type,
        curriculum_stage=args.curriculum_stage,
        episodes=args.episodes,
    )

    summary = {
        "episode_type": args.episode_type,
        "curriculum_stage": args.curriculum_stage,
        "episodes": args.episodes,
        "baseline": baseline_metrics,
        "trained": trained_metrics,
        "delta": {
            "mean_total_reward": trained_metrics["mean_total_reward"] - baseline_metrics["mean_total_reward"],
            "success_rate": trained_metrics["success_rate"] - baseline_metrics["success_rate"],
            "mean_invalid_actions": trained_metrics["mean_invalid_actions"]
            - baseline_metrics["mean_invalid_actions"],
            "mean_turns": trained_metrics["mean_turns"] - baseline_metrics["mean_turns"],
        },
    }

    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "baseline_per_episode.json").write_text(
        json.dumps(baseline_metrics["per_episode"], indent=2), encoding="utf-8"
    )
    (output_dir / "trained_per_episode.json").write_text(
        json.dumps(trained_metrics["per_episode"], indent=2), encoding="utf-8"
    )
    print(f"Wrote eval metrics to {summary_path}")


if __name__ == "__main__":
    main()
