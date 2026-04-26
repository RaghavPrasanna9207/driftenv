"""Train DriftEnv with TRL GRPOTrainer and Unsloth.

This script:
1. Loads `unsloth/Meta-Llama-3.1-8B-Instruct` with 4-bit quantization.
2. Applies LoRA adapters on the requested attention projections.
3. Builds a prompt dataset from ``NUM_EPISODES`` (Phase 8: 50) local DriftEnv
   runs with episode_type=manager_departure and curriculum_stage=1.
4. Scores each sampled action with a full reset -> step until done rollout
   (same action at each step) using the FastAPI `/reset` and `/step` server,
   summing per-step reward as the return for GRPO.
5. Trains with `GRPOTrainer` and records per-episode means in `episode_rewards`
   (console: ``Episode X Reward: Y``).
6. Merges LoRA weights with `merge_and_unload()`.
7. Saves the merged model locally and uploads to the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset
from huggingface_hub import HfApi
from transformers import set_seed
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

from environment.env import DriftEnv


# ---------------------------------------------------------------------------
# Phase 8: fixed run configuration (not implicit dataset-length defaults)
# ---------------------------------------------------------------------------
NUM_EPISODES = 50
EPISODE_TYPE = "manager_departure"  # EP-01; no policy_injection
CURRICULUM_STAGE = 1

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
DEFAULT_ENV_BASE_URL = os.environ.get("DRIFTENV_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_OUTPUT_DIR = Path("outputs") / "driftenv-train"
REWARD_ENV_BASE_URL = DEFAULT_ENV_BASE_URL
DEFAULT_SYSTEM_PROMPT = (
    "You are controlling DriftEnv as an executive assistant agent. "
    "Return exactly one JSON object with the schema "
    '{"action_type":"<action_name>","params":{...}}. '
    "Do not include markdown, prose, or explanations."
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training and Hub upload."""

    parser = argparse.ArgumentParser(description="Train DriftEnv with GRPO + Unsloth.")
    parser.add_argument(
        "--env-base-url",
        default=DEFAULT_ENV_BASE_URL,
        help="Base URL of the FastAPI/OpenEnv server that exposes /reset and /step.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory used for trainer artifacts and the merged model export.",
    )
    parser.add_argument(
        "--hub-model-id",
        default=os.environ.get("HF_HUB_MODEL_ID"),
        help="Destination Hugging Face *model* repo for merged weights (create an empty model repo on HF), e.g. username/my-driftenv-model.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Hugging Face token used to create/upload the target repo.",
    )
    return parser.parse_args()


def _post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    """POST a JSON payload and return the decoded JSON response."""

    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {body}") from exc


def _coerce_batch_column(value: Any, length: int, default: Any) -> list[Any]:
    """Normalize a dataset column forwarded by TRL into a batch-aligned list."""

    if value is None:
        return [default] * length
    if isinstance(value, list):
        return value
    return [value] * length


def _extract_text(value: Any) -> str:
    """Convert TRL prompt/completion payloads into plain text."""

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content", "")
        return _extract_text(content)
    if isinstance(value, list):
        parts = [_extract_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return str(value)


def build_train_dataset(
    tokenizer: Any,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Dataset:
    """Build GRPO prompts: one record per of ``NUM_EPISODES`` (reset -> rollout -> done) runs."""

    records: list[dict[str, Any]] = []

    for _episode_num in range(NUM_EPISODES):
        # One env episode index (1..NUM_EPISODES) with distinct seeds (0..NUM_EPISODES-1).
        seed = _episode_num
        env = DriftEnv(
            episode_type=EPISODE_TYPE,
            curriculum_stage=CURRICULUM_STAGE,
            seed=seed,
        )
        observation = env.reset()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        records.append(
            {
                "prompt": prompt,
                "episode_type": EPISODE_TYPE,
                "curriculum_stage": CURRICULUM_STAGE,
                "seed": seed,
                "episode_id": _episode_num + 1,
            }
        )

    return Dataset.from_list(records)


def _rollout_total_return(
    base_url: str,
    seed: int,
    action_text: str,
) -> float:
    """One full env episode: /reset then /step with the same action until done; return sum of step rewards."""

    _post_json(
        base_url,
        "/reset",
        {
            "episode_type": EPISODE_TYPE,
            "curriculum_stage": CURRICULUM_STAGE,
            "seed": int(seed),
        },
    )
    total = 0.0
    # DriftEnv.MAX_TURNS is 12; keep a hard cap for safety.
    for _ in range(32):
        step_result = _post_json(
            base_url,
            "/step",
            {"action": action_text},
        )
        total += float(step_result.get("reward", 0.0))
        if bool(step_result.get("done")):
            break
    return total


def make_env_reward_fn(
    episode_rewards: list[float],
    episode_logged: list[bool],
) -> Any:
    """Build a reward function that records per-episode mean returns (one row = one of ``NUM_EPISODES`` trials)."""

    def env_reward_fn(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
        del prompts

        batch_size = len(completions)
        seeds = _coerce_batch_column(kwargs.get("seed"), batch_size, 0)
        episode_ids = _coerce_batch_column(kwargs.get("episode_id"), batch_size, 0)

        rewards: list[float] = []
        for completion, seed, episode_id in zip(completions, seeds, episode_ids):
            completion_text = _extract_text(completion).strip()

            try:
                total_return = _rollout_total_return(
                    REWARD_ENV_BASE_URL,
                    int(seed),
                    completion_text,
                )
                rewards.append(total_return)
            except Exception as exc:  # noqa: BLE001
                print(f"[env_reward_fn] failed to score completion: {exc}")
                rewards.append(0.0)

        # One logging line per GRPO prompt group (``num_generations`` samples share one episode_id / seed).
        ep_id = int(episode_ids[0]) if episode_ids else 0
        if 1 <= ep_id <= NUM_EPISODES:
            mean_r = sum(rewards) / max(len(rewards), 1)
            episode_rewards[ep_id - 1] = mean_r
            if not episode_logged[ep_id - 1]:
                episode_logged[ep_id - 1] = True
                print(f"Episode {ep_id} Reward: {round(mean_r, 6)}")

        return rewards

    return env_reward_fn


def plot_reward_curve(episode_rewards: list, save_path: str) -> float | None:
    """Plot episode number vs total reward and return tail-trend slope when available."""

    n = len(episode_rewards)
    if n == 0:
        print("plot_reward_curve: empty episode_rewards; nothing to plot.")
        return None

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(1, n + 1, dtype=float)
    rewards = np.asarray(episode_rewards, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, rewards, marker="o", markersize=3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Episode reward curve")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    k = min(20, n)
    y_tail = rewards[-k:]
    x_tail = episodes[-k:]
    if k < 2:
        print("plot_reward_curve: need at least 2 points for polyfit; cannot report slope.")
        return None
    slope, _intercept = np.polyfit(x_tail, y_tail, 1)
    print(f"Slope of last {k} episodes (linear fit, numpy.polyfit): {slope}")
    return float(slope)


def write_training_summary(
    output_dir: Path,
    episode_rewards: list[float],
    reward_curve_path: Path,
    tail_slope: float | None,
) -> Path:
    """Persist a compact JSON summary with learning-signal artifacts."""

    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "num_episodes": len(episode_rewards),
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "max_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "min_reward": float(np.min(episode_rewards)) if episode_rewards else 0.0,
        "tail_slope_last_20": tail_slope,
        "reward_curve_path": str(reward_curve_path),
        "episode_rewards": [float(value) for value in episode_rewards],
    }
    summary_path = artifacts_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote training summary to: {summary_path}")
    return summary_path


def save_and_push_merged_model(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    hub_model_id: str,
    hf_token: str,
) -> None:
    """Merge LoRA weights, save locally, then upload the merged checkpoint."""

    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # The user specifically requested merge_and_unload() so we export the final
    # model by materializing merged weights first instead of saving adapters only.
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hub_model_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=hub_model_id,
        repo_type="model",
        folder_path=str(merged_dir),
    )


def main() -> None:
    """Run end-to-end GRPO training, merged export, and Hub upload."""

    global REWARD_ENV_BASE_URL

    args = parse_args()
    if not args.hub_model_id:
        raise ValueError("Provide --hub-model-id or set HF_HUB_MODEL_ID.")
    if not args.hf_token:
        raise ValueError("Provide --hf-token or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    REWARD_ENV_BASE_URL = args.env_base_url

    set_seed(3407)

    max_seq_length = 3072
    dtype = None

    # Use a 4-bit Unsloth checkpoint because GRPO is generation-heavy and the
    # smaller memory footprint keeps 8 sampled completions practical on 1 GPU.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,  # Covers 2048 prompt tokens + 512 completion tokens with headroom.
        dtype=dtype,  # Let Unsloth auto-pick FP16/BF16 based on the available GPU.
        load_in_4bit=True,  # Requested QLoRA setup for lower VRAM usage during RL.
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Rank 16 / alpha 32 is the requested LoRA capacity: large enough to adapt
    # action formatting behavior without the full memory cost of dense tuning.
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Requested LoRA rank.
        target_modules=["q_proj", "v_proj"],  # Restrict adaptation to the requested attention projections.
        lora_alpha=32,  # Requested scaling to match the chosen rank.
        lora_dropout=0,  # Keep dropout disabled because Unsloth optimizes this path best.
        bias="none",  # Leave bias weights frozen to minimize trainable parameter count.
        use_gradient_checkpointing="unsloth",  # Trade extra compute for lower activation memory during GRPO.
        random_state=3407,  # Fix the adapter init for reproducible runs.
        max_seq_length=max_seq_length,  # Keep RoPE scaling aligned with the loaded context window.
        use_rslora=False,  # Stick to vanilla LoRA since the user asked for standard rank/alpha settings.
        loftq_config=None,  # Disable LoftQ because the requested setup is standard 4-bit QLoRA.
    )

    # Phase 8: one slot per `episode_id` (1..NUM_EPISODES) for plotting; filled during `make_env_reward_fn`.
    episode_rewards: list[float] = []
    episode_rewards.extend(0.0 for _ in range(NUM_EPISODES))
    episode_reward_logged: list[bool] = [False] * NUM_EPISODES
    train_dataset = build_train_dataset(tokenizer=tokenizer)

    training_args = GRPOConfig(
        output_dir=str(output_dir),  # Trainer metadata and logs still need a working directory.
        learning_rate=5e-6,  # Requested conservative LR for stable online RL updates.
        per_device_train_batch_size=1,  # Requested micro-batch size to fit generation-heavy GRPO on one GPU.
        gradient_accumulation_steps=8,  # Requested accumulation to recover an effective batch of 8 prompts.
        num_train_epochs=1,  # One pass over exactly ``NUM_EPISODES`` training rows (50 Phase 8 episodes).
        num_generations=8,  # Requested 8 sampled completions per prompt for GRPO group comparison.
        max_prompt_length=2048,  # Requested prompt truncation budget.
        max_completion_length=512,  # Requested action-generation budget.
        remove_unused_columns=False,  # Keep episode metadata columns so the env reward can read seed / episode_id.
        shuffle_dataset=False,  # Deterministic order: episode_id 1..NUM_EPISODES.
        bf16=is_bfloat16_supported(),  # Prefer BF16 on newer GPUs for speed/stability when supported.
        fp16=not is_bfloat16_supported(),  # Fall back to FP16 on older GPUs like T4/V100.
        optim="adamw_8bit",  # Use an 8-bit optimizer to reduce optimizer-state memory overhead.
        logging_steps=1,  # Log every step because RL training is noisy and benefits from close monitoring.
        save_strategy="no",  # Skip trainer checkpoints because we export one final merged model at the end.
        report_to="none",  # Keep the script self-contained unless the user wires in an experiment tracker.
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=make_env_reward_fn(episode_rewards, episode_reward_logged),
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    reward_curve_path = output_dir / "episode_reward_curve.png"
    tail_slope = plot_reward_curve(episode_rewards, str(reward_curve_path))
    write_training_summary(
        output_dir=output_dir,
        episode_rewards=episode_rewards,
        reward_curve_path=reward_curve_path,
        tail_slope=tail_slope,
    )

    save_and_push_merged_model(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        hub_model_id=args.hub_model_id,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
