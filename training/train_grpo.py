"""Train DriftEnv with TRL GRPOTrainer and Unsloth.

This script:
1. Loads `unsloth/Meta-Llama-3.1-8B-Instruct` with 4-bit quantization.
2. Applies LoRA adapters on the requested attention projections.
3. Builds a small prompt dataset from local DriftEnv reset observations.
4. Scores sampled completions by calling the FastAPI `/reset` and `/step`
   endpoints exposed by `server.py`.
5. Trains with `GRPOTrainer`.
6. Merges LoRA weights back into the base model with `merge_and_unload()`.
7. Saves the merged model locally and uploads it to the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from datasets import Dataset
from huggingface_hub import HfApi
from transformers import set_seed
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

from environment.env import DriftEnv


MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
DEFAULT_ENV_BASE_URL = os.environ.get("DRIFTENV_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_OUTPUT_DIR = Path("outputs") / "driftenv-grpo"
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
        help="Destination Hugging Face model repo, e.g. username/driftenv-grpo.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Hugging Face token used to create/upload the target repo.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of reset observations to synthesize into training prompts.",
    )
    parser.add_argument(
        "--curriculum-stage",
        type=int,
        default=1,
        help="Curriculum stage passed into DriftEnv reset metadata.",
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
    num_samples: int,
    curriculum_stage: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Dataset:
    """Build GRPO prompts from deterministic DriftEnv reset observations."""

    episode_types = ("manager_departure", "policy_injection")
    records: list[dict[str, Any]] = []

    for seed in range(num_samples):
        episode_type = episode_types[seed % len(episode_types)]
        env = DriftEnv(
            episode_type=episode_type,
            curriculum_stage=curriculum_stage,
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
                "episode_type": episode_type,
                "curriculum_stage": curriculum_stage,
                "seed": seed,
            }
        )

    return Dataset.from_list(records)


def reward_fn(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
    """Score each sampled completion using the environment's /step endpoint."""

    del prompts  # The reset metadata below identifies which episode to replay.

    batch_size = len(completions)
    episode_types = _coerce_batch_column(
        kwargs.get("episode_type"),
        batch_size,
        "manager_departure",
    )
    curriculum_stages = _coerce_batch_column(
        kwargs.get("curriculum_stage"),
        batch_size,
        1,
    )
    seeds = _coerce_batch_column(kwargs.get("seed"), batch_size, 0)

    rewards: list[float] = []
    for completion, episode_type, curriculum_stage, seed in zip(
        completions,
        episode_types,
        curriculum_stages,
        seeds,
    ):
        completion_text = _extract_text(completion).strip()

        try:
            # Reset before every sampled action so each generation is evaluated
            # from the same deterministic starting observation.
            _post_json(
                REWARD_ENV_BASE_URL,
                "/reset",
                {
                    "episode_type": str(episode_type),
                    "curriculum_stage": int(curriculum_stage),
                    "seed": int(seed),
                },
            )

            # The server's /step endpoint wraps DriftEnv.step(action_text).
            step_result = _post_json(
                REWARD_ENV_BASE_URL,
                "/step",
                {"action": completion_text},
            )
            rewards.append(float(step_result.get("reward", 0.0)))
        except Exception as exc:  # noqa: BLE001 - reward functions should stay robust.
            print(f"[reward_fn] failed to score completion: {exc}")
            rewards.append(0.0)

    return rewards


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

    train_dataset = build_train_dataset(
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        curriculum_stage=args.curriculum_stage,
    )

    training_args = GRPOConfig(
        output_dir=str(output_dir),  # Trainer metadata and logs still need a working directory.
        learning_rate=5e-6,  # Requested conservative LR for stable online RL updates.
        per_device_train_batch_size=1,  # Requested micro-batch size to fit generation-heavy GRPO on one GPU.
        gradient_accumulation_steps=8,  # Requested accumulation to recover an effective batch of 8 prompts.
        num_train_epochs=1,  # Requested single-epoch pass over the synthetic reset dataset.
        num_generations=8,  # Requested 8 sampled completions per prompt for GRPO group comparison.
        max_prompt_length=2048,  # Requested prompt truncation budget.
        max_completion_length=512,  # Requested action-generation budget.
        remove_unused_columns=False,  # Keep episode metadata columns so reward_fn can replay the right env state.
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
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    save_and_push_merged_model(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        hub_model_id=args.hub_model_id,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
