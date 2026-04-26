"""Generate DriftEnv_Colab_Train.ipynb with Path A (train_grpo + API) and Path B (inline GRPO)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "DriftEnv_Colab_Train.ipynb"


def code(src: str) -> dict:
    lines = [line + "\n" for line in src.strip("\n").split("\n")]
    if not lines:
        lines = ["# empty\n"]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def md(text: str) -> dict:
    lines = [line + "\n" for line in text.strip("\n").split("\n")]
    if not lines:
        lines = ["\n"]
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def main() -> None:
    cells: list[dict] = [
        md(
            r"""# DriftEnv — Google Colab

This notebook supports **two** training paths:

- **Path A (repo default, matches `training/train_grpo.py`)** — Start the DriftEnv **FastAPI** server, then run `training/train_grpo.py`. The reward function calls your running server with `/reset` and `/step` and scores **full-episode** returns (as shipped in the repo).
- **Path B (legacy in-notebook / interactive)** — **No server**: `import` `DriftEnv` in the same process, build a custom TRL `reward_funcs` + `GRPOTrainer`, then optionally run a **multi-turn** evaluation rollout.

**Episode type:** use `manager_departure` (this replaces the old `EP-01` string, which is not a valid `episode_type` in the current code).

**Security:** do not commit Hugging Face tokens. Use Colab **Secrets** (`HF_TOKEN`) or set `os.environ` once per session.
"""
        ),
        code(
            r"""# --- Configuration ---
GITHUB_REPO = "https://github.com/RaghavPrasanna9207/driftenv.git"
REPO_DIR = "driftenv"

# Hub model id for `train_grpo.py` and optional Path B push
HF_HUB_MODEL_ID = "RaghavPrasanna9207/driftenv-grpo"  # change to your org/name

# Path A: local API
ENV_PORT = 8000
ENV_BASE_URL = f"http://127.0.0.1:{ENV_PORT}"
OUTPUT_DIR = "outputs/driftenv-grpo"

# Path B: in-process DriftEnv
EPISODE_TYPE = "manager_departure"  # not "EP-01"
CURRICULUM_STAGE = 1

# Optional: public Space URL if you want to point a *client* at hosted env (not used by Path B)
# Remapped from the old `OPENENV_BASE_URL` name to the variable `train_grpo` reads:
# DRIFTENV_BASE_URL is set after clone (Path A) or to your public Space if you only run a remote server.
# HF_SPACE_BASE_URL = "https://raghavprasanna9207-driftenv-env.hf.space"
"""
        ),
        code(
            r"""# Optional: load `HF_TOKEN` from Colab Secrets
import os

try:
    from google.colab import userdata
    t = userdata.get("HF_TOKEN")
    if t and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        os.environ["HF_TOKEN"] = t
        print("Loaded HF_TOKEN from Colab secrets.")
    else:
        print("Token already in env, or no Colab secret set.")
except Exception as exc:  # noqa: BLE001
    print("google.colab.userdata not available (local run?):", exc)
# If needed, set once:
# import getpass; os.environ["HF_TOKEN"] = getpass.getpass("HF token: ")
"""
        ),
        code(
            r"""# Clone or update the repo, then install (no shell $vars — works in Colab)
import os
import subprocess
import sys
from pathlib import Path

here = Path.cwd()
dest = here / REPO_DIR
if dest.is_dir() and (dest / ".git").is_dir():
    subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
else:
    subprocess.run(["git", "clone", GITHUB_REPO, str(dest)], check=True)

os.chdir(dest)
print("CWD =", os.getcwd())

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
"""
        ),
        code(
            r"""# Environment variables for `training/train_grpo.py` and the Hub
import os

os.environ["DRIFTENV_BASE_URL"] = ENV_BASE_URL
os.environ["HF_HUB_MODEL_ID"] = HF_HUB_MODEL_ID
if os.environ.get("HF_TOKEN"):
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
# Optional: improve cold-start / telemetry on slow networks (same as many Colab Unsloth setups)
# os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
# os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
"""
        ),
        md("## Path A — `uvicorn` + `train_grpo.py` (full-episode HTTP reward, matches repo)"),
        code(
            r"""# Start the API in the background (this runtime)
import os, subprocess, time, sys

if "server" in dir() and server is not None:
    try:
        server.terminate()
    except Exception:  # noqa: BLE001
        pass

server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", str(ENV_PORT)],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT,
)
time.sleep(3)
print("server pid =", server.pid)
"""
        ),
        code(
            r"""# Quick check that the API is up (avoids `curl` / shell pipes)
import json
import urllib.error
import urllib.request

def http_json(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return f"err: {e}"

print("health =", http_json(ENV_BASE_URL + "/health"))
print("state  =", http_json(ENV_BASE_URL + "/state")[:1000], "...")
"""
        ),
        code(
            r"""# Run `training/train_grpo.py` (Path A) — use subprocess so paths/tokens are safe
import os
import subprocess
import sys

if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
    raise RuntimeError("Set HF_TOKEN in Colab secrets or os.environ before training.")

subprocess.check_call(
    [
        sys.executable,
        "training/train_grpo.py",
        "--env-base-url",
        os.environ["DRIFTENV_BASE_URL"],
        "--output-dir",
        OUTPUT_DIR,
        "--hub-model-id",
        HF_HUB_MODEL_ID,
        "--hf-token",
        os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_HUB_TOKEN", ""),
    ]
)
"""
        ),
        code(
            r"""# List key outputs
from pathlib import Path

od = Path(OUTPUT_DIR)
print("output_dir exists:", od.is_dir())
if od.is_dir():
    for p in sorted(od.rglob("*.json")) + sorted(od.rglob("*.png")):
        if p.is_file() and p.stat().st_size < 1_000_000:
            print(p)
"""
        ),
        md(
            r"""## Path B — In-process `DriftEnv` + custom `GRPOTrainer` (old notebook style)

The original `driftenv.ipynb` did **not** use the HTTP server: it called `env.step` directly. That is a **different credit-assignment** setup than `train_grpo.py` (which sums rewards over a full episode rollout per completion).

The cells below:
1. Load Unsloth + LoRA
2. Define `format_for_server` and a TRL v0.9-style `env_reward_function`
3. Build a small prompt dataset
4. Run `GRPOTrainer` for a few steps
5. Optional `push_to_hub_merged` and a multi-turn rollout

**Note:** define `format_prompt` (this was missing in the old notebook but referenced in the dataset).
"""
        ),
        code(
            r"""# Unsloth model + LoRA (broader target_modules like the old Colab; adjust if VRAM is tight)
import os, torch
from unsloth import FastLanguageModel

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Model ready.")
"""
        ),
        code(
            r"""# In-process environment helpers
from __future__ import annotations

import json
from environment.env import DriftEnv

DEFAULT_SYSTEM_PROMPT = (
    "You are DriftEnv's executive assistant agent. Return exactly one JSON object with "
    '{\"action_type\": \"<action_name>\", \"params\": { ... }} and no other text.'
)


def make_env(seed: int) -> DriftEnv:
    return DriftEnv(episode_type=EPISODE_TYPE, curriculum_stage=CURRICULUM_STAGE, seed=seed)


def reset_state(seed: int):
    # Return a fresh (env, observation) for this seed.
    env = make_env(seed)
    return env, env.reset()


def format_prompt(observation: str) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": observation},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
"""
        ),
        code(
            r"""# TRL-style reward: one fresh episode per sample, one step (stable for GRPO batches)
import json
import re
from datasets import Dataset


def format_for_server(action_str: str) -> str:
    clean_str = re.sub(r"```json|```", "", str(action_str)).strip()
    try:
        json.loads(clean_str)
        return clean_str
    except json.JSONDecodeError:
        return json.dumps(
            {"action_type": "invalid_format_hallucination", "params": {"raw_text": clean_str[:200]}}
        )


def _extract_completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    return ""


def env_reward_function(completions, prompts, **kwargs):
    del prompts
    rewards: list[float] = []
    seed_col = kwargs.get("seed", [])
    for idx, completion in enumerate(completions):
        if isinstance(seed_col, (list, tuple)) and idx < len(seed_col):
            seed = int(seed_col[idx])
        else:
            seed = 1000 + idx
        env, _obs = reset_state(seed)
        action_str = _extract_completion_text(completion).strip()
        action_str = format_for_server(action_str)
        step_data = env.step(action_str)
        rewards.append(float(step_data.get("reward", 0.0)))
    return rewards


# Small dataset: same shape as the old notebook (prompts from many resets)
seeds = list(range(42, 52))
rows = []
for s in seeds:
    _env, obs = reset_state(s)
    rows.append({"prompt": format_prompt(obs), "seed": s})

grpo_train_dataset = Dataset.from_list(rows)
print("dataset rows =", len(grpo_train_dataset))
"""
        ),
        code(
            r"""# (Optional) Step 7.3 style sanity check — garbage vs valid JSON
one_env, one_obs = reset_state(999)
g = "I should move north and ask Marcus."
print("garbage ->", one_env.step(format_for_server(g))["reward"])
v = '{"action_type": "request_clarification", "params": {"question": "status?"}}'
print("valid ->", one_env.step(v)["reward"])
"""
        ),
        code(
            r"""# Path B: short GRPO run (tune for GPU memory)
import torch
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    output_dir="llama_3_1_8b_driftenv_inline_grpo",
    learning_rate=2e-5,
    logging_steps=1,
    max_steps=20,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_generations=4,
    max_prompt_length=1024,
    max_completion_length=256,
    fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else True,
    bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    report_to="none",
)

path_b_trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=grpo_train_dataset,
    processing_class=tokenizer,
    reward_funcs=env_reward_function,
)

# Uncomment to run:
# path_b_trainer.train()
print("Uncomment `path_b_trainer.train()` to start Path B training.")
"""
        ),
        code(
            r"""# Optional: push merged model (Path B) — set YOUR hub id
# from unsloth import FastLanguageModel
# model.push_to_hub_merged("YOUR_ID/Llama-3.1-8B-DriftEnv-RL", tokenizer, save_method="merged_16bit", token=os.environ.get("HF_TOKEN"))
pass
"""
        ),
        code(
            r"""# Full-episode rollout (one completion per turn), same style as the old last cell
import json
import torch
from unsloth import FastLanguageModel

env, obs = reset_state(101)
done = False
turn = 0
FastLanguageModel.for_inference(model)
while not done and turn < 15:
    turn += 1
    messages = [
        {
            "role": "system",
            "content": "Output exactly one JSON object. Schema: {\"action_type\": str, \"params\": object}",
        },
        {"role": "user", "content": f"Observation:\n{obs}\n\nAction:"},
    ]
    enc = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    if torch.cuda.is_available():
        enc = {k: v.to("cuda") for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.7
        )
    gen = tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    step_data = env.step(format_for_server(gen))
    obs = step_data.get("observation", "")
    done = bool(step_data.get("done"))
    print(f"turn {turn} reward={step_data.get('reward')} done={done}")
print("rollout end")
"""
        ),
        code(
            r"""# Stop Path A server when you are done
if "server" in dir() and server is not None:
    server.terminate()
    try:
        server.wait(timeout=5)
    except Exception:  # noqa: BLE001
        pass
    print("Server stopped")
"""
        ),
    ]

    data = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
