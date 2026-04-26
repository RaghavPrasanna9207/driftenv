"""Generate DriftEnv_Colab_Train.ipynb: clone repo, run API + training/train_grpo.py only."""

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
            r"""# DriftEnv — Colab training (`train_grpo.py`)

**Code:** [github.com/RaghavPrasanna9207/driftenv](https://github.com/RaghavPrasanna9207/driftenv)

**Space (optional hosted API):** [huggingface.co/spaces/RaghavPrasanna9207/driftenv-env](https://huggingface.co/spaces/RaghavPrasanna9207/driftenv-env)

**Output model:** [Create an HF *Model* repo](https://huggingface.co/new-model) and set `HF_HUB_MODEL_ID` below (not the Space).

Runtime: **GPU** (T4+). Set `HF_TOKEN` in **Colab Secrets** before training.
"""
        ),
        code(
            r"""# --- Configuration ---
GITHUB_REPO = "https://github.com/RaghavPrasanna9207/driftenv.git"
REPO_DIR = "driftenv"

HF_SPACE_PAGE = "https://huggingface.co/spaces/RaghavPrasanna9207/driftenv-env"
HF_SPACE_APP_URL = "https://raghavprasanna9207-driftenv-env.hf.space"

# Upload target: create a *Model* repo at https://huggingface.co/new-model, then set org/name here
HF_HUB_MODEL_ID = "RaghavPrasanna9207/driftenv-grpo"

USE_LOCAL_ENV_SERVER = True  # False -> use HF Space URL for /reset, /step (no uvicorn in Colab)
ENV_PORT = 8000
ENV_BASE_URL = (
    f"http://127.0.0.1:{ENV_PORT}" if USE_LOCAL_ENV_SERVER else HF_SPACE_APP_URL.rstrip("/")
)
OUTPUT_DIR = "outputs/driftenv-train"

# Same base Instruct model as `training/train_grpo.py` (before your fine-tune)
BASELINE_HF_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
# Rollout count for `training/eval.py` (baseline vs merged checkpoint)
EVAL_EPISODES = 20
# Where `eval.py` writes JSON; plots read `metrics_summary.json` here
EVAL_DIR = f"{OUTPUT_DIR}/eval_baseline_trained"
"""
        ),
        code(
            r"""# Optional: `HF_TOKEN` from Colab Secrets
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
"""
        ),
        code(
            r"""# Clone and install
import os
import subprocess
import sys
from pathlib import Path

# Use a single stable path on Colab. If you use `Path.cwd() / "driftenv"` and your cwd
# is already .../driftenv, you get a broken nested path like /content/driftenv/driftenv.
def _default_repo_parent() -> Path:
    colab = Path("/content")
    return colab if colab.is_dir() else Path.cwd()


dest = (_default_repo_parent() / REPO_DIR).resolve()
if dest.is_dir() and (dest / ".git").is_dir():
    subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
else:
    subprocess.run(["git", "clone", GITHUB_REPO, str(dest)], check=True)

os.chdir(dest)
REPO_ROOT = dest
print("REPO_ROOT =", REPO_ROOT)
print("OK train_grpo.py:", (REPO_ROOT / "training" / "train_grpo.py").is_file())

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
# No -q: pip dependency errors (torch vs unsloth, etc.) are visible in the cell output
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-r", "requirements-train.txt"]
)
"""
        ),
        code(
            r"""# Env vars for `training/train_grpo.py`
import os

os.environ["DRIFTENV_BASE_URL"] = ENV_BASE_URL
os.environ["HF_HUB_MODEL_ID"] = HF_HUB_MODEL_ID
if os.environ.get("HF_TOKEN"):
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
print("DRIFTENV_BASE_URL =", os.environ["DRIFTENV_BASE_URL"], "(local Colab)" if USE_LOCAL_ENV_SERVER else "(HF Space app)")
print("Space page:", HF_SPACE_PAGE)
"""
        ),
        code(
            r"""# Start `uvicorn` in Colab (skip if using hosted Space)
import os, subprocess, time, sys

server = None
if not USE_LOCAL_ENV_SERVER:
    print("Using hosted API:", HF_SPACE_APP_URL)
else:
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", str(ENV_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    time.sleep(3)
    print("Local server pid =", server.pid)
"""
        ),
        code(
            r"""# Check /health and /state (no /reset yet -> GET /state is expected to 400)
import urllib.error
import urllib.request


def http_get(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} ({e.reason!s})"
    except Exception as e:  # noqa: BLE001
        return f"err: {e}"


print("health =", http_get(ENV_BASE_URL + "/health"))
print("state  =", http_get(ENV_BASE_URL + "/state")[:200], "... (400 until you POST /reset once)")
"""
        ),
        code(
            r"""# Run GRPO training (needs HF token for Hub upload at end of script)
import os
import subprocess
import sys
from pathlib import Path

_m = (HF_HUB_MODEL_ID or "").strip()
if not _m or "YOUR_MODEL_REPO_NAME" in _m:
    raise ValueError(
        "Set HF_HUB_MODEL_ID in the *first* cell to your Hugging Face model id (e.g. User/model-name). "
        "Create a repo: https://huggingface.co/new-model"
    )
if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
    raise RuntimeError("Set HF_TOKEN (Colab Secret or os.environ) before training.")


def _resolve_repo_root() -> Path:
    gr = globals().get("REPO_ROOT")
    if gr is not None and (gr / "training" / "train_grpo.py").is_file():
        return gr
    for candidate in (Path("/content") / "driftenv", Path("/content") / "driftenv" / "driftenv", Path.cwd()):
        if (candidate / "training" / "train_grpo.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot find training/train_grpo.py. Re-run the clone cell above with a clean path."
    )


repo = _resolve_repo_root()
os.chdir(repo)
print("Training cwd =", repo)

cmd = [
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
p = subprocess.run(cmd, cwd=str(repo), text=True, capture_output=True, env={**os.environ})
if p.stdout:
    print(p.stdout, end="")
if p.returncode != 0:
    print("--- train_grpo.py stderr (full traceback) ---", flush=True)
    if p.stderr:
        print(p.stderr, end="")
    raise RuntimeError(
        f"train_grpo.py exited with {p.returncode}. Read stderr above (OOM, import, or Hub errors are common)."
    )
"""
        ),
        md(
            r"""## Baseline vs trained evaluation

After training, this runs `training/eval.py` on:

- **Baseline:** the same 8B Instruct checkpoint used before GRPO (no DriftEnv fine-tune).
- **Trained:** your **merged** weights under `OUTPUT_DIR/merged` (written by `train_grpo.py`).

Models are loaded **one at a time** to reduce VRAM spikes on a single GPU. The next cell produces comparison plots (aggregate bars, per-episode curves, and deltas).
"""
        ),
        code(
            r"""# Run offline eval: base Instruct vs merged GRPO model (no API server needed for env — uses local `DriftEnv`)
import os
import subprocess
import sys
from pathlib import Path


def _eval_repo() -> Path:
    gr = globals().get("REPO_ROOT")
    if gr is not None and (gr / "training" / "eval.py").is_file():
        return gr
    for c in (Path("/content") / "driftenv", Path("/content") / "driftenv" / "driftenv", Path.cwd()):
        if (c / "training" / "eval.py").is_file():
            return c.resolve()
    return Path.cwd()


repo = _eval_repo()
merged = repo / OUTPUT_DIR / "merged"
if not merged.is_dir() or not any(merged.iterdir()):
    print("Skip eval: no merged checkpoint at", merged, "- finish training first.")
else:
    os.makedirs(repo / EVAL_DIR, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            "training/eval.py",
            "--episode-type",
            "manager_departure",
            "--curriculum-stage",
            "1",
            "--episodes",
            str(EVAL_EPISODES),
            "--output-dir",
            EVAL_DIR,
            "--baseline-model",
            BASELINE_HF_MODEL,
            "--trained-model",
            str(merged.resolve()),
        ],
        cwd=str(repo),
    )
    print("Wrote metrics to", repo / EVAL_DIR / "metrics_summary.json")
"""
        ),
        code(
            r"""# Plots: baseline vs trained (DriftEnv rollouts)
import json
from pathlib import Path

import matplotlib.pyplot as plt

_rp = globals().get("REPO_ROOT")
if _rp is not None and (_rp / EVAL_DIR / "metrics_summary.json").is_file():
    summary_path = _rp / EVAL_DIR / "metrics_summary.json"
else:
    for _c in (Path("/content") / "driftenv", Path("/content") / "driftenv" / "driftenv", Path.cwd()):
        _p = _c / EVAL_DIR / "metrics_summary.json"
        if _p.is_file():
            summary_path = _p
            break
    else:
        summary_path = Path(EVAL_DIR) / "metrics_summary.json"
if not summary_path.is_file():
    print("No", summary_path, "- run the eval cell after training creates merged/")
else:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    b, t, dlt = data["baseline"], data["trained"], data["delta"]
    color_b, color_t = "#4e79a7", "#f28e2b"  # blue = baseline, orange = trained

    # 1) Four aggregate metrics (same units per subplot)
    fig1, axs = plt.subplots(2, 2, figsize=(11, 8))
    fig1.suptitle("Aggregate metrics (higher reward & success better; lower invalid actions better)", fontsize=13)
    specs = [
        ("mean_total_reward", "Mean total reward", "reward"),
        ("success_rate", "Success rate", "fraction"),
        ("mean_invalid_actions", "Mean invalid actions", "count / ep"),
        ("mean_turns", "Mean turns to done", "turns"),
    ]
    for ax, (key, title, yl) in zip(axs.flat, specs):
        vals = [b[key], t[key]]
        ax.bar(
            [0, 1],
            vals,
            color=[color_b, color_t],
            width=0.55,
            edgecolor="white",
        )
        ax.set_xticks([0, 1], ["Baseline\n(base Instruct)", "Trained\n(merged GRPO)"])
        ax.set_title(title)
        ax.set_ylabel(yl)
    fig1.tight_layout()
    plt.show()

    # 2) Per-episode total return
    b_ep, t_ep = b["per_episode"], t["per_episode"]
    seeds = [row["seed"] for row in b_ep]
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(seeds, [row["total_reward"] for row in b_ep], "o-", color=color_b, label="Baseline", alpha=0.9)
    ax.plot(seeds, [row["total_reward"] for row in t_ep], "s-", color=color_t, label="Trained (GRPO)", alpha=0.9)
    ax.set_xlabel("Episode index (fixed seeds)")
    ax.set_ylabel("Total reward")
    ax.set_title("Per-episode return: same env seeds for both policies")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    plt.show()

    # 3) Per-episode invalid actions and episode length
    fig3, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4))
    ax_l.plot(seeds, [row["invalid_actions"] for row in b_ep], "o-", color=color_b, label="Baseline")
    ax_l.plot(seeds, [row["invalid_actions"] for row in t_ep], "s-", color=color_t, label="Trained")
    ax_l.set_xlabel("Episode")
    ax_l.set_ylabel("Invalid actions (count)")
    ax_l.set_title("Invalid action counts")
    ax_l.legend()
    ax_l.grid(True, alpha=0.3)
    ax_r.plot(seeds, [row["turns"] for row in b_ep], "o-", color=color_b, label="Baseline")
    ax_r.plot(seeds, [row["turns"] for row in t_ep], "s-", color=color_t, label="Trained")
    ax_r.set_xlabel("Episode")
    ax_r.set_ylabel("Turns")
    ax_r.set_title("Turns until episode end")
    ax_r.legend()
    ax_r.grid(True, alpha=0.3)
    fig3.tight_layout()
    plt.show()

    # 4) Task success (1 = resolved) + delta summary
    fig4, ax = plt.subplots(figsize=(10, 3.2))
    jitter = 0.08
    s_b = [float(row["tasks_resolved"]) for row in b_ep]
    s_t = [float(row["tasks_resolved"]) for row in t_ep]
    ax.scatter(
        [s - jitter for s in seeds],
        s_b,
        c=color_b,
        s=60,
        alpha=0.85,
        label="Baseline",
    )
    ax.scatter(
        [s + jitter for s in seeds],
        s_t,
        c=color_t,
        s=60,
        alpha=0.85,
        label="Trained",
    )
    ax.set_xlabel("Episode")
    ax.set_yticks([0, 1], ["Not resolved", "Resolved"])
    ax.set_title("Per-episode task resolution (1 = all tasks done)")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.2)
    fig4.tight_layout()
    plt.show()

    # 5) Trained - baseline (signed deltas). Green = moved in a helpful direction for that metric.
    keys = list(dlt.keys())
    yv = [dlt[k] for k in keys]
    labels = [k.replace("_", " ") for k in keys]
    bar_colors: list[str] = []
    for k, v in zip(keys, yv):
        if k in ("mean_total_reward", "success_rate"):
            bar_colors.append("#59a14f" if v >= 0 else "#e15759")
        elif k == "mean_invalid_actions":
            bar_colors.append("#59a14f" if v <= 0 else "#e15759")
        else:  # mean_turns: neutral (shorter is not always "success" in all scenarios)
            bar_colors.append("#bab0ac")
    fig5, ax5 = plt.subplots(figsize=(7.5, 3.5))
    ax5.barh(labels, yv, color=bar_colors, edgecolor="white")
    ax5.axvline(0, color="black", lw=0.6)
    ax5.set_xlabel("Trained minus baseline")
    ax5.set_title("Deltas: green = better reward/success or fewer invalid; gray = mean turns (contextual)")
    fig5.tight_layout()
    plt.show()
"""
        ),
        code(
            r"""# Outputs (curves, JSON summary, merged model folder)
from pathlib import Path

_root = globals().get("REPO_ROOT")
od = _root / OUTPUT_DIR if _root is not None else Path(OUTPUT_DIR)
if od.is_dir():
    for p in sorted(od.rglob("*.json")) + sorted(od.rglob("*.png")):
        if p.is_file() and p.stat().st_size < 2_000_000:
            print(p)
else:
    print("No output dir yet:", od)
"""
        ),
        code(
            r"""# Stop local server if we started it
if "server" in dir() and server is not None:
    server.terminate()
    try:
        server.wait(timeout=5)
    except Exception:  # noqa: BLE001
        pass
    print("Local server stopped.")
else:
    print("Done (no local server to stop).")
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
