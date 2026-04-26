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

# Create https://huggingface.co/new-model then: "user/new-repo-name"
HF_HUB_MODEL_ID = "RaghavPrasanna9207/YOUR_MODEL_REPO_NAME"

USE_LOCAL_ENV_SERVER = True  # False -> use HF Space URL for /reset, /step (no uvicorn in Colab)
ENV_PORT = 8000
ENV_BASE_URL = (
    f"http://127.0.0.1:{ENV_PORT}" if USE_LOCAL_ENV_SERVER else HF_SPACE_APP_URL.rstrip("/")
)
OUTPUT_DIR = "outputs/driftenv-train"
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

here = Path.cwd()
dest = here / REPO_DIR
if dest.is_dir() and (dest / ".git").is_dir():
    subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
else:
    subprocess.run(["git", "clone", GITHUB_REPO, str(dest)], check=True)

os.chdir(dest)
print("CWD =", os.getcwd())

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-train.txt"]
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
            r"""# Check /health and /state
import urllib.error
import urllib.request


def http_get(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return f"err: {e}"


print("health =", http_get(ENV_BASE_URL + "/health"))
print("state  =", http_get(ENV_BASE_URL + "/state")[:1200])
"""
        ),
        code(
            r"""# Run GRPO training (needs HF token for Hub upload at end of script)
import os
import subprocess
import sys

if "YOUR_MODEL_REPO_NAME" in HF_HUB_MODEL_ID:
    raise ValueError("Set HF_HUB_MODEL_ID to your real user/model (create a Model repo on HF first).")
if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
    raise RuntimeError("Set HF_TOKEN (Colab Secret or os.environ) before training.")

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
            r"""# Outputs (curves, JSON summary, merged model folder)
from pathlib import Path

od = Path(OUTPUT_DIR)
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
