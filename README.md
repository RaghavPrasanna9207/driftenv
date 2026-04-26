---
title: driftenv
sdk: docker
app_port: 7860
---

# DriftEnv

DriftEnv is an OpenEnv-style environment for training LLM agents to handle executive-assistant workflows under hidden graph drift.

The agent must complete practical actions (`draft_reply`, `delegate_task`, etc.), detect hidden graph mutations, and propose repairs with strict JSON actions.

## Theme Alignment

This environment targets:
- **Theme #3 (World Modeling / Personalized Tasks)** via dynamic org-graph execution.
- **Theme #2 (Long-Horizon Planning)** through multi-turn mutation detection and repair.

## Why this environment matters

LLMs often fail when world state changes silently mid-workflow. DriftEnv trains agents to:
- keep a stable internal model of a changing graph,
- detect and flag inconsistencies,
- avoid reward-gaming shortcuts,
- recover with meaningful repairs.

## Repository Layout

- `environment/` - core environment components
  - `env.py` - `DriftEnv` reset/step orchestration
  - `graph_engine.py` - graph loading/mutations/subgraph extraction
  - `action_parser.py` - strict action schema and validation
  - `observation.py` - prompt/observation composition
  - `reward.py` - reward components and shaping terms
- `server.py` - FastAPI surface (`/reset`, `/step`, `/state`, `/tasks`)
- `training/train_grpo.py` - GRPO + Unsloth training pipeline
- `training/eval.py` - baseline-vs-trained rollout metrics
- `training/inference.py` - side-by-side rollout comparison helpers
- `inference.py` - root inference entrypoint
- `openenv.yaml` - OpenEnv metadata manifest
- `tests/` - unit/integration tests

## API Contract

### `POST /reset`
Creates a new active episode.

Example body:
```json
{
  "episode_type": "manager_departure",
  "curriculum_stage": 1,
  "seed": 42
}
```

### `POST /step`
Advances one turn using a raw action string (or JSON object).

Example body:
```json
{
  "action": "{\"action_type\":\"flag_inconsistency\",\"params\":{\"node_id\":\"P1\"}}"
}
```

### `GET /state`
Returns live episode snapshot (turn, done, mutation history, graph stats, valid node IDs).

### `GET /tasks`
Returns high-level scenario task cards for the active episode.

### `GET /health`
Basic API liveness.

## Local Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Environment Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

For Hugging Face Spaces (Docker), the app runs on port `7860`.

## Training (GRPO + Unsloth)

```bash
python training/train_grpo.py \
  --env-base-url http://127.0.0.1:8000 \
  --output-dir outputs/driftenv-grpo \
  --hub-model-id <your-username>/driftenv-grpo \
  --hf-token <hf_token>
```

Training outputs include:
- `outputs/driftenv-grpo/episode_reward_curve.png`
- `outputs/driftenv-grpo/artifacts/training_summary.json`
- merged model checkpoint under `outputs/driftenv-grpo/merged/`

## Evaluation (Baseline vs Trained)

Quick eval with default random-vs-heuristic policies:

```bash
python training/eval.py \
  --episode-type manager_departure \
  --curriculum-stage 1 \
  --episodes 20 \
  --output-dir artifacts/eval
```

Model-backed eval:

```bash
python training/eval.py \
  --baseline-model <base_model_or_path> \
  --trained-model <trained_model_or_path> \
  --episodes 20 \
  --output-dir artifacts/eval
```

Expected evaluation artifacts:
- `artifacts/eval/metrics_summary.json`
- `artifacts/eval/baseline_per_episode.json`
- `artifacts/eval/trained_per_episode.json`

## Inference Comparison Report

```bash
python inference.py \
  --baseline-model <base_model_or_path> \
  --trained-model <trained_model_or_path> \
  --episode-type manager_departure \
  --output artifacts/eval/inference_comparison.txt
```

## Testing

```bash
python -m pytest tests/ -q
```

## Results Checklist (Before Submission)

- [ ] Environment hosted and runnable on Hugging Face Space.
- [ ] `openenv.yaml` present and accurate.
- [ ] README includes train/eval commands and artifact locations.
- [ ] Reward curve image committed (`.png` or `.jpg`).
- [ ] Baseline vs trained metrics committed.
- [ ] Demo links added below.

## Submission Links

- Hugging Face Space: `TODO`
- Hugging Face model/blog: `TODO`
- Demo video (<2 min): `TODO`
- Slides or extra notes: `TODO`
