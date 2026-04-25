"""FastAPI server that exposes a single active DriftEnv episode."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from environment.env import DriftEnv


app = FastAPI(title="DriftEnv API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


current_episode: DriftEnv | None = None


class ResetRequest(BaseModel):
    """Request body for creating and resetting a DriftEnv episode."""

    episode_type: str = Field(..., description="Episode scenario label.")
    curriculum_stage: int = Field(..., description="Curriculum difficulty stage.")
    seed: int = Field(..., description="Seed for deterministic episode behavior.")

    @field_validator("episode_type")
    @classmethod
    def validate_episode_type(cls, value: str) -> str:
        """Reject empty episode labels with a clearer validation error."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("episode_type must be a non-empty string.")
        return normalized


class ResetResponse(BaseModel):
    """Response body returned after resetting an episode."""

    observation: str
    info: dict[str, Any]


class StepRequest(BaseModel):
    """Request body for advancing the active episode by one step."""

    action: str = Field(
        ...,
        description=(
            "Raw model output for the action field. The environment will parse "
            "and validate the text, so invalid JSON can still be penalized."
        ),
    )

    @field_validator("action", mode="before")
    @classmethod
    def normalize_action(cls, value: Any) -> str:
        """Accept either a JSON object or raw text and normalize to a string."""

        if isinstance(value, (dict, list)):
            return json.dumps(value)
        if not isinstance(value, str):
            raise ValueError(
                "action must be a string or a JSON-serializable object."
            )

        normalized = value.strip()
        if not normalized:
            raise ValueError("action must not be empty.")

        return normalized


class StepResponse(BaseModel):
    """Response body returned after stepping the active episode."""

    observation: str
    reward: float
    done: bool
    info: dict[str, Any]


class HealthResponse(BaseModel):
    """Health-check response body."""

    status: str


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Any, exc: RequestValidationError
) -> JSONResponse:
    """Return slightly more actionable 422 responses for malformed JSON bodies."""

    normalized_errors: list[dict[str, Any]] = []
    for error in exc.errors():
        normalized_error = dict(error)
        context = normalized_error.get("ctx")
        if isinstance(context, dict):
            normalized_error["ctx"] = {
                key: str(value) for key, value in context.items()
            }
        normalized_errors.append(normalized_error)

    return JSONResponse(
        status_code=422,
        content={
            "detail": normalized_errors,
            "message": (
                "Request body validation failed. Ensure the top-level body is valid JSON."
            ),
        },
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a simple health indicator for the API."""

    return HealthResponse(status="ok")


@app.post("/reset", response_model=ResetResponse)
def reset_episode(request: ResetRequest) -> ResetResponse:
    """Create a new DriftEnv instance and return its initial observation."""

    global current_episode

    current_episode = DriftEnv(
        episode_type=request.episode_type,
        curriculum_stage=request.curriculum_stage,
        seed=request.seed,
    )
    observation = current_episode.reset()

    return ResetResponse(
        observation=observation,
        info={
            "episode_type": request.episode_type,
            "curriculum_stage": request.curriculum_stage,
            "seed": request.seed,
            "turn": current_episode.turn,
            "done": current_episode.done,
        },
    )


@app.post("/step", response_model=StepResponse)
def step_episode(request: StepRequest) -> StepResponse:
    """Advance the active DriftEnv episode using the raw action text."""

    if current_episode is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

    step_result = current_episode.step(request.action)
    return StepResponse(**step_result)
