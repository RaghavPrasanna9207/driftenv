from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.inference import generate_comparison


HARDCODED_ACTION = (
    '{"action_type":"request_clarification","params":'
    '{"question":"Can you confirm whether the graph changed?"}}'
)


class DummyTokenizer:
    """Minimal tokenizer stub for inference smoke tests."""

    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        del text, return_tensors
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del token_ids, skip_special_tokens
        return HARDCODED_ACTION


class DummyModel:
    """Minimal model stub that always emits the same action."""

    device = "cpu"

    def generate(self, **kwargs) -> torch.Tensor:
        del kwargs
        return torch.tensor([[1, 2, 3, 4, 5]])


def test_generate_comparison_returns_non_empty_turn_summary() -> None:
    """Smoke-test comparison generation before real models are available."""

    tokenizer = DummyTokenizer()
    baseline_model = DummyModel()
    trained_model = DummyModel()

    comparison = generate_comparison(
        episode_type="manager_departure",
        baseline_model=baseline_model,
        trained_model=trained_model,
        tokenizer=tokenizer,
    )

    assert isinstance(comparison, str)
    assert comparison.strip()
    assert "TURN" in comparison.upper()
