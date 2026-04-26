"""CLI wrapper for running DriftEnv inference comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from training.inference import generate_comparison


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for baseline/trained model comparison."""

    parser = argparse.ArgumentParser(description="Compare baseline and trained models in DriftEnv.")
    parser.add_argument(
        "--episode-type",
        default="manager_departure",
        help="Episode type to run during comparison.",
    )
    parser.add_argument(
        "--baseline-model",
        required=True,
        help="Hugging Face model ID or local path for the baseline model.",
    )
    parser.add_argument(
        "--trained-model",
        required=True,
        help="Hugging Face model ID or local path for the trained model.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/eval/inference_comparison.txt",
        help="Path to write the side-by-side textual comparison report.",
    )
    return parser.parse_args()


def _load_model_and_tokenizer(model_name_or_path: str):
    """Load a causal LM and tokenizer from local path or hub ID."""

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    return model, tokenizer


def main() -> None:
    """Run one baseline-vs-trained comparison and persist the report."""

    args = parse_args()
    baseline_model, tokenizer = _load_model_and_tokenizer(args.baseline_model)
    trained_model, _trained_tokenizer = _load_model_and_tokenizer(args.trained_model)

    comparison = generate_comparison(
        episode_type=args.episode_type,
        baseline_model=baseline_model,
        trained_model=trained_model,
        tokenizer=tokenizer,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(comparison, encoding="utf-8")
    print(f"Wrote comparison report to: {output_path}")


if __name__ == "__main__":
    main()
