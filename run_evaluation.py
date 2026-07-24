"""
Run this after any dataset run (or once, over the whole accumulated
results.jsonl) to score every row with the LLM judge, RAGAS, and
deterministic checks.
"""

from pathlib import Path
from app.evaluation.batch_evaluator import BatchEvaluator
import app.utils.logger  # noqa: F401

if __name__ == "__main__":
    evaluator = BatchEvaluator()
    evaluator.run(
        input_path=Path("experiments") / "results.jsonl",
        output_path=Path("experiments") / "evaluated_results.jsonl",
    )
    print("Evaluation complete -- see experiments/evaluated_results.jsonl")