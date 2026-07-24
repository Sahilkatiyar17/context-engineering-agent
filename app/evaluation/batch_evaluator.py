import sys, json, logging
from pathlib import Path
import time

from app.evaluation.llm_judge import LLMJudge
from app.evaluation.ragas_evaluator import RagasEvaluator
from app.evaluation.deterministic_checks import DeterministicChecks
from app.utils.exception import AgentException

logger = logging.getLogger(__name__)

BULLET_APPLICABLE_NOTES = {"q3", "q4", "q5", "q6"}  # bullet preference stated in q3, only fair to check from there on


class BatchEvaluator:
    """
    Reads a completed results.jsonl, scores every row with the LLM judge,
    RAGAS, and deterministic checks, writes evaluated_results.jsonl.
    Runs once per full dataset run (batch), not per-turn -- per the
    latency/cost tradeoff decided earlier.

    A failure on one row is logged and marked, not raised -- one bad
    row (e.g. a judge parse error, a RAGAS API hiccup) should not kill
    evaluation for every other row in the file.
    """

    def __init__(self):
        self.judge = LLMJudge()
        self.ragas = RagasEvaluator()
        self.checks = DeterministicChecks()

    

    def run(self, input_path: Path, output_path: Path) -> None:
        try:
            rows = self._read_jsonl(input_path)
            logger.info(f"Evaluating {len(rows)} rows from {input_path}")

            evaluated_rows = []
            for i, row in enumerate(rows):
                evaluated_rows.append(self._evaluate_row(row))
                if i < len(rows) - 1:
                    time.sleep(4)  # throttle to stay under Groq free-tier RPM

            self._write_jsonl(output_path, evaluated_rows)
            logger.info(f"Wrote {len(evaluated_rows)} evaluated rows to {output_path}")
        except Exception as e:
            raise AgentException(e, sys) from e

    def _evaluate_row(self, row: dict) -> dict:
        question = row.get("question", "")
        answer = row.get("answer", "")
        context = "\n\n".join(filter(None, [row.get("search_context", ""), row.get("memory_context", "")]))

        try:
            judge_scores = self.judge.evaluate(question, answer, context)
        except Exception as e:
            logger.error(f"Judge failed for run_id={row.get('run_id')}: {e}")
            judge_scores = {"error": str(e)}

        try:
            ragas_scores = self.ragas.evaluate(question, answer, context)
        except Exception as e:
            logger.error(f"RAGAS failed for run_id={row.get('run_id')}: {e}")
            ragas_scores = {"error": str(e)}

        try:
            det_scores = self.checks.evaluate(answer, context)
            if row.get("notes") not in BULLET_APPLICABLE_NOTES:
                det_scores["uses_bullet_format"] = "not_applicable"
        except Exception as e:
            logger.error(f"Deterministic checks failed for run_id={row.get('run_id')}: {e}")
            det_scores = {"error": str(e)}

        return {**row, "judge": judge_scores, "ragas": ragas_scores, "deterministic": det_scores}

    def _read_jsonl(self, path: Path) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")