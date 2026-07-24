import sys, json, logging
from pathlib import Path
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import uuid

from app.agents.state import AgentState
from app.utils.exception import AgentException

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("experiments")
RESULTS_FILE = EXPERIMENTS_DIR / "results.jsonl"


class ExperimentRecord(BaseModel):
    timestamp: str
    experiment_id: str
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy: str
    question: str
    answer: str
    total_tokens: int | None
    latency_seconds: float | None
    search_context: str = ""     # NEW -- raw retrieved search content, for faithfulness checks later
    memory_context: str = ""     # NEW -- raw retrieved long-term memories, for faithfulness checks later
    notes: str = ""


class ExperimentLogger:
    def __init__(self, results_file: Path = RESULTS_FILE):
        self.results_file = results_file
        self.results_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, experiment_id: str, strategy: str, state: AgentState, notes: str = "") -> None:
        try:
            record = ExperimentRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                experiment_id=experiment_id,
                strategy=strategy,
                question=state.latest_question,
                answer=state.answer or "",
                total_tokens=state.total_tokens,
                latency_seconds=state.latency_seconds,
                search_context=self._extract_search_context(state),
                memory_context="\n".join(state.long_term_memories),
                notes=notes,
            )
            with open(self.results_file, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
            logger.info(f"Logged experiment: {experiment_id} / {strategy}")
        except Exception as e:
            raise AgentException(e, sys) from e

    def _extract_search_context(self, state: AgentState) -> str:
        return "\n\n".join(self._as_str(r.get("content", "")) for r in state.search_results)

    def _as_str(self, content) -> str:
        if isinstance(content, list):
            return "\n".join(str(c) for c in content)
        return str(content)