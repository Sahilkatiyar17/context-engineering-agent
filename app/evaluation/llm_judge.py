import sys, json, logging
from app.utils.llm_client import LLMClient
from app.utils.exception import AgentException

logger = logging.getLogger(__name__)

JUDGE_MODEL = "openai/gpt-oss-120b"  # different from the generation model, on purpose

RUBRIC_PROMPT = """You are a strict evaluator of AI assistant answers. Score the answer below on these dimensions, each from 1 (poor) to 5 (excellent):

- relevance: does the answer address what was actually asked?
- correctness: is the answer factually accurate?
- completeness: does it cover all parts of the question?
{groundedness_line}

Question: {question}

Answer: {answer}
{context_block}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"relevance": <1-5>, "correctness": <1-5>, "completeness": <1-5>{groundedness_key}, "justification": "<one sentence explaining the scores>"}}
"""


class LLMJudge:
    """
    Scores answers on a fixed rubric using a model DIFFERENT from the one
    that generated the answers, to avoid self-evaluation bias.
    Groundedness is only scored when retrieved context is available --
    otherwise it's omitted, not defaulted to a number.
    """

    def __init__(self):
        self.llm = LLMClient(model=JUDGE_MODEL, temperature=0.0)

    def evaluate(self, question: str, answer: str, context: str = "") -> dict:
        try:
            has_context = bool(context.strip())
            prompt = RUBRIC_PROMPT.format(
                question=question,
                answer=answer,
                context_block=f"\nRetrieved context:\n{context}" if has_context else "",
                groundedness_line="- groundedness: is every claim traceable to the retrieved context?" if has_context else "",
                groundedness_key=', "groundedness": <1-5>' if has_context else "",
            )
            raw = self.llm.invoke([{"role": "user", "content": prompt}])
            scores = self._parse_json(raw)

            if not has_context:
                scores["groundedness"] = "context_unavailable"

            return scores
        except Exception as e:
            raise AgentException(e, sys) from e

    def _parse_json(self, raw: str) -> dict:
        """Reasoning models sometimes prepend thinking text -- extract the last {...} block."""
        start = raw.rfind("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            logger.warning(f"Judge did not return parseable JSON: {raw[:200]}")
            return {"relevance": None, "correctness": None, "completeness": None, "justification": "parse_error"}
        return json.loads(raw[start:end + 1])