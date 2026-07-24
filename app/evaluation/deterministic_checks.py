import re
import logging

logger = logging.getLogger(__name__)


class DeterministicChecks:
    """
    Cheap, non-LLM checks -- instant, free, 100% reproducible.
    Meant as a sanity cross-check alongside the LLM judge and RAGAS,
    not a replacement for either.
    """

    def groundedness_overlap(self, answer: str, context: str) -> float | str:
        """
        Crude proxy for groundedness: what fraction of the answer's
        significant words also appear in the retrieved context.
        Not a substitute for RAGAS faithfulness -- just an independent,
        free cross-check that doesn't rely on any LLM's judgment.
        """
        if not context.strip():
            return "context_unavailable"

        answer_words = self._significant_words(answer)
        context_words = self._significant_words(context)

        if not answer_words:
            return 0.0

        overlap = answer_words & context_words
        return round(len(overlap) / len(answer_words), 3)

    def _significant_words(self, text: str) -> set:
        """Lowercased words, 4+ chars, to skip trivial overlap from 'the', 'is', 'and', etc."""
        words = re.findall(r"[a-zA-Z]+", text.lower())
        return {w for w in words if len(w) >= 4}

    def uses_bullet_format(self, answer: str) -> bool:
        """
        Checks the answer's stated formatting preference from the dataset
        (q3: 'concise, bullet-point answers'). Only meaningful for
        questions asked AFTER that preference was stated in a run.
        """
        bullet_patterns = [r"^\s*[-*•]\s", r"^\s*\d+\.\s"]
        lines = answer.strip().split("\n")
        bullet_lines = sum(1 for line in lines if any(re.match(p, line) for p in bullet_patterns))
        return bullet_lines >= 2  # at least 2 bullet-style lines to count as "used bullets"

    def evaluate(self, answer: str, context: str) -> dict:
        return {
            "groundedness_overlap": self.groundedness_overlap(answer, context),
            "uses_bullet_format": self.uses_bullet_format(answer),
        }