import sys, logging
from langchain_google_genai import ChatGoogleGenerativeAI
from app.utils.config import get_settings
from app.utils.exception import AgentException

logger = logging.getLogger(__name__)


class CompressionLLMClient:
    """Small, cheap, separate-provider LLM used ONLY for compressing context -- never for answering."""

    def __init__(self, model: str = "gemini-3.5-flash-lite", temperature: float = 0.0):
        settings = get_settings()
        self.model_name = model
        self._llm = self._build_llm(settings.gemini_api_key, temperature)

    def _build_llm(self, api_key: str, temperature: float) -> ChatGoogleGenerativeAI:
        try:
            logger.info(f"Initializing compression LLM: {self.model_name}")
            return ChatGoogleGenerativeAI(model=self.model_name, google_api_key=api_key, temperature=temperature)
        except Exception as e:
            raise AgentException(e, sys) from e

    def invoke(self, prompt: str) -> str:
        try:
            response = self._llm.invoke(prompt)
            return self._flatten(response.content)
        except Exception as e:
            raise AgentException(e, sys) from e

    def _flatten(self, content) -> str:
        """Gemini can return content as a string OR a list of content parts -- always normalize to one string."""
        if isinstance(content, list):
            return "\n".join(
                part.get("text", str(part)) if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)