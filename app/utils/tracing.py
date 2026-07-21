# app/utils/tracing.py

import os
import logging

from app.utils.config import get_settings

logger = logging.getLogger(__name__)


class TracingConfigurator:
    """
    Bridges our typed Settings into the env vars LangChain's tracer
    actually reads. Called once, at app startup, before any graph runs.
    """

    @staticmethod
    def configure() -> None:
        settings = get_settings()

        if not settings.langsmith_tracing:
            logger.info("LangSmith tracing disabled")
            return

        if not settings.langsmith_api_key:
            logger.warning("LANGSMITH_TRACING is true but LANGSMITH_API_KEY is missing -- skipping tracing")
            return

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

        logger.info(f"LangSmith tracing enabled -- project={settings.langsmith_project}")
        
    