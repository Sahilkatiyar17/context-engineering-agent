import sys, asyncio, logging
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI

from app.utils.config import get_settings
from app.utils.exception import AgentException

logger = logging.getLogger(__name__)

RAGAS_LLM_MODEL = "meta/llama-3.1-70b-instruct"  # NVIDIA NIM catalog name -- separate provider from Groq


class RagasEvaluator:
    def __init__(self):
        settings = get_settings()
        llm = ChatOpenAI(
            model=RAGAS_LLM_MODEL,
            api_key=settings.nvidia_api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=0.0,
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", google_api_key=settings.gemini_api_key  # was text-embedding-004, deprecated Jan 2026
        )
        self.ragas_llm = LangchainLLMWrapper(llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
        self.faithfulness = Faithfulness(llm=self.ragas_llm)
        self.answer_relevancy = ResponseRelevancy(llm=self.ragas_llm, embeddings=self.ragas_embeddings)

    def evaluate(self, question: str, answer: str, context: str) -> dict:
        try:
            if not context.strip():
                return {"faithfulness": "context_unavailable", "answer_relevancy": "context_unavailable"}

            sample = SingleTurnSample(user_input=question, response=answer, retrieved_contexts=[context])

            faithfulness_score = asyncio.run(self.faithfulness.single_turn_ascore(sample))
            relevancy_score = asyncio.run(self.answer_relevancy.single_turn_ascore(sample))

            return {
                "faithfulness": round(faithfulness_score, 3),
                "answer_relevancy": round(relevancy_score, 3),
            }
        except Exception as e:
            raise AgentException(e, sys) from e