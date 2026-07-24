import sys, time, logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage

from app.agents.state import AgentState
from app.search.web_search import WebSearchClient
from app.utils.llm_client import LLMClient
from app.memory.long_term import LongTermMemory
from app.utils.exception import AgentException
from app.context.filters import ContextFilter
from app.context.ranking import ContextRanker
from app.context.deduplication import ContextDeduplicator
from app.context.compression import ContextCompressor


logger = logging.getLogger(__name__)


class AgentNodes:
    def __init__(self, search_client: WebSearchClient, llm_client: LLMClient,
                 long_term_memory: LongTermMemory, user_id: str,
                 summarize_after_n_messages: int = 6, keep_last_n_verbatim: int = 2,context_filter: ContextFilter = None,context_ranker: ContextRanker = None,
                 context_deduplicator: ContextDeduplicator = None,context_compressor: ContextCompressor = None):
        self.search_client = search_client
        self.llm_client = llm_client
        self.long_term_memory = long_term_memory
        self.user_id = user_id
        self.summarize_after_n_messages = summarize_after_n_messages
        self.keep_last_n_verbatim = keep_last_n_verbatim
        self.context_filter = context_filter
        self.context_ranker = context_ranker
        self.context_deduplicator = context_deduplicator
        self.context_compressor = context_compressor

    def recall_memory_node(self, state: AgentState) -> dict:
        try:
            memories = self.long_term_memory.recall(self.user_id, state.latest_question)
            return {"long_term_memories": memories}
        except Exception as e:
            raise AgentException(e, sys) from e

    def search_node(self, state: AgentState) -> dict:
        try:
            results = self.search_client.search(state.latest_question)
            return {"search_results": [r.model_dump() for r in results]}
        except Exception as e:
            raise AgentException(e, sys) from e

    def chat_node(self, state: AgentState) -> dict:
        try:
            messages = self._build_prompt(state)
            start = time.perf_counter()
            result = self.llm_client.invoke_with_usage(messages)
            elapsed = time.perf_counter() - start
            return {
                "messages": [AIMessage(content=result["text"])],
                "answer": result["text"],
                "total_tokens": result["total_tokens"],
                "latency_seconds": round(elapsed, 3),
            }
        except Exception as e:
            raise AgentException(e, sys) from e
    
    
    
    def filter_context_node(self, state: AgentState) -> dict:
        try:
            before = len(state.search_results)
            filtered = self.context_filter.apply(state.search_results)
            logger.info(f"[filter_context] {before} -> {len(filtered)} search results")
            return {"search_results": filtered}
        except Exception as e:
            raise AgentException(e, sys) from e
    
    

    def _build_prompt(self, state: AgentState) -> list:
        prompt = []
        if state.long_term_memories:
            facts = "\n".join(f"- {m}" for m in state.long_term_memories)
            prompt.append(SystemMessage(content=f"Known facts about this user:\n{facts}"))
        if state.summary:
            prompt.append(SystemMessage(content=f"Conversation summary so far:\n{state.summary}"))
        if state.search_results:
            sources = "\n\n".join(f"Source: {r['title']} ({r['url']})\n{r['content']}" for r in state.search_results)
            prompt.append(SystemMessage(content=f"Relevant sources:\n{sources}"))
        prompt.extend(state.messages)
        return prompt

    def remember_node(self, state: AgentState) -> dict:
        try:
            self.long_term_memory.remember(self.user_id, state.latest_question, state.answer)
            return {}
        except Exception as e:
            raise AgentException(e, sys) from e

    def summarize_conversation(self, state: AgentState) -> dict:
        try:
            prompt = (
                f"Existing summary:\n{state.summary}\n\nExtend the summary using the new conversation above."
                if state.summary else "Summarize the conversation above."
            )
            messages_for_summary = state.messages + [HumanMessage(content=prompt)]
            new_summary = self.llm_client.invoke(messages_for_summary)
            messages_to_delete = state.messages[:-self.keep_last_n_verbatim]
            return {
                "summary": new_summary,
                "messages": [RemoveMessage(id=m.id) for m in messages_to_delete],
            }
        except Exception as e:
            raise AgentException(e, sys) from e

    def should_summarize(self, state: AgentState) -> bool:
        return len(state.messages) > self.summarize_after_n_messages
    
    def rank_context_node(self, state: AgentState) -> dict:
        try:
            ranked = self.context_ranker.apply(state.latest_question, state.search_results)
            return {"search_results": ranked}
        except Exception as e:
            raise AgentException(e, sys) from e
    
    def dedupe_context_node(self, state: AgentState) -> dict:
        try:
            before = len(state.search_results)
            deduped = self.context_deduplicator.apply(state.search_results)
            logger.info(f"[dedupe_context] {before} -> {len(deduped)} search results")
            return {"search_results": deduped}
        except Exception as e:
            raise AgentException(e, sys) from e
        
    def compress_context_node(self, state: AgentState) -> dict:
        try:
            before = len(state.search_results)
            compressed = self.context_compressor.apply(state.search_results)
            logger.info(f"[compress_context] {before} results -> compressed block")
            return {"search_results": compressed}
        except Exception as e:
            raise AgentException(e, sys) from e