from app.utils.llm_client import LLMClient
from langchain_core.messages import HumanMessage

client = LLMClient()
print(client.invoke([HumanMessage(content="Reply with exactly: setup ok")]))



from app.search.web_search import WebSearchClient

client = WebSearchClient(max_results=3)
results = client.search("What is PPO in reinforcement learning?")
for r in results:
    print(r.title, "-", r.url)