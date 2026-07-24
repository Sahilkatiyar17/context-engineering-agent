# Decisions Log

Short records of *why* something was chosen, not just what. Written as
I go, so I don't have to reconstruct reasoning later for the write-up.

---

## Phase 2 — Short-term memory strategy

**Decision:** Use trim + threshold-based summarization (summarize every
N messages, not every turn) as the *only* short-term memory strategy.
Did not implement/compare last-5, last-10, or full-history variants.

**Why:**
- `full_history` grows tokens unboundedly as a conversation gets longer —
  not viable past a certain length.
- `last_N` silently drops information. If a later question depends on
  something said many turns ago, past N, it's just gone with no warning.
- Trim + summarize solves both at once: bounded token growth (summary
  stays short regardless of conversation length) while preserving older
  context in compressed form, rather than dropping it outright.
- Summarizing every turn (rather than every N) was rejected early —
  it doubles LLM calls per turn for little benefit when nothing new
  has happened since the last summary.

**Not explored:** Not experimentally comparing this against last-N/full-history
is a deliberate scope decision, not an oversight — this project's actual
experimentation budget is going into Phase 4 (context pipeline), where the
open questions are less settled and more worth measuring directly.

**Implementation:** `app/agents/nodes.py` (`summarize_conversation`,
`should_summarize`), threshold configurable via `summarize_after_n_messages`
on `ResearchAgentGraph`.

---

## Phase 2 — Persistence: SQLite over InMemorySaver

**Decision:** Use `SqliteSaver` (file-backed) instead of `InMemorySaver`
for the LangGraph checkpointer.

**Why:** `InMemorySaver` only persists for the lifetime of a single
Python process. Since `thread_id` is just a lookup key into whatever
checkpointer instance exists, a fresh `python main.py` run creates a
brand-new empty store — same thread_id, no memory. Confirmed this
directly: identical thread_id across two separate process runs returned
"beginning of conversation" with InMemorySaver, and correctly recalled
prior context after switching to SqliteSaver.

**Implementation:** `app/agents/graph.py`, checkpoint file at
`data/checkpoints.db` (gitignored — local run state, not source).

---

## Phase 2 — thread_id: manual, not auto-generated

**Decision:** `THREAD_ID` is a plain string constant in `main.py`, set
and changed by hand. Considered auto-timestamping a fresh thread_id
per run, rejected.

**Why:** Wanted explicit control over when a conversation is "new" vs.
"continuing" — sometimes intentionally re-running the same session to
extend it, sometimes starting fresh. Auto-generation removes that choice.

## Phase 3 — Long-term memory (Mem0 Cloud)

**Decision:** Mem0 Cloud (free tier), no self-hosted setup.

**Why:** Free tier (10K memories/1K retrieval calls/month) fully covers
this project's needs. Self-hosting only changes where the vector store
lives, not the extraction/retrieval algorithm itself -- no meaningful
capability trade-off for a solo project at this scale, and would add
ops overhead for no benefit right now.

**Scoping:** long-term memory keyed by user_id (persists across all
threads), separate from short-term memory keyed by thread_id (dies with
the conversation). Verified both: same-user recall across a brand-new
thread works; a different user_id in a fresh thread has no access to
those facts (isolation confirmed).

**Implementation:** app/memory/long_term.py (Mem0Client wrapper),
graph flow: recall_memory -> search -> chat -> remember -> [summarize
conditional, unchanged from Phase 2].

## Phase 4 — Filter stage caught a bad search result entirely

**Observation:** On the final synthesis question ("Summarize everything we've
discussed about PPO, and remind me how I like my answers formatted"), the
agent still triggered a web search -- the fixed pipeline searches every turn,
regardless of whether the question needs it. For this question, Tavily
returned irrelevant results (in one run, unrelated Federal Register healthcare
regulation text -- nothing to do with PPO).

In the raw/baseline pipeline, that irrelevant content still got pushed into
the prompt, costing tokens for zero benefit. In the fully-engineered pipeline,
the relevance filter (RelevanceScoreFilter, min_score=0.3) scored every one
of those search results below threshold and dropped all of them -- reducing
that turn's search_results from 5 down to 0.

**Effect:** With no irrelevant search content competing for space, the answer
was generated almost entirely from short-term memory (the running conversation
summary) and long-term memory (Mem0-retrieved facts, e.g. the stated bullet-
point preference) -- exactly the sources that actually mattered for a
synthesis question. Fewer tokens spent, and a cleaner, better-grounded answer,
because nothing irrelevant was diluting the context.

**Why this matters:** This is filtering doing its job correctly under a real
failure condition, not just trimming already-good results. It's also a
concrete illustration of why a context planner (deciding whether to search
at all, per question) is the natural next step -- the filter caught the
damage *after* an unnecessary search already happened; a planner could avoid
triggering that search in the first place, saving the Tavily call itself,
not just its output.


## Phase 6 — Evaluation dependency chain fragility

Hit significant version-compatibility issues chaining ragas + langchain_community
+ langchain_core + langchain_groq together -- each library's latest version
assumed different versions of the others. Resolved by pinning ragas<0.4 (0.4.x
has a hardcoded broken import to a relocated langchain_community submodule)
and patching that file directly. Also hit two Groq model deprecations mid-project
(deepseek-r1-distill-llama-70b, and llama-3.3-70b-versatile is deprecated with
an 08/16/26 shutdown -- migration still pending). Lesson: freeze requirements.txt
immediately after any environment that's confirmed working, don't wait.