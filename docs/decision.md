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