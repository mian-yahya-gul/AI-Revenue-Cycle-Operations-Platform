# AI Medical Billing Revenue Cycle Team

A multi agent AI **Operations Platform** for healthcare Revenue Cycle Management (RCM). 
Every insurance claim becomes a
stateful workflow coordinated by **LangGraph** across a team of specialist agents
that genuinely collaborate: they challenge each other's output, send work back for
revision, gate submission behind a compliance review, and pause for a human when
documentation runs out, all through shared state and conditional routing, with
**RAG grounded** payer policy lookups, deterministic **tool calling**, **structured
outputs**, and an **event driven** data layer underneath.

---

## Business Problem

Hospitals and billing companies process thousands of insurance claims every day.
Claims are delayed or denied because of:

- Missing documentation
- Incorrect ICD-10 / CPT coding
- Missing modifiers
- Insurance policy violations (e.g. missing prior authorization)
- Duplicate claims
- Claim validation failures
- Missing physician notes

Each denial costs money and generates rework. Revenue cycle teams need a way to
**catch these problems before submission**, and to manage denied claims
efficiently when they do occur.

## Why Multi Agent Architecture?

A real hospital billing department isn't one person who does everything. It's a
team of specialists: a coder, a documentation reviewer, a compliance officer, a
policy analyst, an appeals specialist, and a manager who coordinates all of them.
Each person is deep in one area and hands off to the next when their part is
done, but they also push back on each other, a coder tells validation "I need
another look at this," compliance tells the manager "this can't go out yet,"
and the manager decides what happens next.

That's the shape this platform mirrors, and it's a deliberate architectural
choice rather than an aesthetic one:

- **A single large LLM call asked to "process this claim" has no natural place
  to disagree with itself.** It either gets coding, validation, and policy
  compliance right in one pass or it doesn't , there's no structural mechanism
  for "wait, send that back." Splitting the work into agents with a real
  feedback edge between them (Coding <-> Validation, in this platform) gives
  the system a place to catch and correct its own mistakes before they become
  a claim denial.
- **Specialization improves the reliability of each individual judgment.** The
  Insurance Policy Agent's only job is grounding answers in retrieved payer
  policy text, it never has to also worry about ICD-10 accuracy. The
  Compliance Agent's only job is reviewing what the others produced, it never
  generates new clinical facts, only judges completeness and consistency. Each
  agent's prompt, schema, and rule logic stays narrow and
  auditable instead of being one sprawling do everything instruction set.
- **A quality gate needs to sit structurally between "work happened" and
  "work gets submitted."** The Compliance Agent occupies exactly that position
  in both graphs, every claim passes through it before the Revenue Cycle
  Manager can mark it ready. That's not something you can bolt onto a single
  monolithic call after the fact; it has to be a real stage in the pipeline.
- **Escalation and pausing need an authority above the agent that flagged the
  problem.** When Compliance blocks a claim, the Revenue Cycle Manager, not
  Compliance itself, decides what actually happens (escalate, queue, or
  proceed). When documentation runs out entirely, the graph itself pauses via
  a LangGraph interrupt rather than any single agent deciding to give up.
  Orchestration is a distinct responsibility from any one specialist's
  judgment, which is exactly why it's its own agent here.


## Solution Overview

This platform treats every claim as a **workflow instance** that moves through a
team of specialized AI agents that read and write shared state rather than
simply executing one after another:

1. Clinical Documentation extracts a structured summary from raw physician notes
   and if there's nothing usable in them at all, the workflow **pauses** and
   waits for a human to supply more.
2. Medical Coding recommends ICD-10 / CPT codes with confidence scores.
3. Claim Validation checks completeness, duplicates, and coding consistency, 
   and if it finds a fixable coding problem, it **sends the claim back to
   Medical Coding for a revision** before proceeding, up to a configured retry
   limit.
4. Insurance Policy checks payer policy, grounded in retrieved policy text via
   RAG, not model memory, for coverage, prior authorization requirements, and
   any conflicting rules across retrieved documents.
5. **Compliance** : reviews everything the first four agents produced as a
   whole, builds an audit trail, and issues the single authoritative
   human review determination, the quality gate every claim must clear.
6. The **Revenue Cycle Manager** : the orchestrator, treats that
   determination as final and routes the claim to **"ready for submission"**
   or **"needs human review,"** with a concrete next step directive either way.

If a claim is later denied, a second workflow takes over: root cause analysis,
a policy re check, the same Compliance quality gate applied to the appeal
basis, appeal letter generation, and a final Revenue Cycle Manager
assessment. routing non appealable denials, or denials Compliance blocks,
straight to a human.

## Architecture

```
+-----------------------------------------------------------------------+
|                         Streamlit Dashboard                           |
|   Dashboard . New Claim . Claim Details . Denied Claims . Policy      |
|   Search . Agent Activity . Revenue Dashboard . Knowledge Base        |
+--------------------------------+---------------------------------------+
                                  |
                     graph/events.py (event-driven entry points)
                      NEW_CLAIM . DOCUMENT_UPDATED . CLAIM_DENIED . ...
                                  |
               +------------------+------------------+
               v                                      v
    Claim Intake Graph (LangGraph)          Denial/Appeal Graph (LangGraph)
    checkpointed - supports pause/resume     shares Insurance Policy,
               |                             Compliance, Revenue Cycle
    +----------+--------------------+        Manager with the intake graph
    | Clinical Doc [pause/resume]   |                   |
    |      |                        |         +---------+---------+
    | Coding <-> Validation (loop)  |         | Denial Analysis    |
    |      |                        |         |      |             |
    | Insurance Policy (RAG)        |         | Insurance Policy    |
    |      |                        |         |      |             |
    | Compliance (quality gate)     |         | Compliance (gate)   |
    |      |                        |         |      |             |
    | Revenue Cycle Manager         |         | Appeal Generation   |
    | (orchestrator, final route)   |         |      |             |
    +----------+--------------------+         | Revenue Cycle Mgr   |
               v                              +---------+-----------+
         SQLite (claims, patients, appeals, agent_logs, events)
               |
               v
    NumPy vector store  <--  rag/ pipeline  <--  sample_data/payer_policies/
```

## Multi-Agent Team

| # | Agent | Responsibility |
|---|-------|-----------------|
| 1 | **Clinical Documentation Agent** | Extracts diagnoses/procedures from physician notes; can pause the whole workflow if nothing is usable |
| 2 | **Medical Coding Agent** | Recommends ICD-10 / CPT codes with reasoning + confidence; revises on Validation's feedback |
| 3 | **Claim Validation Agent** | Checks completeness, duplicates, coding consistency; sends coding back for revision when warranted |
| 4 | **Insurance Policy Agent** | RAG grounded coverage / prior-auth determination, with rationale and conflict detection, the *only* agent that queries the policy knowledge base |
| 5 | **Compliance Agent** | Reviews every prior agent's output as a whole, builds the audit trail, and is the sole authority on mandatory human review |
| 6 | **Denial Analysis Agent** | Root causes a payer denial, classifies it, recommends corrections |
| 7 | **Appeal Generation Agent** | Drafts a formal appeal letter + missing documentation checklist |
| 8 | **Revenue Cycle Manager Agent** | The orchestrator: treats Compliance's verdict as authoritative and issues the final routing decision for every claim in both workflows |


## Agent Collaboration in Practice

These are the three collaboration patterns the codebase actually implements, 
not just conceptually, but as real conditional edges and interrupts in
`graph/graph.py` and `graph/routing.py`:

**1. Coding <-> Validation feedback loop.** If Claim Validation finds the coding
inadequate (missing codes, or confidence below `VALIDATION_CONFIDENCE_THRESHOLD`),
it routes the claim back to Medical Coding with a specific feedback message
instead of just failing the claim. Medical Coding treats that feedback like a
colleague's revision request, clearly flagged fallback where it safely can, and refuses to fabricate a
diagnosis it has no support for. This repeats up to `MAX_CODING_REVISIONS`
times (default 2) before the claim proceeds regardless, at which point
Compliance will flag the unresolved issue for a human.

**2. Insurance Policy -> Compliance -> Revenue Cycle Manager escalation.** When
Insurance Policy determines prior authorization is required, that's not itself
a stop, it's a finding written to shared state. The Compliance Agent reads it
and marks `human_review_required = True` as a hard block. The Revenue Cycle
Manager then treats that flag as authoritative and issues
`next_step = "escalate_to_human_review"` regardless of what the numeric risk
score alone would suggest, mirroring the exact "policy agent flags it,
compliance blocks it, manager escalates it" chain a real RCM team follows.

**3. Documentation pause and resume.** If Clinical Documentation extracts
nothing usable at all, the intake graph pauses via LangGraph's `interrupt()`
checkpointed, so the pause survives across Streamlit reruns, and the claim
sits in `AWAITING_DOCUMENTATION` status. A human supplies additional notes
(from the New Claim page right after submission, or from Claim Details at any
later point), and the workflow resumes from exactly that point: Clinical
Documentation re runs against the corrected notes, and every downstream agent
runs fresh, since coding, validation, policy, compliance, and the final
decision all depend on that summary. If the new notes are *still* insufficient,
it pauses again rather than forcing an empty claim through the rest of the team.

## LangGraph Orchestration

**Workflow 1  Claim Intake** (checkpointed for pause/resume)

```
Claim Received
   -> Clinical Documentation Agent
        |-- nothing extractable -> PAUSE (interrupt) -> await documentation
        |        `-- resumed with new notes -> re-run Clinical Documentation
        `-- proceeds
   -> Medical Coding Agent  <----------------------.
   -> Claim Validation Agent                        | revise (feedback loop,
        |-- coding inadequate & budget remains -----' bounded by
        `-- proceeds                                  MAX_CODING_REVISIONS)
   -> Insurance Policy Agent (RAG)
   -> Compliance Agent (quality gate - reviews everything above)
   -> Revenue Cycle Manager (orchestrator - final decision)
        |-- compliance clear   -> Ready for Submission
        `-- compliance blocked -> Human Review
```

**Workflow 2  Insurance Denial**

```
Insurance Denial
   -> Denial Analysis Agent
        |-- non-appealable -> Human Review
        `-- appealable
             -> Insurance Policy Agent (RAG recheck)
             -> Compliance Agent (quality gate, is there a sound basis to appeal?)
                  |-- blocked -> Human Review
                  `-- cleared
                       -> Appeal Generation Agent
                       -> Revenue Cycle Manager -> Ready for Resubmission
```

Both graphs are implemented with `langgraph.graph.StateGraph` over a shared,
typed state (`graph/state.py`), with conditional routing, loops, and an
interrupt (`graph/routing.py`, `graph/graph.py`) replacing what used to be a
straight ine pipeline. The claim-intake graph is compiled with a
checkpointer (`MemorySaver`, keyed per claim by `thread_id`) specifically so
the documentation pause interrupt can be resumed later via
`Command(resume=...)`  including from a completely different Streamlit
rerun than the one that triggered the pause.

## Event-Driven Workflow

The event driven layer (`graph/events.py`) sits above the graphs: firing an
event persists it and triggers only the relevant graph (or resumes a paused
one) this is the seam where a future message queue (e.g. SQS, Kafka) could
be dropped in without touching agent logic. Events recorded through a claim's
lifecycle include:

`NEW_CLAIM` . `DOCUMENTATION_REQUIRED` . `DOCUMENT_UPDATED` .
`CODING_COMPLETED` . `CODING_REVISION_REQUESTED` . `VALIDATION_FAILED` .
`COMPLIANCE_FAILED` . `CLAIM_READY_FOR_SUBMISSION` . `HUMAN_REVIEW_REQUIRED` .
`CLAIM_DENIED` . `APPEAL_SUBMITTED` . `CLAIM_APPROVED` . `PAYMENT_RECEIVED`

Every side effect around the documentation pause interrupt (`DOCUMENTATION_REQUIRED`,
`DOCUMENT_UPDATED`) is written to be idempotent against LangGraph's replay on resume
semantics, a node that pauses gets re executed from the top on every later resume,
so the event log is checked against durable state before writing, rather than
assuming a node runs exactly once.

## RAG Pipeline

The **Insurance Policy Agent** must never answer from model memory, payer
rules vary by plan and change frequently, and it is deliberately the *only*
agent in the system with access to the policy knowledge base; every other
agent depends on its findings rather than querying RAG themselves.

```
sample_data/payer_policies/*.txt
        |  (8 realistic policy documents: UHC, Aetna, Cigna, BCBS x
        |   prior authorization + coverage criteria)
        v
  rag/loader.py       - loads documents, tags payer metadata
        v
  rag/splitter.py     - RecursiveCharacterTextSplitter, section-aware chunking
        v
  rag/embeddings.py   - OpenAIEmbeddings (live) or a deterministic hashing
        |                embedding (mock mode, fully offline)
        v
  rag/vector_store.py - brute-force cosine similarity NumPy index, persisted
        |                to rag/vector_store_index/ (no compiled deps)
        v
  rag/retriever.py     - similarity search + payer-filtered ranking
```

The retrieved chunks are passed directly into the Insurance Policy Agent's
prompt or used to populate `PolicyReference`,
findings are traceable back to a specific policy document, and any
contradiction across retrieved excerpts is surfaced as `conflicting_rules`
rather than silently resolved.

> **Why not FAISS?** FAISS ships as a compiled binary (SWIG bindings), and its
> wheel availability is inconsistent across Python versions, OS/architecture
> combinations, and NumPy ABI versions - exactly the kind of "works on my
> machine" friction a portfolio project should avoid. For a corpus this size
> (a handful of payer policy documents, a few hundred chunks), brute force
> cosine similarity over a NumPy matrix is effectively instant and needs
> nothing beyond NumPy, which the rest of the stack already depends on. The
> vector store interface (`build_vector_store` / `load_vector_store` /
> `similarity_search`) is intentionally shaped like LangChain's, so swapping
> in FAISS, Chroma, or a hosted store later only touches `rag/vector_store.py`.

## Tools

Reusable, independently callable tools (`tools/`), each exposed as a LangChain
`@tool` so an LLM-driven agent can invoke them directly:

- `claim_lookup_tool`, `patient_lookup_tool` - data access
- `policy_search_tool` - the RAG interface (Insurance Policy Agent only)
- `icd_lookup_tool`, `cpt_lookup_tool` - mock coding reference lookups
- `claim_validation_tool` - rule-based completeness/duplicate/consistency checks
- `appeal_generator_tool` - appeal letter + checklist generation
- `report_generator_tool` - Markdown claim / dashboard reports
