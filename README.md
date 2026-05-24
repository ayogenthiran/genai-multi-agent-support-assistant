# GenAI Multi-Agent Customer Support Assistant

A Python reference implementation of an AI-powered customer support assistant for support executives. The system combines:

- **LangGraph** for multi-agent workflow orchestration (supervisor + specialist agents)
- An **MCP-style tool layer** that exposes a small, fixed set of backend tools to the agents (also runnable as a standalone MCP stdio server)
- **SQLite** as the structured data store, queried only through **predefined, parameterized SQL tools** for customer profile and support ticket lookup
- **ChromaDB** for retrieval-augmented Q&A over uploaded / generated policy PDFs
- **Streamlit** as the chat UI

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Assignment Objective](#assignment-objective)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [MCP Server Role](#mcp-server-role)
- [Multi-Agent Workflow](#multi-agent-workflow)
- [Agents](#agents)
- [Tech Stack](#tech-stack)
- [Folder Structure](#folder-structure)
- [Setup Instructions](#setup-instructions)
- [Environment Variables](#environment-variables)
- [Create the Dummy Database](#create-the-dummy-database)
- [Run the Streamlit App](#run-the-streamlit-app)
- [Sample Questions](#sample-questions)
- [Demo Video](#demo-video)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Project Overview

This project is a **demo-scale multi-agent GenAI application** that helps customer support teams answer questions about a fixed, narrow set of scenarios:

- **Customer accounts** — profile lookup, full support ticket history, open tickets, refund-related tickets, and high-priority open tickets
- **Company policies** — refunds, warranties, shipping, cancellations (from indexed PDFs)
- **Combined scenarios** — e.g., *"Can Ema Johnson get a refund based on her support history and the refund policy?"*

Three named demo customers are seeded with deterministic IDs (1=Ema Johnson, 2=Daniel Smith, 3=Priya Patel) so the SQL agent can reliably answer questions for more than one customer.

The **LangGraph Supervisor Agent** routes each question to the right specialist. Specialists fetch data through an **MCP-style tool layer**, and the **Response Synthesis Agent** produces a single, executive-ready reply.

The SQL Customer Agent uses **predefined, parameterized SQL tools** to retrieve customer profiles and support tickets from SQLite. It does **not** generate SQL from natural language, and it does **not** answer ad-hoc analytics, orders, subscriptions, or refunds questions — only the five lookups listed in the [SQL Customer Agent](#sql-customer-agent) section.

---

## Problem Statement

Customer support teams often work across two disconnected information sources:

1. **Structured data** — customer profiles and support tickets stored in a database
2. **Unstructured knowledge** — refund, warranty, shipping, and cancellation policies stored in PDF documents

Switching between SQL dashboards and policy PDFs is slow and error-prone. Support executives need one interface that understands their question, retrieves the right data, and returns a clear, grounded answer.

---

## Assignment Objective

This project was built to demonstrate:

| Goal | Implementation |
|------|----------------|
| Multi-agent orchestration | **LangGraph** state graph with supervisor routing |
| Tool modularity | **MCP-style tool layer** (in-process registry + optional stdio MCP server) |
| Structured data access | **SQLite** + SQLAlchemy, called through predefined parameterized SQL tools |
| Policy knowledge retrieval | **ChromaDB** vector store with OpenAI embeddings over uploaded / generated PDFs |
| User interface | **Streamlit** chat UI with PDF upload support |
| LLM integration | **LangChain** + OpenAI for classification, RAG, and synthesis |

---

## Key Features

- **LangGraph orchestration** — A compiled state graph routes each query through a supervisor, optional specialist agents, and a final response node
- **Intelligent routing** — Supervisor classifies queries as SQL, RAG, both, or general
- **Predefined SQL lookups** — Parameterized queries for customer profile, full ticket history, open tickets, refund-related tickets, and high-priority open tickets (no LLM-generated SQL, no ad-hoc analytics)
- **Policy Q&A with ChromaDB** — Retrieval-augmented generation over uploaded / generated policy PDFs
- **Hybrid answers** — Combines customer ticket data and policy context in one response
- **PDF ingestion** — Upload and index new policy documents from the Streamlit sidebar
- **Source attribution** — Final answers state whether they are based on customer data, policy documents, or both
- **Optional standalone MCP server** — The same eight tools can be exposed over stdio via FastMCP for external MCP hosts; not required for the Streamlit demo, which uses the same tools in-process

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI (app.py)                           │
│              Chat interface · Example questions · PDF upload            │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ user query
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   LangGraph Orchestration Layer                         │
│                      (src/graph/workflow.py)                            │
│                                                                         │
│   START ──► Supervisor ──► [conditional routing]                      │
│                  │                                                      │
│                  ├── sql ──────► SQL Customer Agent                     │
│                  ├── rag ──────► Policy RAG Agent                       │
│                  ├── both ─────► SQL Agent ──► RAG Agent                │
│                  └── general ──► (skip specialists)                     │
│                                         │                               │
│                                         ▼                               │
│                              Response Synthesis Agent                   │
│                                         │                               │
│                                         ▼                               │
│                                       END                               │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ tool calls
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              MCP Tool-Access Layer (src/mcp_server/server.py)           │
│                                                                         │
│   customer_profile_lookup · customer_ticket_lookup · open_ticket_lookup │
│   refund_ticket_lookup · high_priority_open_ticket_lookup               │
│   policy_document_search · policy_question_answer · pdf_ingestion       │
└───────────────┬─────────────────────────────────────┬───────────────────┘
                │                                     │
                ▼                                     ▼
┌───────────────────────────────┐   ┌─────────────────────────────────────┐
│   SQLite (data/customers.db)  │   │  ChromaDB (data/chroma/)            │
│   customers · support_tickets │   │  embedded policy document chunks    │
└───────────────────────────────┘   └─────────────────────────────────────┘
```

---

## MCP Server Role

The project ships an **MCP-style tool layer** (`src/mcp_server/server.py`) — a clean boundary between the LangGraph agents and backend capabilities. It is intentionally lightweight: a name-based Python registry plus an optional FastMCP stdio server that exposes the same tools to external MCP hosts.

**Why it matters:**

- **Modularity** — Agents call tools by name instead of importing database or vector-store code directly
- **Reusability** — The same eight tools are available via Python callables, LangChain `@tool` wrappers, and an optional FastMCP stdio server
- **Separation of concerns** — SQL logic lives in `src/tools/sql_tools.py`; RAG logic lives in `src/rag/`; the MCP layer only exposes a stable interface
- **Demo-friendly** — The Streamlit app calls tools through the in-process registry, so you do **not** need to start a separate MCP process to run the demo

**Registered tools:**

| Tool | Purpose |
|------|---------|
| `customer_profile_lookup` | Look up a customer profile by name |
| `customer_ticket_lookup` | List all support tickets for a customer by name |
| `open_ticket_lookup` | List open support tickets for a customer by name |
| `refund_ticket_lookup` | List refund-related support tickets for a customer by name |
| `high_priority_open_ticket_lookup` | List High/Critical open tickets, optionally for one customer |
| `policy_document_search` | Semantic search over indexed policy chunks |
| `policy_question_answer` | Answer a policy question with RAG |
| `pdf_ingestion` | Index a policy PDF into ChromaDB |

Optionally run the MCP server standalone over stdio for use by external MCP hosts (not required for the Streamlit demo):

```bash
python -m src.mcp_server.server
```

This command starts a FastMCP server (`mcp.server.fastmcp.FastMCP`) that registers the same eight tools listed above.

---

## Agent Design

This project uses a LangGraph-based supervisor workflow instead of a single monolithic agent.

- The Supervisor Agent classifies the user query and routes it to the correct specialist agent.
- The SQL Customer Agent retrieves structured customer profile and support ticket data from SQLite through MCP-style tools.
- The Policy RAG Agent retrieves relevant policy context from ChromaDB using LLM-based query rewriting and source-aware retrieval.
- The Response Synthesis Agent combines SQL and RAG outputs into a clear final response for the support executive.

This design was chosen over a fully open-ended ReAct agent to improve reliability, testability, and demo stability.

**Implementation notes:**

- **Prompts:** centralized under `src/prompts/system_prompts.py`
- **Agent type:** LangGraph supervisor-router workflow
- **Tool use:** MCP-style tool-calling agents
- **RAG:** LLM query rewrite + Chroma retrieval + grounded answer generation
- **Avoided:** one big ReAct agent, LLM-generated raw SQL, and hardcoded prompts scattered across files

---

## Multi-Agent Workflow

1. **User submits a question** in the Streamlit chat UI.
2. **Supervisor Agent** analyzes intent using keyword heuristics and optional LLM classification.
3. **Conditional routing** sends the query to zero, one, or two specialist agents:
   - `sql` → SQL Customer Agent only
   - `rag` → Policy RAG Agent only
   - `both` → SQL Agent, then RAG Agent (sequential)
   - `general` → skip specialists, go straight to synthesis
4. **Specialist agents** call MCP tools and write results into shared graph state (`sql_result`, `rag_context`).
5. **Response Synthesis Agent** merges all context into one professional, source-attributed reply.
6. **Streamlit displays** the final answer and which agent(s) were used.

Shared state is defined in `SupportState` (`src/graph/workflow.py`) and passed between nodes by LangGraph.

---

## Agents

### LangGraph Supervisor Agent

**File:** `src/agents/supervisor.py`

The entry point of the graph. It reads the user query and returns a routing label: `sql`, `rag`, `both`, or `general`.

- Uses keyword matching for customer names, ticket terms, and policy terms
- Falls back to an OpenAI LLM when heuristics are inconclusive
- Does not fetch data itself — only decides which specialist(s) to invoke

### SQL Customer Agent

**File:** `src/agents/sql_agent.py`

Handles questions about customer accounts and support tickets using a small,
fixed set of **predefined, parameterized SQL lookups** over the `customers`
and `support_tickets` tables. The agent does **not** generate SQL from
natural language; it picks one of the lookups below based on simple intent
rules and binds the customer name as a query parameter.

- Extracts customer names from natural language (heuristics + optional LLM)
- If a customer name is required but missing, returns a helpful prompt asking for it
- Routes the query to one of these MCP-style tools:
  - `customer_profile_lookup` + `customer_ticket_lookup` — profile and full ticket history
  - `open_ticket_lookup` — open support tickets for a customer
  - `refund_ticket_lookup` — refund-related support tickets for a customer
  - `high_priority_open_ticket_lookup` — High/Critical open tickets (optionally for one customer)
- Returns structured JSON (`sql_result`) for downstream synthesis
- Backed by **SQLite** via SQLAlchemy (`data/customers.db`)

**Out of scope for this agent:**

- Arbitrary natural-language → SQL generation
- General-purpose SQL analytics, ad-hoc reporting, or aggregations
- Queries against the `orders`, `subscriptions`, or `refunds` tables (these tables exist in the seed schema for context but are **not** exposed via tools in this demo)

### Policy RAG Agent

**File:** `src/agents/rag_agent.py`

Answers policy questions using company policy PDFs that have been uploaded
through the Streamlit sidebar (or generated locally with
`python -m src.create_policy_pdfs`) and indexed into **ChromaDB**.

- Calls MCP-style tools: `policy_question_answer` and `policy_document_search`
- Retrieves relevant chunks from **ChromaDB** using OpenAI embeddings
- Returns grounded policy context (`rag_context`) with supporting excerpts (file name + page number)
- Returns a clear "no documents indexed" message when ChromaDB is empty

### RAG Strategy

Policy retrieval is implemented in `src/rag/` and exposed to agents through MCP-style tools in `src/tools/document_tools.py`:

1. **PDF extraction** — Policy PDFs are read page by page with **PyMuPDF**. Text is cleaned to remove excessive whitespace.
2. **Chunking** — Page text is split with a **RecursiveCharacterTextSplitter** (`chunk_size=800`, `chunk_overlap=150`) so overlapping chunks preserve context across section boundaries.
3. **Embedding** — Chunks are embedded with OpenAI **`text-embedding-3-small`**.
4. **Storage** — Chunks and metadata (`source`, `page`, `policy_type`, `chunk_id`) are persisted in **ChromaDB** under the `policy_documents` collection (`chroma_db/`).
5. **Query rewrite** — User policy questions are rewritten with **`gpt-4o-mini`** into concise search queries for better retrieval.
6. **Retrieval + rerank** — The top 12 vector matches are retrieved, then reranked with a metadata-aware pass: if the query clearly references a policy category (e.g. *refund*, *warranty*, *shipping*, *cancellation*), chunks whose `policy_type` metadata matches are floated to the top; ties and general queries fall back to keyword-overlap scoring. The best 5 chunks are kept.
7. **Grounded answer** — The final answer is generated only from retrieved context. If the context is insufficient, the assistant says so explicitly.
8. **Source attribution** — Answers include file name and page number for each supporting excerpt.

**Demo flow:** run `python -m src.create_policy_pdfs`, upload/index PDFs from the Streamlit sidebar, then ask policy questions such as *"What is the current refund policy?"*

### Response Synthesis Agent

**File:** `src/agents/response_agent.py`

Produces the final customer-facing reply for the support executive.

- Receives `sql_result` and/or `rag_context` from upstream agents
- Uses an OpenAI LLM to write a concise, action-oriented summary
- Explicitly states whether the answer is based on customer data, policy documents, or both
- Handles general greetings and capability questions without specialist input

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Workflow orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM framework | [LangChain](https://github.com/langchain-ai/langchain) |
| LLM provider | OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) |
| Tool layer | In-process MCP-style registry + optional stdio [MCP](https://modelcontextprotocol.io/) server via `mcp.server.fastmcp.FastMCP` |
| Structured data | SQLite + SQLAlchemy (predefined parameterized queries only) |
| Vector store | ChromaDB (`langchain-chroma`) |
| PDF processing | PyMuPDF |
| UI | Streamlit |
| Config | pydantic-settings, python-dotenv |
| Testing | pytest |

---

## Folder Structure

```text
genai-multi-agent-support-assistant/
├── app.py                      # Streamlit UI entry point
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── README.md
│
├── data/                       # Runtime data (gitignored binaries)
│   ├── customers.db            # SQLite database (created by seed script)
│   ├── chroma/                 # ChromaDB persist directory
│   └── policies/               # Policy PDF storage
│
├── src/
│   ├── config.py               # Settings from .env
│   ├── database.py             # SQLAlchemy models and session helpers
│   ├── create_dummy_data.py    # Seed script → data/customers.db
│   ├── create_policy_pdfs.py   # Demo policy PDF generator (PyMuPDF)
│   │
│   ├── agents/
│   │   ├── supervisor.py       # LangGraph Supervisor Agent
│   │   ├── sql_agent.py        # SQL Customer Agent
│   │   ├── rag_agent.py        # Policy RAG Agent
│   │   └── response_agent.py   # Response Synthesis Agent
│   │
│   ├── graph/
│   │   └── workflow.py         # LangGraph state, nodes, and routing
│   │
│   ├── mcp_server/
│   │   └── server.py           # MCP tool registry and FastMCP server
│   │
│   ├── tools/
│   │   ├── sql_tools.py        # SQLite query functions
│   │   └── document_tools.py   # RAG search and Q&A functions
│   │
│   └── rag/
│       ├── document_loader.py  # PDF page extraction and metadata
│       ├── vector_store.py     # Chunking, ChromaDB embed and persist
│       └── retriever.py        # Query rewrite, search, rerank, answer
│
└── tests/
    ├── test_agents.py
    ├── test_rag.py
    └── test_router.py
```

---

## Setup Instructions

**Prerequisites:** Python 3.10+, an OpenAI API key

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/genai-multi-agent-support-assistant.git
   cd genai-multi-agent-support-assistant
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY
   ```

5. **Seed the demo database**

   ```bash
   python -m src.create_dummy_data
   ```

6. **Generate sample policy PDFs** (optional but recommended for local RAG demos)

   ```bash
   python -m src.create_policy_pdfs
   ```

   Then upload/index them via the Streamlit sidebar, or index programmatically from `data/policies/`.

7. **Run the app**

   ```bash
   streamlit run app.py
   ```

8. **Run tests** (optional)

   ```bash
   pytest
   ```

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key for chat and embeddings |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Chat model for agents |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model for ChromaDB |
| `SQLITE_DB_PATH` | No | `data/customers.db` | Path to SQLite customer database |
| `CHROMA_PERSIST_DIR` | No | `chroma_db` | ChromaDB persistence directory |
| `POLICIES_DIR` | No | `data/policies` | Directory for policy PDF files |
| `RAG_CHUNK_SIZE` | No | `800` | Characters per document chunk |
| `RAG_CHUNK_OVERLAP` | No | `150` | Overlap between consecutive chunks |
| `RAG_TOP_K` | No | `4` | Number of chunks retrieved per query |
| `LOG_LEVEL` | No | `INFO` | Application log level |

Optional LangSmith tracing:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=genai-multi-agent-support-assistant
```

---

## Create the Dummy Database

The seed script creates `data/customers.db` with five tables:
`customers`, `orders`, `support_tickets`, `subscriptions`, and `refunds`.

```bash
python -m src.create_dummy_data
```

**Record counts:**

| Table | Rows | Used by SQL Agent? |
|---|---|---|
| customers | 50 | Yes |
| support_tickets | 70 | Yes |
| orders | 80 | No (schema only) |
| subscriptions | 25 | No (schema only) |
| refunds | 18 | No (schema only) |

> The SQL Customer Agent only queries `customers` and `support_tickets`
> through the predefined parameterized lookups listed in the
> [SQL Customer Agent](#sql-customer-agent) section. The `orders`,
> `subscriptions`, and `refunds` tables are seeded so the schema is
> complete and easy to extend later, but they are **not** exposed via tools
> in this demo.

**Demo customers (deterministic, pre-seeded):**

| customer_id | Name | Tier | Highlights |
|---|---|---|---|
| 1 | Ema Johnson  | Premium    | Orders, multiple tickets (resolved Damaged Product, open Refund Request, open Shipping Delay, open High-priority Billing Issue) and one approved refund. |
| 2 | Daniel Smith | Basic      | Open Shipping Delay ticket plus a resolved Account Update ticket. |
| 3 | Priya Patel  | Enterprise | Open High-priority warranty-replacement ticket and a resolved Refund Request ticket. |

These three customers guarantee that the SQL agent's profile, open-ticket,
refund-ticket, and high-priority lookups all return rich, deterministic data
out of the box. Faker-generated customers fill IDs 4..50.

The data is **100% synthetic** and generated locally by Faker with a fixed seed (`42`),
so results are fully reproducible. No external API calls are made.

Re-running the script **drops and recreates** all tables from scratch — safe to rerun
any number of times.

The database file (`data/customers.db`) is listed in `.gitignore` and must **not** be
committed. The `data/` folder is kept in Git via `data/.gitkeep`.

A plain-English schema reference is at `data/schema.md`.

---

## Run the Streamlit App

```bash
streamlit run app.py
```

Or with the project virtualenv directly:

```bash
.venv/bin/streamlit run app.py
```

The app opens in your browser with:

- A **chat interface** for natural-language questions
- A **sidebar** for uploading and indexing policy PDFs
- **Example question buttons** for quick demos
- An **agent-used indicator** on each assistant reply

---

## Sample Questions

Try these in the chat UI — every question maps to a lookup the assistant
actually supports:

| Question | Expected routing |
|----------|------------------|
| Give me a quick overview of customer Ema Johnson's profile and past support ticket details. | SQL |
| Show me Daniel Smith's open support tickets. | SQL |
| Give me Priya Patel's support ticket history. | SQL |
| Show me Priya Patel's refund-related tickets. | SQL |
| Are there any high-priority open tickets? | SQL |
| Can Ema Johnson get a refund based on her support history and the refund policy? | Both |
| What is the current refund policy? | RAG |
| What does the warranty policy say? | RAG |
| Hi, what can you do? | General |

---

## Demo Video

A short walkthrough of the assistant running end-to-end in Streamlit:

> **Demo video:** _Add your Loom / YouTube / MP4 link here._

Suggested recording flow:

1. Seed the database (`python -m src.create_dummy_data`).
2. Generate sample policy PDFs (`python -m src.create_policy_pdfs`) and upload / index them from the Streamlit sidebar.
3. Ask one SQL question (e.g. *"Show me Daniel Smith's open support tickets."*).
4. Ask one RAG / policy question (e.g. *"What is the current refund policy?"*).
5. Ask one hybrid question (e.g. *"Can Ema Johnson get a refund based on her support history and the refund policy?"*).
6. Point out the **agent-used** label that Streamlit renders on each reply (SQL Agent, Policy RAG Agent, Both, or General Response).

---

## Limitations

- **Demo-scale data** — SQLite database is seeded with 50 customers, 70 support tickets, and synthetic `orders` / `subscriptions` / `refunds` rows that are **not** queried by the SQL agent
- **Scoped SQL lookups only** — The SQL Customer Agent only exposes the five predefined customer/ticket lookups listed in the [SQL Customer Agent](#sql-customer-agent) section. It does **not** answer general SQL analytics, orders / subscriptions / refunds reports, or arbitrary natural-language SQL questions, and it does **not** generate raw SQL from the user query
- **Name-based SQL lookups** — Customer extraction relies on heuristics + optional LLM; ambiguous names may fail
- **Policy PDFs required for RAG** — RAG answers nothing until at least one PDF is indexed; generate demo PDFs with `python -m src.create_policy_pdfs` or upload your own via the Streamlit sidebar
- **No authentication** — Streamlit UI has no login or role-based access control
- **Single-tenant** — One shared ChromaDB collection; no per-organization isolation
- **English only** — Routing keywords and prompts are English-centric
- **No conversation memory** — Each question is processed independently (no multi-turn context in the graph)
- **OpenAI dependency** — Requires a valid API key; no local/offline LLM fallback

---

## Future Improvements

- [ ] Multi-turn conversation memory in LangGraph state
- [ ] Automatic batch indexing of PDFs in `data/policies/`
- [ ] Additional predefined, parameterized SQL lookups (e.g. orders by customer, refund records) — kept off by default to preserve safe, deterministic behavior
- [ ] Human-in-the-loop approval for sensitive actions (refunds, account changes)
- [ ] LangSmith tracing dashboard for agent observability
- [ ] Authentication and audit logging for production deployment
- [ ] Support for additional document formats (Word, HTML, Confluence exports)
- [ ] Parallel execution of SQL and RAG agents for `both` routes
- [ ] Evaluation suite with golden Q&A pairs for regression testing
- [ ] Docker Compose setup for one-command deployment

---

## License

MIT
