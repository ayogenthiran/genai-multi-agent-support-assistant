# GenAI Multi-Agent Customer Support Assistant

A Python reference implementation of an AI-powered customer support assistant for support executives. The system combines **LangGraph** for multi-agent orchestration, an **MCP-style tool layer** for modular backend access, **SQLite** for structured customer data, **ChromaDB** for policy document retrieval, and **Streamlit** as the chat UI.

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

This project demonstrates how to build a production-style **multi-agent GenAI application** that helps customer support teams answer questions quickly and accurately. A support executive can ask natural-language questions about:

- **Customer accounts** — profiles, ticket history, open issues
- **Company policies** — refunds, warranties, shipping, cancellations
- **Combined scenarios** — e.g., *"Can Ema get a refund based on her ticket history and the refund policy?"*

The **LangGraph Supervisor Agent** routes each question to the right specialist. Specialists fetch data through a unified **MCP tool-access layer**, and the **Response Synthesis Agent** produces a single, executive-ready reply.

---

## Problem Statement

Customer support teams often work across two disconnected information sources:

1. **Structured data** — customer profiles, orders, and support tickets stored in a database
2. **Unstructured knowledge** — refund, warranty, and shipping policies stored in PDF documents

Switching between SQL dashboards and policy PDFs is slow and error-prone. Support executives need one interface that understands their question, retrieves the right data, and returns a clear, grounded answer.

---

## Assignment Objective

This project was built to demonstrate:

| Goal | Implementation |
|------|----------------|
| Multi-agent orchestration | **LangGraph** state graph with supervisor routing |
| Tool modularity | **MCP Server** as a standardized tool-access layer |
| Structured data access | **SQLite** + SQLAlchemy for customer/ticket lookups |
| Policy knowledge retrieval | **ChromaDB** vector store with OpenAI embeddings |
| User interface | **Streamlit** chat UI with PDF upload support |
| LLM integration | **LangChain** + OpenAI for classification, RAG, and synthesis |

---

## Key Features

- **Intelligent routing** — Supervisor classifies queries as SQL, RAG, both, or general
- **Customer lookups** — Profile and ticket history from SQLite via MCP tools
- **Policy Q&A** — Retrieval-augmented generation over uploaded policy PDFs
- **Hybrid answers** — Combines customer data and policy context in one response
- **PDF ingestion** — Upload and index new policy documents from the Streamlit sidebar
- **Source attribution** — Final answers state whether they are based on customer data, policy documents, or both
- **Optional MCP server** — Tools can be exposed over stdio for external MCP hosts

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

The **MCP (Model Context Protocol) Server** acts as a **tool-access layer** — a clean boundary between AI agents and backend capabilities.

**Why it matters:**

- **Modularity** — Agents call tools by name instead of importing database or vector-store code directly
- **Reusability** — The same six tools are available via Python callables, LangChain `@tool` wrappers, and an optional FastMCP stdio server
- **Separation of concerns** — SQL logic lives in `src/tools/sql_tools.py`; RAG logic lives in `src/rag/`; the MCP layer only exposes a stable interface

**Registered tools:**

| Tool | Purpose |
|------|---------|
| `customer_profile_lookup` | Look up a customer by name |
| `customer_ticket_lookup` | List all tickets for a customer |
| `open_ticket_lookup` | List open tickets for a customer |
| `policy_document_search` | Semantic search over indexed policy chunks |
| `policy_question_answer` | Answer a policy question with RAG |
| `pdf_ingestion` | Index a policy PDF into ChromaDB |

Run the MCP server standalone (stdio transport):

```bash
python -m src.mcp_server.server
```

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

Handles questions about customer accounts and support tickets.

- Extracts customer names from natural language (heuristics + optional LLM)
- Calls MCP tools: `customer_profile_lookup`, `customer_ticket_lookup`, `open_ticket_lookup`
- Returns structured JSON (`sql_result`) for downstream synthesis
- Backed by **SQLite** via SQLAlchemy (`data/customers.db`)

### Policy RAG Agent

**File:** `src/agents/rag_agent.py`

Answers questions using company policy and FAQ documents.

- Calls MCP tools: `policy_question_answer` and `policy_document_search`
- Retrieves relevant chunks from **ChromaDB** using OpenAI embeddings
- Returns grounded policy context (`rag_context`) with supporting excerpts

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
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM framework | [LangChain](https://github.com/langchain-ai/langchain) |
| LLM provider | OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) |
| Tool protocol | [MCP](https://modelcontextprotocol.io/) (FastMCP) |
| Structured data | SQLite + SQLAlchemy |
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
│       ├── document_loader.py  # PDF loading and chunking
│       ├── vector_store.py     # ChromaDB embed and persist
│       └── retriever.py        # Similarity search at query time
│
└── tests/
    ├── test_agents.py
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
| `CHROMA_PERSIST_DIR` | No | `data/chroma` | ChromaDB persistence directory |
| `POLICIES_DIR` | No | `data/policies` | Directory for policy PDF files |
| `RAG_CHUNK_SIZE` | No | `800` | Characters per document chunk |
| `RAG_CHUNK_OVERLAP` | No | `120` | Overlap between consecutive chunks |
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

The seed script creates `data/customers.db` with **5 customers** and **10 support tickets**:

```bash
python -m src.create_dummy_data
```

**Sample customers:**

| Name | Type | Location |
|------|------|----------|
| Ema Johnson | premium | Seattle, WA |
| Daniel Smith | standard | Austin, TX |
| Priya Patel | premium | Chicago, IL |
| Michael Brown | standard | Denver, CO |
| Sara Wilson | premium | Boston, MA |

Tickets include refunds, warranty claims, delivery delays, account issues, and product replacements with varied statuses (`open`, `resolved`, `in progress`, `closed`).

Re-running the script **recreates** the database from scratch.

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

Try these in the chat UI:

| Question | Expected routing |
|----------|------------------|
| What is the current refund policy? | RAG |
| Give me a quick overview of customer Ema's profile and past support ticket details. | SQL |
| Can Ema get a refund based on her past support ticket and the refund policy? | Both |
| Show me Ema's open support tickets. | SQL |
| What does the warranty policy say? | RAG |
| Hi, what can you do? | General |

---

## Demo Video

> **[Demo video link — add your Loom/YouTube URL here]**
>
> _Suggested recording flow: seed database → upload a policy PDF → ask one SQL, one RAG, and one hybrid question → show agent routing labels._

---

## Limitations

- **Demo-scale data** — SQLite database contains only 5 customers and 10 tickets
- **Name-based SQL lookups** — Customer extraction relies on heuristics; ambiguous names may fail
- **No authentication** — Streamlit UI has no login or role-based access control
- **Single-tenant** — One shared ChromaDB collection; no per-organization isolation
- **English only** — Routing keywords and prompts are English-centric
- **No conversation memory** — Each question is processed independently (no multi-turn context in the graph)
- **Policy PDFs required for RAG** — Generate demo PDFs with `python -m src.create_policy_pdfs` or upload your own via the Streamlit sidebar
- **OpenAI dependency** — Requires a valid API key; no local/offline LLM fallback

---

## Future Improvements

- [ ] Multi-turn conversation memory in LangGraph state
- [ ] Automatic batch indexing of PDFs in `data/policies/`
- [ ] SQL agent with LLM-generated SQL for ad-hoc analytics queries
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
