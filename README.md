# GenAI Multi-Agent Customer Support Assistant

## Overview

This project is a GenAI multi-agent assistant for a customer support executive. It answers natural language questions using:

- Structured customer data from SQLite
- Uploaded policy PDFs through RAG

The assistant can look up customer profiles and ticket history, search indexed policy documents, and combine both sources into a clear support response.

## Assignment Objective

The system supports:

- Querying structured customer-related data using natural language
- Processing policy PDFs into a searchable knowledge base
- Generating accurate, context-aware support responses

## Key Features

- SQL Customer Agent for customer profiles and support ticket history
- Policy RAG Agent for uploaded policy PDFs
- LangGraph Supervisor Agent for routing
- Response Synthesis Agent for final answers
- MCP Server / tool layer for SQL and document tools
- Streamlit chat UI with PDF upload

## Architecture

![GenAI multi-agent customer support assistant architecture](assets/architecture.png)

```text
┌──────────────────────────────────────────────────────────────────────┐
│                        Customer Support Executive                     │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          Streamlit Chat UI                            │
│              Natural language chat + policy PDF upload                │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      LangGraph Supervisor Agent                       │
│           Routes each question to SQL, RAG, both, or general          │
└───────────────┬──────────────────┬──────────────────┬────────────────┘
                │                  │                  │
                ▼                  ▼                  ▼
┌────────────────────────┐ ┌────────────────────────┐ ┌────────────────────────┐
│   SQL Customer Agent   │ │    Policy RAG Agent    │ │  Final Response Agent  │
│ Customer profile and   │ │ Policy search and Q&A  │ │ Combines retrieved     │
│ support ticket lookup  │ │ over uploaded PDFs     │ │ data into final answer │
└───────────┬────────────┘ └───────────┬────────────┘ └───────────▲────────────┘
            │                          │                          │
            ▼                          ▼                          │
┌────────────────────────┐ ┌────────────────────────┐              │
│     MCP SQL Tools      │ │  MCP Document Tools    │              │
│ Parameterized lookups  │ │ PDF ingestion + RAG    │              │
└───────────┬────────────┘ └───────────┬────────────┘              │
            │                          │                          │
            ▼                          ▼                          │
┌────────────────────────┐ ┌────────────────────────┐              │
│ SQLite Customer DB     │ │ ChromaDB Vector Store  │              │
│ customers + tickets    │ │ indexed policy chunks  │              │
└───────────┬────────────┘ └───────────┬────────────┘              │
            │                          │                          │
            └──────────────┬───────────┘                          │
                           │                                      │
                           ▼                                      │
                 ┌──────────────────────┐                        │
                 │ Retrieved Context    │────────────────────────┘
                 │ SQL data + policies  │
                 └──────────┬───────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 Final Context-Aware Support Response                  │
└──────────────────────────────────────────────────────────────────────┘
```

The supervisor decides whether a question needs customer data, policy context, both, or a general response. Specialist agents retrieve the relevant context through the MCP-style tool layer, and the response agent produces the final answer for the support executive.

## Project Setup

### Prerequisites

- Python 3.10+
- OpenAI API key

### Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the virtual environment with:

```bash
.venv\Scripts\activate
```

### Configure Environment

Copy the example environment file and add your OpenAI API key:

```bash
cp .env.example .env
```

Required variable:

```env
OPENAI_API_KEY=your-openai-api-key
```

Optional variables such as `OPENAI_MODEL`, `SQLITE_DB_PATH`, `CHROMA_PERSIST_DIR`, and `POLICIES_DIR` can be left at their defaults for the demo.

### Create Demo Data

Seed the SQLite database:

```bash
python -m src.create_dummy_data
```

Run this before using customer-related SQL questions.

Generate optional synthetic demo policy PDFs:

```bash
python -m src.create_policy_pdfs
```

The SQLite database is created at `data/customers.db`. Policy PDFs used for the demo may include public/sample documents or synthetic demo policy PDFs. The synthetic refund, warranty, and shipping policy PDFs are generated in `data/policies/` for broader RAG testing.

The project uses a synthetic SQLite database generated by `src.create_dummy_data`. It includes customer profiles, support tickets, orders, refunds, and subscriptions. The SQL Customer Agent uses predefined, parameterized SQL tools to retrieve customer profiles, ticket history, open tickets, and refund-related support tickets. The additional tables make the dataset more realistic and can be extended in future versions.

Note on generated local files:

- `data/customers.db` is generated locally and not committed.
- ChromaDB/vector index files are generated locally and not committed.

## Usage Instructions

### Run the App

```bash
streamlit run app.py
```

The Streamlit app provides:

- A chat interface for natural language support questions
- A sidebar for uploading and indexing policy PDFs
- An indicator showing which agent handled each response

### Use Policy PDFs

To use RAG over policies:

1. Start the Streamlit app.
2. Upload a policy PDF from the sidebar, or use the sample/demo PDFs from `data/policies/`.
3. Click **Process Policy PDF**.
4. Ask a policy-related question in the chat.

### Example Questions

- Give me a quick overview of customer Ema Johnson's profile and past support ticket details.
- Show me Daniel Smith's open support tickets.
- Show me Priya Patel's refund-related tickets.
- What is the current refund policy?
- What does the warranty policy say?
- Can Ema Johnson get a refund based on her support history and the refund policy?

## Demo Video

Demo video: [Add URL here]

## Main Components

- `app.py` - Streamlit UI
- `src/graph/workflow.py` - LangGraph workflow and routing
- `src/agents/supervisor.py` - Supervisor routing agent
- `src/agents/sql_agent.py` - Customer data agent
- `src/agents/rag_agent.py` - Policy RAG agent
- `src/agents/response_agent.py` - Final response agent
- `src/mcp_server/server.py` - MCP-style tool registry and optional server
- `src/tools/sql_tools.py` - SQLite lookup tools
- `src/tools/document_tools.py` - PDF ingestion and policy search tools
- `src/rag/` - PDF loading, vector storage, retrieval, and RAG logic

## Running Tests

```bash
python -m pytest
```

## Notes

- Customer data is synthetic and stored in SQLite.
- Policy PDFs can be public/sample or synthetic demo PDFs.
- SQL access uses predefined, parameterized tools instead of LLM-generated raw SQL.
- RAG answers are generated from retrieved policy document context.
- Mixed questions combine SQL customer data and policy document context.
- Generated files such as `data/customers.db` and ChromaDB indexes are created locally and should not be committed.
