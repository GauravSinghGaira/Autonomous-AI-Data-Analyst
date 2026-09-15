# 🤖 Autonomous AI Data Analyst

An end-to-end system that lets users upload a CSV/Excel dataset and interact with it through
an AI-powered analytical agent — combining classical data science (Pandas/NumPy/Scikit-learn),
agentic orchestration (LangGraph + LangChain), tool calling, and retrieval-augmented generation
(RAG) over supporting business documentation.

This is **not an LLM wrapper or chatbot**. The LLM never performs a calculation itself — every
number reported to the user comes from a verified Python tool call (pandas aggregation,
scikit-learn model, etc.), and the LLM's job is strictly to route requests, call the right
tools, and explain the results in natural language.

---

## Why this project exists

Most "AI + data" demos either (a) let an LLM freely hallucinate statistics from a data preview,
or (b) are a thin wrapper around `df.describe()` with a chat UI bolted on. This project is
built to demonstrate the actual skill set needed to ship this kind of system in production:

- Real EDA logic (profiling, distributions, correlations, duplicates, missingness)
- Real ML (Isolation Forest anomaly detection, baseline predictive modeling with genuine
  train/test metrics)
- An agent architecture where tool calling is enforced, not optional
- A RAG pipeline for grounding answers in business context (KPI definitions, data dictionary)
- A router that decides which capability a request needs, rather than one giant do-everything
  prompt
- A proper service boundary: FastAPI backend, Streamlit frontend, Docker packaging, tests

---

## Architecture

```
┌─────────────────┐        HTTP         ┌──────────────────────────────────────────┐
│  Streamlit UI    │ ───────────────────▶│              FastAPI Backend              │
│  (app/ui)         │◀─────────────────── │              (app/api)                    │
└─────────────────┘                     └───────────────┬────────────────────────────┘
                                                          │
                                                          ▼
                                          ┌───────────────────────────────┐
                                          │        AnalystAgent            │
                                          │      (app/agents)              │
                                          └───────────────┬─────────────────┘
                                                          │
                                                          ▼
                              ┌────────────────────────────────────────────────┐
                              │              LangGraph Workflow                  │
                              │              (app/workflow/graph.py)             │
                              │                                                  │
                              │   entry ──▶ router ──▶ branch (tool-calling LLM) │
                              │                    │                             │
                              │         ┌──────────┼──────────┬──────────┐       │
                              │         ▼          ▼          ▼          ▼       │
                              │   data_analysis  visualization ml_anomaly  RAG    │
                              └─────────┬──────────┬──────────┬──────────┬────────┘
                                        ▼          ▼          ▼          ▼
                              ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌────────────┐
                              │ data_tools │ │ viz_tools  │ │ ml_tools │ │ rag_tools  │
                              └─────┬──────┘ └─────┬──────┘ └────┬─────┘ └─────┬──────┘
                                    ▼              ▼              ▼             ▼
                              profiler.py    visualizer.py   anomaly.py    vectorstore.py
                              (Pandas/NumPy)   (Plotly)     prediction.py  (ChromaDB +
                                                             (Scikit-learn) Sentence-Transformers)
```

### Key architectural decisions

- **Router-first design.** Every user message is classified by the LLM (with a keyword-based
  fallback if the LLM call fails) into `data_analysis`, `visualization`, `ml_anomaly`,
  `document_retrieval`, or `general` *before* any tools are bound. Each branch's LLM call only
  sees the 1-2 tools relevant to it. In practice this is far more reliable than giving one LLM
  call access to all 6+ tools at once and hoping it picks correctly.
- **Tool calling is mandatory, not advisory.** The system prompt explicitly forbids the LLM
  from computing numbers itself. All arithmetic, aggregation, correlation, and modeling logic
  lives in plain Python functions (`app/data_processing`, `app/ml`) that are unit-tested
  independently of the LLM.
- **Session-scoped state, not global state.** An uploaded dataset lives in an in-memory
  `SessionRegistry` keyed by `session_id`, so multiple users/datasets can be handled by one
  running backend without cross-contamination.
- **Pydantic everywhere structure matters.** Tool arguments, API requests/responses, and
  configuration are all Pydantic models, so malformed LLM tool-calling arguments are rejected
  before they reach pandas/sklearn, and the API has a documented, validated contract.

---

## Project structure

```
ai-data-analyst/
├── app/
│   ├── agents/            # LLM factory + high-level AnalystAgent facade
│   ├── workflow/           # LangGraph state, router, graph orchestration
│   ├── tools/               # LangChain StructuredTool wrappers (+ Pydantic schemas)
│   ├── data_processing/    # Loading, profiling, querying, visualization, reports, sessions
│   ├── ml/                 # Isolation Forest anomaly detection, baseline prediction
│   ├── rag/                 # Embeddings, ChromaDB vector store, document ingestion
│   ├── api/                 # FastAPI app (upload / chat / report / health endpoints)
│   ├── ui/                  # Streamlit frontend
│   └── utils/               # Config (pydantic-settings) and logging
├── tests/                    # Pytest unit tests for profiler, anomaly, prediction, router
├── sample_data/
│   ├── sales_data.csv        # Synthetic sample dataset (outliers, missing values, duplicates)
│   └── docs/                 # Sample RAG source docs: data dictionary, KPI definitions
├── scripts/setup.py          # One-shot: create storage dirs + ingest sample RAG docs
├── requirements.txt
├── Dockerfile                # FastAPI backend image
├── Dockerfile.streamlit      # Streamlit frontend image
├── docker-compose.yml        # Orchestrates API + UI + local Ollama
├── .env.example
└── README.md
```

---

## Technologies used

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph (routing + tool-calling subgraphs) |
| LLM/tool integration | LangChain (`StructuredTool`, `bind_tools`) |
| LLM | Local via Ollama (default, free) or OpenAI (optional) |
| Data analysis | Pandas, NumPy |
| ML | Scikit-learn (IsolationForest, RandomForest, StandardScaler) |
| Visualization | Plotly |
| RAG embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (persistent, local) |
| Validation | Pydantic v2 / pydantic-settings |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Testing | Pytest |
| Packaging | Docker / Docker Compose |

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

By default `LLM_PROVIDER=ollama` (free, local, no API key). To use OpenAI instead, set
`LLM_PROVIDER=openai` and `OPENAI_API_KEY` in `.env`.

If using Ollama, install it from [ollama.com](https://ollama.com) and pull a model:

```bash
ollama pull llama3.2:3b
```

### 3. Run setup (creates storage dirs + indexes sample RAG docs)

```bash
python scripts/setup.py
```

### 4. Run the backend

```bash
uvicorn app.api.main:app --reload --port 8000
```

### 5. Run the frontend (separate terminal)

```bash
streamlit run app/ui/streamlit_app.py
```

Open `http://localhost:8501`, upload `sample_data/sales_data.csv`, and start asking questions.

### Or: run everything with Docker Compose

```bash
docker compose up --build
```

This starts Ollama, the FastAPI backend, and the Streamlit UI together. Pull a model into the
Ollama container once it's up: `docker exec -it ai-analyst-ollama ollama pull llama3.2:3b`.

### Run tests

```bash
pytest
```

---

## End-to-end workflow

1. **Upload** — user uploads a CSV/Excel file via Streamlit → FastAPI `/api/upload`.
2. **Automatic EDA** — the backend immediately runs `profile_dataset()` (shape, dtypes,
   missingness, duplicates, distributions, correlations), `auto_generate_charts()` (histograms,
   correlation heatmap, top-category bar chart, strongest-pair scatter plot), and
   `detect_anomalies()` (Isolation Forest over numeric columns) — all before the user asks
   anything, so they land on a populated dashboard.
3. **Ask a question** — the user types a natural-language question in the chat tab.
4. **Route** — the router LLM call classifies the question into one of four capability
   branches.
5. **Tool-calling agent turn** — the branch-specific LangGraph subgraph gives the LLM only the
   relevant tools and loops until the LLM produces a final natural-language answer, having
   pulled real numbers from pandas/scikit-learn/ChromaDB along the way.
6. **Response assembly** — any charts or retrieved document snippets generated during the turn
   are attached to the response alongside the LLM's explanation.
7. **Report** — at any point, the user can generate and download a Markdown analysis report
   summarizing the profile and anomaly findings.

---

## Example queries to try

With `sample_data/sales_data.csv` loaded:

- *"What's in this dataset?"* → routes to `data_analysis`, calls `profile_dataset`
- *"What's the average revenue by region?"* → `data_analysis`, calls `query_dataframe` with
  `groupby=region`
- *"Show me a histogram of customer satisfaction"* → `visualization`, calls `generate_chart`
- *"Are there any outliers in this data?"* → `ml_anomaly`, calls `detect_anomalies`
- *"Can we predict churn from the other columns?"* → `ml_anomaly`, calls `predict_target`
- *"What does an at-risk customer mean?"* → `document_retrieval`, calls `search_documents`
  against the KPI definitions doc
- *"Why do outlier orders matter for average order value?"* → `document_retrieval`, grounds
  the answer in `data_dictionary.md` / `kpi_definitions.md`

---

## Adding your own supporting documents

Drop additional `.md`/`.txt` files (data dictionaries, KPI glossaries, business context) into
`sample_data/docs/` and re-run:

```bash
python -m app.rag.ingest
```

---

## Known limitations / what a production version would add

- Session state is in-memory (single process) — a production deployment would move this to
  Redis or a database so the API can scale horizontally.
- The baseline predictive model (`app/ml/prediction.py`) only uses numeric features; a
  production version would encode categoricals and support hyperparameter search.
- No authentication/authorization is implemented on the API.
- The router relies on the LLM's structured output; while it has a keyword fallback, a
  production system would also log/monitor misrouted queries to improve the classifier prompt
  over time.
