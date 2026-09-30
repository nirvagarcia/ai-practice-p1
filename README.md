# AI Practice pt.1

A personal learning playground to practice modern AI engineering technologies in a small, fast, throwaway project. No production goals here — just hands-on experimentation.

## Purpose

This repo exists to get practical experience with:

- **LangGraph / LangChain** — building agent workflows and finite state machine (FSM)-style graphs
- **OpenTelemetry** — instrumenting LLM calls and tracing agent execution
- **RAGAS** — evaluating RAG (Retrieval-Augmented Generation) pipelines
- **OpenAI API** — via `langchain-openai`

## Tech Stack

| Tool | Purpose |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | FSM/graph-based agent orchestration |
| [LangChain OpenAI](https://python.langchain.com/) | LLM integration |
| [OpenTelemetry SDK](https://opentelemetry.io/) | Local manual spans with privacy-first attributes |
| [RAGAS](https://github.com/explodinggradients/ragas) | RAG pipeline evaluation metrics |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | Environment variable management |

## Setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install the package with dev tooling; add the `eval` extra to enable RAGAS:

   ```powershell
   python -m pip install -e ".[dev]"
   # Use this instead if you want RAGAS evaluation:
   python -m pip install -e ".[dev,eval]"
   ```

3. Fill in `.env` (see `.env.example`). Only `OPENAI_API_KEY` is required; empty values use defaults.

4. Run:

   ```powershell
   python -m pytest                                  # offline, no API key needed
   python -m evaluation.common.validate_datasets     # dataset integrity
   python -m bank_assistant.rag.ingest               # build local Chroma index
   python -m bank_assistant.main                     # interactive chat (CLI)
   python -m bank_assistant.main --mode graph-smoke  # one message per route
   python -m bank_assistant.main --mode rag-smoke    # retrieve + grounded generation
   python -m evaluation.rag.run --variant rag-v1 [--split dev] [--ragas]
   python -m evaluation.rag.run --compare rag-v1,crag --dataset data/eval/rag.jsonl --split holdout
   python -m evaluation.routing.run [--split dev]
   python -m evaluation.end_to_end.run --smoke --report evaluation/results/runtime-smoke.json
   ```

OpenTelemetry exports spans to the console. Prompts, responses, account identifiers and
fixture values are never attached to spans; only request IDs, workflow/outcome, model role,
retrieval counts, tool names, durations and error types are recorded.

The RAG baseline remains the default. Set `RAG_STRATEGY=crag` to enable bounded corrective
retrieval, or compare both on the same cases with:

```powershell
python -m evaluation.rag.run --compare rag-v1,crag --dataset data/eval/rag.jsonl --split holdout
```

CRAG is experimental; the report
shows paired answer/retrieval changes, latency and estimated cost rather than choosing a winner.

Request limits include `MAX_INPUT_CHARS`, `MAX_OUTPUT_CHARS`, `MAX_TOTAL_TOKENS_PER_REQUEST`,
per-call `MAX_TOKENS_PER_REQUEST` and model timeout. Cost caps require input/output USD-per-million
token rates plus `PRICING_VERSION`; use conservative rates for the most expensive configured
model. Token estimates and costs are approximations, not provider invoices. `ENABLE_MODERATION`
opts into sending request text to OpenAI moderation; it is off by default, and
`MODERATION_FAIL_CLOSED` controls behavior when that call fails.

`python -m evaluation.end_to_end.run --smoke --report evaluation/results/runtime-smoke.json`
measures a small end-to-end sample. Missing provider usage is reported as unknown, not zero.

## Status

🚧 Work in progress — this is a sandbox for learning, expect experimental and messy code.

## Disclaimer

This project is for educational purposes only and is not intended for production use.
