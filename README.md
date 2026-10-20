# Rice Yield Prediction & Advisory — Kanpur, Uttar Pradesh

Pixel-level, satellite-based rice yield prediction with a local, grounded RAG advisory.
Offline-first: inference uses Ollama only (ADR-018).

> Status: Phase 0 scaffold. Sections below are filled in as phases complete.

## Hardware used
32 GB RAM, NVIDIA RTX 5070 Ti (16 GB VRAM).

## Install
```bash
uv sync --all-groups
cp .env.example .env      # fill in GEE project + CDS key
ollama pull qwen2.5:14b-instruct-q6_K
ollama pull llama3:latest
ollama pull nomic-embed-text
```

## Run
`make data | features | train | rag | test | app | paper | all` (see Makefile).
App: http://localhost:8501.

## Design records
See `knowledge/DECISIONS.md` (ADRs), `knowledge/PROJECT_FLOW.md`.

## GEE quota discipline
Every Earth Engine export is estimated and logged to `logs/gee_quota.jsonl` before it runs;
dry-run on a 1-year subset first (Community tier, 150 EECU-hours).

## License
MIT
