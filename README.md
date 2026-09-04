# Red Team Agent

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-1C3C3C)
![Gemini](https://img.shields.io/badge/Gemini-2.5--flash-4285F4?logo=google&logoColor=white)
![Arize Phoenix](https://img.shields.io/badge/Arize-Phoenix-7E5BEF)
![FastAPI](https://img.shields.io/badge/FastAPI-REST-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![CI](https://github.com/Erenemirr/red-agent/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue)

> An autonomous multi-agent system that probes enterprise LLM chatbots for security
> vulnerabilities, **learns from every round**, and closes the self-improvement loop by
> reading its own observability traces.

Built with **LangGraph** for orchestration, **Gemini** for generation/judging, and
**Arize Phoenix** for full observability. Designed for the Google Cloud Rapid Agent
Hackathon (Arize track) — and as a foundation for a real LLM security tool.

> ⚠️ **Authorized use only.** This is a defensive security tool for testing systems you
> own or are explicitly permitted to test. Do not use it against third-party systems.

---

## What it does

You give it a target chatbot's **system prompt** (e.g. a bank or healthcare assistant).
The agent then runs an adversarial loop: it generates attacks, sends them to the target,
scores whether the target's guardrails held, learns which strategies work, and produces a
security report — all while streaming traces to a Phoenix dashboard.

The key idea: **the agent improves across runs.** Every judged attack is written to Phoenix
as a queryable span; on the next run the Analyzer reads that history back and steers the
Attacker toward what has worked — even across separate sessions.

## Highlights

- **7-node LangGraph pipeline** with conditional routing and a real loop.
- **Self-improvement loop** — Analyzer reads past traces from Phoenix (cross-run learning).
- **Two-layer judge** — fast rule-based detection (PII, system-prompt leak, persona, etc.)
  + LLM-as-a-judge (Gemini, structured JSON, 0.0–1.0 score).
- **Multi-turn attacks** — a *campaign* holds a stateful conversation with the target
  (up to `MAX_TURNS`); the attacker sees each response and *escalates*, so techniques like
  crescendo and incremental-escalation actually work across turns.
- **Adaptive attacker intelligence:**
  - *Epsilon-greedy* category selection (explore vs exploit, avoids collapse onto one winner).
  - *PAIR* prompt refinement (learns from failed attempts, tries genuinely different angles).
  - *Laplace smoothing* on success rates (a single lucky success doesn't dominate).
  - *Objective × technique* learning (picks the best technique for each objective type).
- **12 attack techniques × 105 objectives** — objectives sourced from the real
  [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) benchmark + LLM-security goals.
- **Free-tier resilience** — automatic retry-with-backoff on `429 RESOURCE_EXHAUSTED`
  (honors the server's `retryDelay`), optional proactive RPM throttle, and — crucially —
  quota-errored turns are **excluded from learning** so a rate-limit never gets mislearned
  as "the target defended."
- **Human-in-the-loop** — pauses at a configurable critical-finding threshold (CLI prompt or
  REST `/resume` endpoint), powered by LangGraph `interrupt()` + checkpointing.
- **API security** — optional `X-API-Key` auth + per-client sliding-window rate limiting
  on the FastAPI endpoints (no external deps; `/health` stays open for probes).
- **Full observability** — every node and Gemini call traced to Arize Phoenix.
- **Three interfaces** — CLI, REST API (FastAPI), and LangGraph Studio. Containerized with Docker.
- **106 hermetic tests** — run with zero API quota (mock mode), deterministic, fast.

---

## Architecture

```
                                 ┌──────────────► reporter ──► END
                                 │ report
START ─► orchestrator ─►(route)──┼──────────────► human ─►(continue/stop)
            ▲                    │ interrupt
            │                    │ continue
            │                    ▼
            └─ analyzer ◄─ judge ◄─ target ◄─ attacker
```

| Node | Role |
|------|------|
| **orchestrator** | Loop control: continue / human interrupt (≥N critical findings) / report (max rounds). |
| **attacker** | Picks a technique (epsilon-greedy) + objective, generates an attack prompt (Gemini + PAIR). |
| **target** | The chatbot under test (Gemini behind a swappable interface; `EchoTarget` mock for dev). |
| **judge** | Scores the attack: rule-based layer + LLM-as-a-judge. Writes a queryable `attack_evaluation` span. |
| **analyzer** | Computes insights from local history **+ Phoenix cross-run data**; updates strategy. |
| **human** | Human-in-the-loop checkpoint (LangGraph `interrupt`). |
| **reporter** | Generates Markdown + JSON reports. |

Nodes don't call each other — they communicate through a shared `RedTeamState`, merged via
LangGraph reducers (append-only lists for history, overwrite for scalars).

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph |
| LLM | Google Gemini (`google-genai`, `gemini-2.5-flash`) |
| Observability | Arize Phoenix (OpenInference instrumentation) |
| API | FastAPI + Uvicorn |
| Tests | pytest |
| Packaging | Docker / Docker Compose |
| Attack data | JailbreakBench, hand-curated technique repertoire |

## Project structure

```
red-agent/
├── main.py                 # CLI entry point + LangGraph wiring (graph, run, tracing)
├── api.py                  # FastAPI app (/health, /scan, /resume)
├── security.py             # API-key auth + in-memory rate limiting
├── config/settings.py      # env vars & constants
├── agents/
│   ├── state.py            # RedTeamState + reducers
│   ├── orchestrator.py     # loop decisions
│   ├── attacker.py         # category/objective selection, PAIR, prompt generation
│   ├── target.py           # target interface (Gemini / mock)
│   ├── judge.py            # rule-based + LLM judge, queryable spans
│   ├── analyzer.py         # cross-run insights (+ Laplace smoothing)
│   └── reporter.py         # Markdown + JSON reports
├── eval/llm_judge.py       # LLM-as-a-judge (Gemini structured output)
├── tools/
│   ├── gemini.py           # central google-genai client
│   ├── phoenix_mcp.py      # Phoenix read-back (cross-run category stats)
│   └── usage.py            # API call counter (free-tier awareness)
├── data/
│   ├── attack_categories.json    # 12 attack techniques
│   ├── jailbreak_objectives.json # 105 objectives (JailbreakBench + LLM-security)
│   └── target_systems/           # example target system prompts
├── tests/                  # 106 hermetic tests
├── reports/                # generated reports
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

---

## Getting started

### Prerequisites
- Python 3.11+ (developed on 3.13)
- A free [Google AI Studio](https://aistudio.google.com) API key (Gemini)
- *(Optional)* A free [Arize Phoenix Cloud](https://app.phoenix.arize.com) account for tracing

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your keys:
#   GEMINI_API_KEY=...                     (required for real runs)
#   GEMINI_MODEL=gemini-2.5-flash
#   PHOENIX_API_KEY=...                    (optional, for tracing)
#   PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/<your-space>
```

> Without `GEMINI_API_KEY` the system runs in **mock mode** (`EchoTarget` + rule-based judge,
> zero API calls) — useful for development and tests.

---

## Usage

### CLI

```bash
# Run against the built-in example target
python main.py --rounds 3 --yes

# Use your own target system prompt
python main.py --rounds 3 --yes --target data/target_systems/mediasist_saglik_asistani.txt

# Interactive human-in-the-loop (omit --yes): pauses and asks at the critical threshold
python main.py --rounds 6
```

Each round makes ~3 Gemini calls. Reports land in `reports/`; traces stream to Phoenix.

### REST API

```bash
uvicorn api:app --reload
```

Then open **http://127.0.0.1:8000/docs** (Swagger UI), or:

```bash
# Start a scan — pauses at the critical threshold and returns an interrupt
curl -X POST http://127.0.0.1:8000/scan -H "Content-Type: application/json" \
  -d '{"target_system": "Sen Acme Bank botusun...", "max_rounds": 6}'

# Review the findings, then resume
curl -X POST http://127.0.0.1:8000/resume -H "Content-Type: application/json" \
  -d '{"thread_id": "<from /scan>", "decision": "continue"}'
```

Pass `"auto_continue": true` to `/scan` to skip the human checkpoint.

### LangGraph Studio (visual)

```bash
pip install -U "langgraph-cli[inmem]"
langgraph dev    # opens the graph visualizer in your browser
```

### Docker

```bash
docker compose up --build      # API on http://localhost:8000
```

### Tests

```bash
pytest            # 106 hermetic tests, no API quota used
pytest -v         # verbose
```

---

## How the intelligence works

**Explore → exploit → refine.** Early rounds explore the full technique repertoire. Once
explored, the Attacker exploits the historically best technique (`top_categories`) — but with
**epsilon-greedy** randomness so it never collapses onto a single strategy. When an attack
fails, **PAIR** feeds the failed prompt and the target's refusal back to the Attacker, which
generates a fundamentally different angle instead of repeating itself.

**Robust ranking.** The Analyzer ranks techniques by Laplace-smoothed success rate
`(successes+1)/(attempts+2)`, so a single lucky `1/1` doesn't outrank a consistent `4/5`.

**Cross-run learning.** The Judge writes each result as an `attack_evaluation` span with
`attack.category / score / success` attributes. On later runs the Analyzer queries Phoenix
(`tools/phoenix_mcp.py`) and blends that historical data with the current run — so the agent
gets smarter across sessions, not just within one.

## Attack repertoire

**12 techniques (HOW):** role-playing, indirect-injection, context-overflow, persona-switching,
incremental-escalation, authority-claim, token-smuggling, refusal-suppression, crescendo,
many-shot, payload-splitting, virtualization. Hand-curated; example prompts are *inspired by*
[Garak](https://github.com/NVIDIA/garak) probe families (e.g. `dan`, `encoding`, `promptinject`) —
each category's `garak_probe` field cites its inspiration. (Garak itself is not a dependency.)

**105 objectives (WHAT):** 100 from JailbreakBench (Malware, Privacy, Fraud, Disinformation, …)
+ 5 LLM-security goals (system-prompt leak, PII exfiltration, policy bypass, phishing).

Each attack combines one technique with one objective — e.g. *"use persona-switching to make
the target leak its system prompt."*

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Google AI Studio key (empty → mock mode) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model (free-tier friendly) |
| `PHOENIX_API_KEY` | — | Phoenix Cloud key (empty → tracing off) |
| `PHOENIX_COLLECTOR_ENDPOINT` | — | Space-specific Phoenix URL |
| `MAX_ROUNDS` | `10` | Max attack rounds (turns) |
| `MAX_TURNS` | `3` | Max conversation turns per multi-turn campaign |
| `CRITICAL_FINDINGS_FOR_INTERRUPT` | `3` | Critical findings before human checkpoint |
| `CRITICAL_SCORE_THRESHOLD` | `0.8` | Score ≥ this is "critical" |
| `SUCCESS_SCORE_THRESHOLD` | `0.6` | Score ≥ this is a "successful" attack |
| `EXPLORE_EPSILON` | `0.3` | Epsilon-greedy exploration rate |
| `GEMINI_MAX_OUTPUT_TOKENS` | `1024` | Token cap per call |
| `SKIP_LLM_JUDGE_ON_RULE_HIT` | `true` | Skip LLM judge when rule-based already confident |
| `GEMINI_MAX_RETRIES` | `5` | Retries on `429` before giving up (raises `QuotaExceededError`) |
| `GEMINI_RETRY_BASE_DELAY` | `2.0` | Exponential backoff base (server `retryDelay` wins) |
| `GEMINI_RPM` | `0` | Proactive rate limit (calls/min); `0` = off, `5` = free-tier safe |
| `API_KEY` | — | `X-API-Key` required on `/scan` `/resume` (empty → auth off) |
| `RATE_LIMIT_PER_MINUTE` | `60` | Max requests/min per client IP (`0` = off) |
| `ENABLE_TRACING` | `true` | Phoenix tracing on/off |

## Observability

When `PHOENIX_API_KEY` is set, every node and Gemini call is traced. Two instrumentors:
`LangChainInstrumentor` (graph flow) and `GoogleGenAIInstrumentor` (LLM calls — prompts,
responses, token counts). The Analyzer reads these traces back to close the learning loop.

---

## License & ethics

Licensed under the [MIT License](LICENSE).

For authorized security testing, research, and education only. The attack techniques and
objectives are included to test and harden your own systems — use responsibly.
