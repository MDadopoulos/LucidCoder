# LucidCoder

A self-disciplined coding agent for [Terminal-Bench 2.0](https://www.tbench.ai/), packaged as a purple agent for the [AgentBeats](https://agentbeats.dev) / [AgentX](https://rdi.berkeley.edu/agentx-agentbeats) coding-track.

LucidCoder is **not** a generic ReAct loop. It runs a strict four-stage pipeline on every task — **decompose → plan (with checker) → execute (with retry + anti-paralysis guard) → verify (test run + anti-pattern grep + artifact check)** — inspired by the GSD framework's goal-backward verification, the superpowers library's verification-before-completion gate, and the morphllm-observed pattern that *scaffolding > model*.

---

## How it works

The Green orchestrator (`terminal-bench-green`) opens an A2A conversation, sends the task description, and exposes a remote shell via the `terminal-bench-shell-v1` protocol. LucidCoder responds turn-by-turn with `exec_request` JSON payloads until it emits a `final` payload.

```
A2A message/send: {"kind":"task", "instruction":"..."}
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│                  LucidCoder per-session FSM                       │
│                                                                   │
│ STAGE 1  DECOMPOSE                                                │
│   • 5 canned probes (pwd, file scan, README, test scripts, tools) │
│   • LLM call -> truths + artifacts + key_links + verify_cmd       │
│                                                                   │
│ STAGE 2  PLAN + CHECKER                                           │
│   • LLM emits bite-sized steps (cmd, timeout, expected_exit,      │
│     fail_action, writes)                                          │
│   • Plan-checker audits against 7 dimensions; ≤1 revision         │
│                                                                   │
│ STAGE 3  EXECUTE                                                  │
│   • One exec_request per step                                     │
│   • Up to 3 fix attempts per step (LLM proposes single fix cmd)   │
│   • Anti-paralysis guard: ≥5 read-only commands triggers diagnose │
│                                                                   │
│ STAGE 4  VERIFY                                                   │
│   • Run verify_cmd                                                │
│   • Anti-pattern grep panel + artifact-existence check (one shell │
│     command, batched)                                             │
│   • LLM judge: passed | gaps                                      │
│   • On fail, replan with gap delta (≤1 replan)                    │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
A2A reply: {"kind":"final", "output":"<summary>"}
```

All per-session state — probe outputs, decompose JSON, plan, trace, verify result — is persisted under `workspace/scratch/{context_id}/` for inspection and debugging.

---

## Repository layout

```
LucidCoder/
├── workspace/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── amber-manifest.json5            # AgentBeats registration
│   ├── .env.example
│   ├── src/
│   │   ├── server.py                   # A2A HTTP server (port 9009)
│   │   ├── executor.py                 # A2A AgentExecutor wrapper
│   │   ├── agent.py                    # Per-context_id orchestrator
│   │   ├── controller.py               # 4-stage state machine
│   │   ├── protocol.py                 # terminal-bench-shell-v1 codec
│   │   ├── session.py                  # Per-context_id state object
│   │   ├── prompts.py                  # All stage prompts
│   │   ├── model.py                    # Gemini adapter + key rotation
│   │   ├── scratch.py                  # Per-session FS lifecycle
│   │   └── stages/
│   │       ├── decompose.py            # canned probes + LLM derive
│   │       ├── plan.py                 # plan + plan-checker
│   │       ├── execute.py              # step expectations, fixes
│   │       └── verify.py               # verify_cmd + panel + judge
│   └── tests/
│       ├── test_protocol.py
│       ├── test_execute_heuristics.py
│       └── test_controller_smoke.py    # full FSM walk with stub LLM
├── README.md
└── .gitignore
```

---

## Configuration

| Variable             | Default                  | Purpose                                |
|----------------------|--------------------------|----------------------------------------|
| `GOOGLE_API_KEY` ★   | —                        | Gemini AI Studio API key (required)    |
| `GOOGLE_API_KEY_2..5`★| —                       | Backup keys for rotation on quota fail |
| `MODEL_ID`           | `gemini-3.1-pro-preview` | Model used by all four stages          |
| `SERVER_PORT`        | `9009`                   | A2A server port                        |
| `SERVER_HOST`        | `0.0.0.0`                | A2A server bind                        |
| `LOG_LEVEL`          | `INFO`                   | Python logging level                   |
| `SCRATCH_DIR`        | `workspace/scratch`      | Per-session scratch root               |

Put secrets in `workspace/.env` for local runs, or supply them through the AgentBeats config when deploying.

---

## Running locally

```bash
cd workspace
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env
# fill GOOGLE_API_KEY

python -m src.server                                  # listens on :9009
```

Health and discovery:

```bash
curl http://localhost:9009/health
curl http://localhost:9009/.well-known/agent-card.json
```

End-to-end test against the green agent (in another terminal/container):

```bash
docker run --rm -p 9010:9010 -v /var/run/docker.sock:/var/run/docker.sock \
  -e WORKSPACE=/workspace -e EXEC_BASE_URL=http://host.docker.internal:9010 \
  ghcr.io/RDI-Foundation/terminal-bench-green:latest
```

---

## Running in Docker

```bash
docker build -t lucidcoder -f workspace/Dockerfile .
docker run --rm -p 9009:9009 -e GOOGLE_API_KEY=$GOOGLE_API_KEY lucidcoder
```

The published image used by the AgentBeats manifest is `ghcr.io/mdadopoulos/lucidcoder:latest`.

---

## Tests

```bash
cd workspace
pytest tests/                          # full suite (no LLM calls; uses stubs)
pytest tests/test_protocol.py          # wire format
pytest tests/test_execute_heuristics.py # anti-paralysis guard classifier
pytest tests/test_controller_smoke.py  # full FSM walk
```

---

## Design provenance

The four-stage discipline draws on three sources:

- **GSD (Get-Shit-Done) framework** — goal-backward verification (verify the artifact, not "I ran something"), `must_haves` frontmatter (truths/artifacts/key_links), atomic per-step commits, fix-attempt cap = 3, analysis-paralysis guard.
- **Superpowers skill library** (Jesse Vincent / `obra/superpowers`) — verification-before-completion gate ("no completion claim without fresh evidence"), test-driven discipline, bite-sized writing-plans format.
- **Morph TB-2 analysis + leaderboard signals** — explicit planner with updatable todo list, fallback model on failure, scaffolding > model.

Compound-engineering's cross-task skill-evolution loop is **deliberately deferred**: a one-shot benchmark submission has no opportunity to compound between tasks within scoring, and the architecture cleanly supports adding a `skills/learned/` retrieval layer later.

---

## License

MIT
