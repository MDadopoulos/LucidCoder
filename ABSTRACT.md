# LucidCoder — Abstract (AgentX Sprint 3 / Coding Agent Track)

LucidCoder is a Terminal-Bench 2.0 purple agent that replaces the typical ReAct
inner-loop with a strict four-stage pipeline executed across the
`terminal-bench-shell-v1` A2A protocol: **decompose → plan (with checker) →
execute (with retry + anti-paralysis guard) → verify (test-run + anti-pattern
grep + artifact-check).** Each task is treated as a finite-state machine
spanning many A2A turns; per-stage state is persisted to a per-session scratch
directory for auditability.

The DECOMPOSE stage runs five deterministic shell probes (orient, file scan,
README/task read, test-script read, tooling detection) before a single LLM
call derives the observable *truths* the task demands, the *artifacts* that
must exist or be modified, the cross-file *key links* that must be wired, and
the exact *verify command* that the verifier will judge against. PLAN converts
the decompose into bite-sized steps (one shell command each, with
`expected_exit`, `expected_stdout_contains`, `fail_action`, and a list of
expected writes); a plan-checker audits the plan along seven dimensions —
coverage, artifacts, key-links, scope-reduction language, scope budget,
verify-last, and anti-pattern shapes — and triggers up to one revision.
EXECUTE issues one step at a time as A2A `exec_request` payloads, retries
failures up to three times with a model-proposed fix, and aborts if five
consecutive read-only commands have made no progress (anti-paralysis guard).
VERIFY first runs the verify command, then a single batched shell call that
performs an anti-pattern grep on every authored file plus an existence check
on every must-exist artifact, before an LLM judge renders PASS/FAIL with a
gap list; on FAIL, the controller re-plans once with the gap delta before
emitting `final`.

The agent is built on the AgentBeats purple-agent template — A2A on port
9009, registered via `amber-manifest.json5`, distributed as
`ghcr.io/mdadopoulos/lucidcoder:latest`. Gemini 3.1 Pro is the only model;
up to five rotating API keys handle quota pressure. The discipline is
inspired by the GSD framework's goal-backward verification, the superpowers
library's verification-before-completion gate, and morphllm's TB-2
observation that scaffolding outweighs raw model choice. The design is
deliberately conservative for Sprint 3: a cross-task skill-evolution layer
is left as a clean extension point (a `skills/learned/` directory queried
from DECOMPOSE) but not implemented in this submission.
