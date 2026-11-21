# Agent-Eval

A customer-support **refund agent** built with LangChain, and a **DeepEval** test suite
that grades how well it behaves — not just its final answer, but which tools it calls,
with what arguments, and how sensibly it plans its steps.

Both the agent and the evaluation judge run on **Groq's free tier**, and a **Streamlit
dashboard** shows the results and lets you chat with the agent live.

## How it works

```
user message ──► refund agent (LangChain create_agent, Groq LLM)
                   │  tools: get_user_purchase_history, get_today_date,
                   │         calculate_right_refund, issue_refund_ticket
                   ▼
              structured Answer {answer, conversation}
                   │
                   ▼
   DeepEval (judge = Groq LLM) scores the trace + final answer on 7 metrics
```

### The agent (`agent/`)

| File | Purpose |
|------|---------|
| `agent.py` | Builds the agent: model + 4 tools + system prompt + `Answer` output + in-memory chat history |
| `tools.py` | The 4 tools. `issue_refund_ticket` is a stub (nothing is really refunded) |
| `prompts.py` | Refund policy: identify the user first, 30-day limit, refund only requested & purchased items |
| `data.py` | Mock purchase history for `user123`–`user132` |
| `objects.py` | `Answer` (structured reply; exposed to the model as the `answer` tool) and `PurchaseItem` |
| `models.py` | Groq chat model (default `openai/gpt-oss-120b`) |
| `middleware.py` | Recovers when the model replies in plain text instead of calling the `answer` tool |

### The evaluations (`evaluations/`)

`dataset.py` holds 7 goldens (input, expected answer, expected tool calls) covering a
full refund, a partial refund, an info-only request, an unknown user, an expired item,
a mix of eligible/expired items, and an item the user never bought.

| Metric | Type | What it checks |
|--------|------|----------------|
| Task Completion | trace | Did the agent accomplish the task per the policy? |
| Step Efficiency | trace | No unnecessary steps |
| Plan Quality | trace | Was the plan sound? |
| Plan Adherence | trace | Did it follow its own plan? |
| Tool Correctness | single-turn | Right tools with the right arguments vs. `expected_tools` |
| Argument Correctness | single-turn | Were the tool arguments sensible for the input? |
| Answer Correctness (GEval) | single-turn | Same conclusion as `expected_output` |

Other files: `metrics.py` (metric definitions), `model.py` (Groq judge wrapped as a
`DeepEvalBaseLLM`), `common.py` (runs the agent, builds the trace, scores, saves results).

**Compact trace:** DeepEval's LangChain callback records every full LLM prompt (~15k
tokens per run), but Groq's free tier allows only 8k tokens per minute per model, so a
single judge request would be rejected. `common.py` instead builds a compact trace —
the policy, the agent's reasoning, each tool call with its result, and the final answer —
and attaches it to the test case before calling each metric's `measure()`.

Each full run saves `results/run_<timestamp>.json` and `results/latest.json`
(git-ignored); the dashboard reads them.

**Fixed date:** the refund goldens depend on "today". Evaluations pin it with
`AGENT_TODAY=2026-10-20` so results don't drift as real time passes. Set `AGENT_TODAY`
in `.env` to change it; when unset outside the evals, the agent uses the real date.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10.

```powershell
uv sync
copy .env.example .env
```

Put a free Groq key (https://console.groq.com/keys) in `.env` as `GROQ_API_KEY`.

> Behind an SSL-intercepting antivirus (e.g. Kaspersky), add `--native-tls` to `uv` commands.
> At runtime the code uses `truststore`, so Python trusts the Windows certificate store.

## Run

Dashboard (results, per-case drill-down, live chat, run new evals):

```powershell
uv run streamlit run app.py
```

Chat with the agent in the terminal:

```powershell
uv run agent-eval
```

Run all 7 metrics:

```powershell
uv run python evaluations/eval_all.py
```

Only some metrics (saved too):

```powershell
uv run python evaluations/eval_all.py "Tool Correctness" "Answer Correctness"
```

Or one metric per script, printed but not saved:

```powershell
uv run python evaluations/tool_correctness.py
```

Each run makes many LLM calls (agent + judge for every golden and metric). Calls run
one at a time and the Groq client retries on rate limits, so a full run takes several minutes.

## Configuration (`.env`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `GROQ_API_KEY` | — | Required |
| `GROQ_AGENT_MODEL` | `openai/gpt-oss-120b` | Model the agent uses (needs tool calling) |
| `GROQ_JUDGE_MODEL` | `qwen/qwen3.8-27b` | Model DeepEval uses as judge (separate rate-limit quota from the agent) |
| `AGENT_TODAY` | real date (evals: `2026-10-20`) | Pins `get_today_date` |

## Notes and limitations

- Groq's free model list changes; `llama-3.3-70b-versatile` is no longer available. If a
  model disappears, set `GROQ_AGENT_MODEL` / `GROQ_JUDGE_MODEL` to one listed at
  https://console.groq.com/docs/models.
- `gpt-oss` models call the reply tool in lowercase, which is why it is named `answer`.
- The LLM judge is not deterministic; scores can move a little between runs.
- `issue_refund_ticket` is a stub, and the purchase data is mock data.
