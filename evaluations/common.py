"""Shared evaluation runner: run the agent, score metrics, save results."""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluations"))

load_dotenv(ROOT / ".env")
# Pin "today" so the 30-day refund goldens stay valid. Override in .env if needed.
os.environ.setdefault("AGENT_TODAY", "2026-10-20")

from deepeval.test_case import LLMTestCase, ToolCall  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from agent.agent import agent  # noqa: E402
from agent.models import AGENT_MODEL  # noqa: E402
from agent.prompts import agent_system_prompt  # noqa: E402
from dataset import dataset  # noqa: E402
from metrics import METRICS, TRACE_METRICS  # noqa: E402
from model import eval_model  # noqa: E402

RESULTS_DIR = ROOT / "results"
LATEST = RESULTS_DIR / "latest.json"


def run_agent(user_input: str) -> dict:
    """Run one fresh conversation with the agent."""
    return agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": str(uuid4())}},
    )


def _is_answer(name: str) -> bool:
    return name.lower() == "answer"


def tools_called(response: dict) -> list[ToolCall]:
    """Tool calls the agent made, excluding the structured-output "answer" tool."""
    calls = []
    for message in response["messages"]:
        for call in getattr(message, "tool_calls", []) or []:
            if not _is_answer(call["name"]):
                calls.append(ToolCall(name=call["name"], input_parameters=call["args"]))
    return calls


def build_trace(user_input: str, response: dict) -> dict:
    """A compact agent trace for the agentic metrics.

    DeepEval's LangChain callback records every LLM prompt in full (~15k
    tokens per run), which exceeds Groq's free-tier 8k tokens/minute limit.
    This keeps what the metrics actually judge: the policy the agent was given,
    its reasoning, each tool call with its result, and the final answer.
    """
    results = {
        m.tool_call_id: m.content for m in response["messages"] if isinstance(m, ToolMessage)
    }
    children = []
    for message in response["messages"]:
        if not isinstance(message, AIMessage):
            continue
        reasoning = (message.additional_kwargs or {}).get("reasoning_content") or message.content
        if reasoning:
            children.append({"name": "agent_reasoning", "type": "llm", "output": str(reasoning)[:1500]})
        for call in message.tool_calls or []:
            if _is_answer(call["name"]):
                continue
            children.append(
                {
                    "name": call["name"],
                    "type": "tool",
                    "input": call["args"],
                    "output": results.get(call["id"]),
                }
            )
    return {
        "name": "refund_agent",
        "type": "agent",
        "description": agent_system_prompt.strip(),
        "input": user_input,
        "output": response["structured_response"].answer,
        "children": children,
    }


def _measure(metric, test_case) -> dict:
    start = time.time()
    try:
        metric.measure(test_case, _show_indicator=False)
        return {
            "score": metric.score,
            "success": bool(metric.success),
            "threshold": metric.threshold,
            "reason": metric.reason,
            "seconds": round(time.time() - start, 1),
        }
    except Exception as err:  # keep going; record the failure
        return {"score": None, "success": False, "threshold": metric.threshold,
                "reason": None, "error": f"{type(err).__name__}: {err}"[:500],
                "seconds": round(time.time() - start, 1)}


def evaluate_dataset(metric_names: list[str] | None = None, save: bool = True) -> dict:
    """Run every golden through the agent and score the chosen metrics."""
    metric_names = metric_names or list(METRICS)
    cases = []
    for i, golden in enumerate(dataset.goldens, 1):
        print(f"[{i}/{len(dataset.goldens)}] {golden.input}", flush=True)
        start = time.time()
        response = run_agent(golden.input)
        called = tools_called(response)
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=response["structured_response"].answer,
            expected_output=golden.expected_output,
            tools_called=called,
            expected_tools=golden.expected_tools,
        )
        test_case._trace_dict = build_trace(golden.input, response)
        agent_seconds = round(time.time() - start, 1)

        scores = {}
        for name in metric_names:
            scores[name] = _measure(METRICS[name](), test_case)
            s = scores[name]
            mark = "ERR" if s["score"] is None else ("PASS" if s["success"] else "FAIL")
            print(f"    {name:<22} {mark:<4} {s['score']}", flush=True)

        cases.append(
            {
                "input": golden.input,
                "expected_output": golden.expected_output,
                "actual_output": test_case.actual_output,
                "tools_called": [t.model_dump() for t in called],
                "expected_tools": [t.model_dump() for t in golden.expected_tools or []],
                "trace": test_case._trace_dict,
                "agent_seconds": agent_seconds,
                "metrics": scores,
            }
        )

    run = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "agent_model": AGENT_MODEL,
        "judge_model": ", ".join(sorted(eval_model.models_used)),
        "agent_today": os.environ["AGENT_TODAY"],
        "metrics": metric_names,
        "trace_metrics": [m for m in metric_names if m in TRACE_METRICS],
        "cases": cases,
    }
    if save:
        save_run(run)
    print_summary(run)
    return run


def save_run(run: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    text = json.dumps(run, indent=2, default=str)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RESULTS_DIR / f"run_{stamp}.json").write_text(text, encoding="utf-8")
    LATEST.write_text(text, encoding="utf-8")
    print(f"\nSaved results to {LATEST}")


def retry_errors() -> dict:
    """Re-score only the metrics that errored in results/latest.json.

    Uses the saved answer, tool calls and trace, so the agent is not re-run.
    """
    run = json.loads(LATEST.read_text(encoding="utf-8"))
    for i, case in enumerate(run["cases"], 1):
        failed = [n for n, m in case["metrics"].items() if m.get("error")]
        if not failed:
            continue
        print(f"[{i}/{len(run['cases'])}] {case['input']}", flush=True)
        test_case = LLMTestCase(
            input=case["input"],
            actual_output=case["actual_output"],
            expected_output=case["expected_output"],
            tools_called=[ToolCall(**t) for t in case["tools_called"]],
            expected_tools=[ToolCall(**t) for t in case["expected_tools"]],
        )
        test_case._trace_dict = case["trace"]
        for name in failed:
            case["metrics"][name] = s = _measure(METRICS[name](), test_case)
            mark = "ERR" if s["score"] is None else ("PASS" if s["success"] else "FAIL")
            print(f"    {name:<22} {mark:<4} {s['score']}", flush=True)
    used = set(run["judge_model"].split(", ")) | eval_model.models_used
    run["judge_model"] = ", ".join(sorted(used))
    run["timestamp"] = datetime.now().isoformat(timespec="seconds")
    save_run(run)
    print_summary(run)
    return run


def print_summary(run: dict) -> None:
    print("\nMetric                 pass   avg score")
    for name in run["metrics"]:
        rows = [c["metrics"][name] for c in run["cases"]]
        scored = [r["score"] for r in rows if r["score"] is not None]
        passed = sum(r["success"] for r in rows)
        avg = f"{sum(scored) / len(scored):.2f}" if scored else "n/a"
        print(f"{name:<22} {passed}/{len(rows)}    {avg}")
