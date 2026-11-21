"""Streamlit dashboard: DeepEval results for the refund agent + a live chat.

Run: uv run streamlit run app.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluations"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Agent-Eval", page_icon="🧾", layout="wide")


# ---------------------------------------------------------------- data
def list_runs() -> list[Path]:
    return sorted(RESULTS_DIR.glob("run_*.json"), reverse=True)


@st.cache_data
def load_run(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def short(text: str, n: int = 48) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def case_label(i: int, case: dict) -> str:
    return f"#{i + 1} · {short(case['input'].replace('I am ', ''))}"


def metric_frame(run: dict) -> pd.DataFrame:
    rows = []
    for i, case in enumerate(run["cases"]):
        for name, m in case["metrics"].items():
            rows.append(
                {
                    "case": case_label(i, case),
                    "metric": name,
                    "kind": "Trace" if name in run["trace_metrics"] else "Single-turn",
                    "score": m["score"],
                    "passed": m["success"],
                    "threshold": m["threshold"],
                    "error": m.get("error"),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("Agent-Eval")
    st.caption("A LangChain refund agent graded by DeepEval on 7 agentic metrics.")
    runs = list_runs()
    run = None
    if runs:
        choice = st.selectbox(
            "Evaluation run",
            runs,
            format_func=lambda p: p.stem.replace("run_", "").replace("_", " · "),
        )
        run = load_run(str(choice), choice.stat().st_mtime)
        st.divider()
        st.markdown(f"**Agent model** \n`{run['agent_model']}`")
        st.markdown(f"**Judge model** \n`{run['judge_model']}`")
        st.markdown(f"**Pinned 'today'** \n`{run['agent_today']}`")
        st.caption(f"Run at {run['timestamp'].replace('T', ' ')} · Groq free tier")
    else:
        st.info("No results yet. Use the **Run evaluation** tab.")

tab_overview, tab_cases, tab_chat, tab_run = st.tabs(
    ["📊 Overview", "🔍 Test cases", "💬 Live agent", "▶️ Run evaluation"]
)


# ---------------------------------------------------------------- overview
with tab_overview:
    if not run:
        st.info("Run an evaluation to see results here.")
    else:
        df = metric_frame(run)
        n_cases = len(run["cases"])
        all_pass = sum(all(m["success"] for m in c["metrics"].values()) for c in run["cases"])

        st.header("Evaluation overview", anchor=False)
        st.caption(
            "Seven refund scenarios, each run through the agent once and scored by an LLM judge. "
            "Trace metrics judge *how* the agent worked; single-turn metrics judge its tool calls and final answer."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Checks passed", f"{int(df['passed'].sum())} / {len(df)}",
                  help="One check = one metric on one test case")
        c2.metric("Pass rate", f"{df['passed'].mean():.0%}")
        c3.metric("Average score", f"{df['score'].mean():.2f}", help="0–1, higher is better")
        c4.metric("Cases passing every metric", f"{all_pass} / {n_cases}")

        if df["error"].notna().any():
            st.warning(f"{int(df['error'].notna().sum())} metric run(s) errored — see the Test cases tab.")

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Score by metric", anchor=False)
            summary = (
                df.groupby(["metric", "kind"], sort=False)
                .agg(score=("score", "mean"), passed=("passed", "sum"), total=("passed", "size"))
                .reset_index()
            )
            summary["label"] = summary.apply(lambda r: f"{int(r.passed)}/{int(r.total)} pass", axis=1)
            base = alt.Chart(summary).encode(
                y=alt.Y("metric:N", sort=list(summary["metric"]), title=None),
                x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), title="Average score"),
            )
            bars = base.mark_bar(cornerRadiusEnd=3).encode(
                color=alt.Color("kind:N", title=None, legend=alt.Legend(orient="bottom")),
                tooltip=["metric", "kind", alt.Tooltip("score:Q", format=".2f"), "label"],
            )
            text = base.mark_text(align="left", dx=4).encode(text="label:N")
            rule = alt.Chart(pd.DataFrame({"t": [0.8]})).mark_rule(strokeDash=[4, 4]).encode(x="t:Q")
            st.altair_chart((bars + text + rule).properties(height=300), use_container_width=True)
            st.caption("Dashed line = 0.8 pass threshold.")

        with right:
            st.subheader("Every check", anchor=False)
            heat = (
                alt.Chart(df)
                .mark_rect(stroke="white", strokeWidth=2)
                .encode(
                    x=alt.X("metric:N", sort=run["metrics"], title=None,
                            axis=alt.Axis(labelAngle=-35, labelLimit=140)),
                    y=alt.Y("case:N", sort=list(dict.fromkeys(df["case"])), title=None,
                            axis=alt.Axis(labelLimit=220)),
                    color=alt.Color("score:Q", scale=alt.Scale(domain=[0, 1], scheme="redyellowgreen"),
                                    title="Score"),
                    tooltip=["case", "metric", alt.Tooltip("score:Q", format=".2f"), "passed"],
                )
            )
            labels = alt.Chart(df).mark_text(fontSize=11).encode(
                x=alt.X("metric:N", sort=run["metrics"]),
                y=alt.Y("case:N", sort=list(dict.fromkeys(df["case"]))),
                text=alt.Text("score:Q", format=".2f"),
            )
            st.altair_chart((heat + labels).properties(height=330), use_container_width=True)

        st.subheader("Pass / fail table", anchor=False)
        table = df.pivot(index="case", columns="metric", values="passed")
        table = table.reindex(index=list(dict.fromkeys(df["case"])), columns=run["metrics"])
        st.dataframe(table.map(lambda v: "✅" if v else "❌"), use_container_width=True)


# ---------------------------------------------------------------- cases
def tool_table(expected: list[dict], called: list[dict]) -> pd.DataFrame:
    rows, used = [], set()
    for e in expected:
        match = next(
            (j for j, c in enumerate(called) if j not in used and c["name"] == e["name"]), None
        )
        c = called[match] if match is not None else None
        if match is not None:
            used.add(match)
        same_args = c is not None and (c.get("input_parameters") or {}) == (e.get("input_parameters") or {})
        rows.append({
            "tool": e["name"],
            "expected args": json.dumps(e.get("input_parameters") or {}),
            "actual args": json.dumps(c.get("input_parameters") or {}) if c else "— not called —",
            "match": "✅" if same_args else ("🟡 args differ" if c else "❌ missing"),
        })
    for j, c in enumerate(called):
        if j not in used:
            rows.append({"tool": c["name"], "expected args": "— not expected —",
                         "actual args": json.dumps(c.get("input_parameters") or {}), "match": "➕ extra"})
    return pd.DataFrame(rows)


with tab_cases:
    if not run:
        st.info("Run an evaluation to see results here.")
    else:
        idx = st.selectbox(
            "Test case", range(len(run["cases"])),
            format_func=lambda i: case_label(i, run["cases"][i]),
        )
        case = run["cases"][idx]
        passed = sum(m["success"] for m in case["metrics"].values())
        st.header(case["input"], anchor=False)
        st.caption(f"{passed}/{len(case['metrics'])} metrics passed · agent took {case['agent_seconds']}s")

        a, b = st.columns(2)
        with a:
            st.markdown("**Expected outcome**")
            st.info(case["expected_output"])
        with b:
            st.markdown("**Agent's answer**")
            if passed == len(case["metrics"]):
                st.success(case["actual_output"])
            else:
                st.warning(case["actual_output"])

        st.subheader("Tool calls", anchor=False)
        st.dataframe(tool_table(case["expected_tools"], case["tools_called"]),
                     use_container_width=True, hide_index=True)

        st.subheader("Metric verdicts", anchor=False)
        for name, m in case["metrics"].items():
            icon = "⚠️" if m.get("error") else ("✅" if m["success"] else "❌")
            score = "error" if m["score"] is None else f"{m['score']:.2f}"
            kind = "trace" if name in run["trace_metrics"] else "single-turn"
            with st.expander(f"{icon} **{name}** — {score}  ·  _{kind}, threshold {m['threshold']}_"):
                if m.get("error"):
                    st.error(m["error"])
                else:
                    st.write(m["reason"] or "_No reason given._")

        with st.expander("Agent trace sent to the judge"):
            for step in case["trace"]["children"]:
                if step["type"] == "tool":
                    st.markdown(f"🔧 **{step['name']}** `{json.dumps(step['input'])}`")
                    st.code(str(step["output"]), language="json")
                else:
                    st.markdown("🧠 **reasoning**")
                    st.caption(step["output"])
            st.markdown("💬 **final answer**")
            st.caption(case["trace"]["output"])


# ---------------------------------------------------------------- live chat
with tab_chat:
    st.header("Talk to the agent", anchor=False)
    st.caption(
        "Try `I am user123, I'd like a refund for the pen` or an unknown user like `user999`. "
        "Users: user123 – user132."
    )
    today = st.date_input("Agent's 'today'", value=pd.Timestamp("2026-10-20"),
                          help="The refund policy is 30 days, so this decides which purchases are eligible.")
    os.environ["AGENT_TODAY"] = today.isoformat()

    if "thread" not in st.session_state:
        st.session_state.thread = str(uuid4())
        st.session_state.history = []

    if st.button("New conversation"):
        st.session_state.thread = str(uuid4())
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            if turn.get("tools"):
                with st.expander(f"🔧 {len(turn['tools'])} tool call(s)"):
                    for t in turn["tools"]:
                        st.markdown(f"**{t['name']}** `{json.dumps(t['args'])}` → `{t['result']}`")

    prompt = st.chat_input("Ask for a refund…")
    if prompt:
        from langchain_core.messages import AIMessage, ToolMessage

        from agent.agent import agent

        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Agent is working…"):
                before = len(agent.get_state({"configurable": {"thread_id": st.session_state.thread}}).values.get("messages", []))
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={"configurable": {"thread_id": st.session_state.thread}},
                )
            new = result["messages"][before:]
            outputs = {m.tool_call_id: m.content for m in new if isinstance(m, ToolMessage)}
            tools = [
                {"name": c["name"], "args": c["args"], "result": outputs.get(c["id"], "")}
                for m in new if isinstance(m, AIMessage)
                for c in m.tool_calls if c["name"].lower() != "answer"
            ]
            reply = result["structured_response"]
            st.write(reply.answer)
            if tools:
                with st.expander(f"🔧 {len(tools)} tool call(s)"):
                    for t in tools:
                        st.markdown(f"**{t['name']}** `{json.dumps(t['args'])}` → `{t['result']}`")
            if not reply.conversation:
                st.caption("The agent marked this conversation as finished.")
        st.session_state.history.append({"role": "assistant", "content": reply.answer, "tools": tools})


# ---------------------------------------------------------------- run eval
with tab_run:
    st.header("Run a new evaluation", anchor=False)
    st.caption(
        "Runs all 7 goldens through the agent and scores the selected metrics. "
        "On Groq's free tier a full run takes several minutes."
    )
    from metrics import METRICS

    chosen = st.multiselect("Metrics", list(METRICS), default=list(METRICS))
    if st.button("Start evaluation", type="primary", disabled=not chosen):
        log = st.empty()
        lines = []
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "evaluations" / "eval_all.py"), *chosen],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", env=env,
        )
        with st.spinner("Evaluating…"):
            for line in proc.stdout:
                if "HTTP Request" in line or " | INFO | " in line:
                    continue
                lines.append(line.rstrip())
                log.code("\n".join(lines[-40:]))
        if proc.wait() == 0:
            st.success("Done — pick the new run in the sidebar.")
            st.cache_data.clear()
        else:
            st.error("The evaluation failed; see the log above.")
