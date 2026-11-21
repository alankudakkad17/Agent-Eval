"""The refund-support agent that the evaluations exercise."""
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver

from .middleware import recover_plain_reply
from .models import model
from .objects import answer
from .prompts import agent_system_prompt
from .tools import (
    calculate_right_refund,
    get_today_date,
    get_user_purchase_history,
    issue_refund_ticket,
)

agent = create_agent(
    model=model,
    tools=[
        get_user_purchase_history,
        get_today_date,
        calculate_right_refund,
        issue_refund_ticket,
    ],
    system_prompt=agent_system_prompt,
    # Groq rejects JSON mode together with tools, so the reply comes back via a tool call.
    response_format=ToolStrategy(answer),
    middleware=[recover_plain_reply],
    checkpointer=InMemorySaver(),
)


def chat() -> None:
    """Small interactive loop: `uv run agent-eval`."""
    from uuid import uuid4

    config = {"configurable": {"thread_id": str(uuid4())}}
    print("Refund agent (type 'quit' to exit)")
    while True:
        text = input("you> ").strip()
        if text.lower() in {"quit", "exit", ""}:
            break
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config=config)
        answer = result["structured_response"]
        print(f"agent> {answer.answer}")
        if not answer.conversation:
            break
