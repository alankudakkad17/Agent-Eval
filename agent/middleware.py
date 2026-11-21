"""Recover from Groq rejecting a reply that skipped the `answer` tool.

With a tool-based response format the model must finish by calling `answer`.
gpt-oss sometimes writes the final reply as plain text instead; Groq then
returns 400 `tool_use_failed` with the text in `failed_generation`. This
middleware turns that text into a proper `answer` tool call so the run
finishes normally.
"""
import json
from uuid import uuid4

import groq
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import AIMessage


def _recovered_call(text: str) -> dict:
    try:
        data = json.loads(text)
        if isinstance(data, dict) and str(data.get("name", "")).lower() == "answer":
            args = data.get("arguments") or data.get("parameters") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if "answer" in args:
                return args
    except (json.JSONDecodeError, TypeError):
        pass
    return {"answer": text.strip(), "conversation": True}


@wrap_model_call
def recover_plain_reply(request, handler):
    try:
        return handler(request)
    except groq.BadRequestError as err:
        body = err.body if isinstance(err.body, dict) else {}
        detail = body.get("error", body)
        text = detail.get("failed_generation") if isinstance(detail, dict) else None
        if not text or detail.get("code") != "tool_use_failed":
            raise
        args = _recovered_call(text)
        return AIMessage(
            content="",
            tool_calls=[{"name": "answer", "args": args, "id": f"recovered-{uuid4().hex[:8]}"}],
        )
