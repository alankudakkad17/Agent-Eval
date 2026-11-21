import sys
from pathlib import Path


def main() -> None:
    """Console entry point: chat with the refund agent."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from agent.agent import chat

    chat()
