"""Chat model used by the refund agent (Groq free tier)."""
import os

import truststore

# Use the Windows certificate store (works behind SSL-scanning antivirus).
truststore.inject_into_ssl()

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

AGENT_MODEL = os.getenv("GROQ_AGENT_MODEL", "openai/gpt-oss-120b")

model = ChatGroq(model=AGENT_MODEL, temperature=0, max_retries=6)
