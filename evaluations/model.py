"""Groq-hosted judge model for DeepEval metrics (free tier)."""
import json
import os

import truststore

# Use the Windows certificate store (works behind SSL-scanning antivirus).
truststore.inject_into_ssl()

import groq  # noqa: E402
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402

load_dotenv()

JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "qwen/qwen3.8-27b")
# Free-tier models each have a 200k tokens/day cap. When the judge runs out,
# continue on the next model instead of failing the rest of the run.
FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GROQ_JUDGE_FALLBACKS", "openai/gpt-oss-20b,openai/gpt-oss-120b").split(",")
    if m.strip()
]


def _daily_limit_hit(err: Exception) -> bool:
    return isinstance(err, groq.RateLimitError) and "tokens per day" in str(err)


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = JUDGE_MODEL):
        self.chain = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]
        self.model_name = model_name
        self.models_used = {model_name}
        super().__init__(model_name)

    def load_model(self):
        # Short retries: a per-minute limit clears quickly, a daily one never will.
        return ChatGroq(model=self.model_name, temperature=0, max_retries=4)

    def _switch_model(self) -> bool:
        idx = self.chain.index(self.model_name)
        if idx + 1 >= len(self.chain):
            return False
        self.model_name = self.chain[idx + 1]
        self.models_used.add(self.model_name)
        self.model = self.load_model()
        print(f"    (judge hit its daily token limit; switching to {self.model_name})", flush=True)
        return True

    def _parse(self, schema, text: str):
        start, end = text.find("{"), text.rfind("}")
        return schema.model_validate(json.loads(text[start : end + 1]))

    def _generate_once(self, prompt: str, schema):
        if schema is None:
            return self.model.invoke(prompt).content
        try:
            return self.model.with_structured_output(schema).invoke(prompt)
        except groq.RateLimitError:
            raise
        except Exception:
            # Fall back to free-form JSON if tool-based structured output fails.
            return self._parse(schema, self.model.invoke(prompt).content)

    async def _a_generate_once(self, prompt: str, schema):
        if schema is None:
            return (await self.model.ainvoke(prompt)).content
        try:
            return await self.model.with_structured_output(schema).ainvoke(prompt)
        except groq.RateLimitError:
            raise
        except Exception:
            return self._parse(schema, (await self.model.ainvoke(prompt)).content)

    def generate(self, prompt: str, schema=None):
        while True:
            try:
                return self._generate_once(prompt, schema)
            except Exception as err:
                if not (_daily_limit_hit(err) and self._switch_model()):
                    raise

    async def a_generate(self, prompt: str, schema=None):
        while True:
            try:
                return await self._a_generate_once(prompt, schema)
            except Exception as err:
                if not (_daily_limit_hit(err) and self._switch_model()):
                    raise

    def get_model_name(self) -> str:
        return f"groq/{self.model_name}"


eval_model = GroqJudge()
