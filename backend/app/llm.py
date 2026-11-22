import json

from openai import OpenAI

from app.config import Settings, get_settings
from app.schemas import Usage


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다")
        self.client = OpenAI(api_key=self.settings.openai_api_key,
                             base_url=self.settings.openai_base_url, timeout=30, max_retries=0)

    def _add_cost(self, usage: Usage, tokens: int, rate: float | None) -> None:
        if not tokens:
            return
        if rate is None:
            usage.cost_known = False
            usage.estimated_cost_usd = None
        elif usage.cost_known:
            usage.estimated_cost_usd = (usage.estimated_cost_usd or 0) + tokens * rate / 1_000_000

    def embed(self, texts: list[str], usage: Usage | None = None) -> tuple[list[list[float]], int]:
        vectors = []
        total_tokens = 0
        for start in range(0, len(texts), 4):
            if usage is not None:
                usage.calls += 1
            response = self.client.embeddings.create(model=self.settings.embedding_model,
                                                     input=texts[start:start + 4])
            tokens = (response.usage.total_tokens or 0) if response.usage else 0
            total_tokens += tokens
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
            if usage is not None:
                usage.embedding_tokens += tokens
                self._add_cost(usage, tokens, self.settings.embedding_usd_per_million)
        return vectors, total_tokens

    def chat_json(self, system: str, user: str, usage: Usage) -> dict:
        for _ in range(2):
            usage.calls += 1
            response = self.client.chat.completions.create(
                model=self.settings.chat_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            if response.usage:
                input_tokens = response.usage.prompt_tokens or 0
                output_tokens = response.usage.completion_tokens or 0
                usage.input_tokens += input_tokens
                usage.output_tokens += output_tokens
                self._add_cost(usage, input_tokens, self.settings.chat_input_usd_per_million)
                self._add_cost(usage, output_tokens, self.settings.chat_output_usd_per_million)
            try:
                value = json.loads(response.choices[0].message.content or "")
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
        raise ValueError("LLM이 유효한 JSON을 반환하지 않았습니다")
