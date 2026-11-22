from types import SimpleNamespace

from app.config import Settings
from app.llm import LLMClient
from app.schemas import Usage


def test_chat_counts_call_when_compatible_api_omits_token_counts():
    settings = Settings(database_url="", openai_base_url=None, openai_api_key="test",
                        chat_model="test", embedding_model="test",
                        chat_input_usd_per_million=None, chat_output_usd_per_million=None,
                        embedding_usd_per_million=None)
    llm = LLMClient(settings)
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=None, completion_tokens=None),
                               choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))])
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    usage = Usage()
    assert llm.chat_json("system", "user", usage) == {"ok": True}
    assert usage.calls == 1
    assert usage.input_tokens == 0


def test_embeddings_are_batched_and_keep_order_and_usage():
    settings = Settings(database_url="", openai_base_url=None, openai_api_key="test",
                        chat_model="test", embedding_model="test",
                        chat_input_usd_per_million=None, chat_output_usd_per_million=None,
                        embedding_usd_per_million=1.0)
    llm = LLMClient(settings)
    calls = []

    def create(model, input):
        calls.append(input)
        return SimpleNamespace(usage=SimpleNamespace(total_tokens=len(input)),
                               data=[SimpleNamespace(index=i, embedding=[float(i)]) for i in reversed(range(len(input)))])

    llm.client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    usage = Usage()
    vectors, tokens = llm.embed([str(i) for i in range(10)], usage)
    assert [len(batch) for batch in calls] == [4, 4, 2]
    assert vectors == [[0.0], [1.0], [2.0], [3.0], [0.0], [1.0], [2.0], [3.0], [0.0], [1.0]]
    assert tokens == 10
    assert usage.calls == 3
    assert usage.embedding_tokens == 10


def test_partial_pricing_does_not_report_total_cost():
    settings = Settings(database_url="", openai_base_url=None, openai_api_key="test",
                        chat_model="test", embedding_model="test",
                        chat_input_usd_per_million=1.0, chat_output_usd_per_million=None,
                        embedding_usd_per_million=None)
    llm = LLMClient(settings)
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                               choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))])
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    usage = Usage()
    llm.chat_json("system", "user", usage)
    assert usage.estimated_cost_usd is None
