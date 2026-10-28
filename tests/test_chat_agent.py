from __future__ import annotations

from types import SimpleNamespace

from agents.chat_agent import ChatAgent
from providers.base import GenerationResponse


class RecordingProvider:
    def __init__(self):
        self.prompt = ""

    def generate_text(self, model_id, prompt, **kwargs):
        self.prompt = prompt
        return GenerationResponse("Use the second option from earlier.", model_id)


def test_chat_agent_uses_compact_prior_turns_and_relevant_private_memory() -> None:
    provider = RecordingProvider()
    captured = {}

    class Memory:
        def search(self, query, *, limit):
            captured["query"] = query
            captured["limit"] = limit
            return [SimpleNamespace(
                user_query="Compare the two methods",
                conclusions="The second method is cheaper and met the requirements.",
                findings="",
                unresolved_questions=["Confirm availability in the selected region"],
                memory_id="memory-1",
            )]

    answer = ChatAgent(provider, "model", memory_manager=Memory()).answer(
        "continue with that choice",
        prior_turns=[
            {"role": "user", "content": "Compare the two methods"},
            {"role": "assistant", "content": "The second method is cheaper."},
        ],
        use_web=False,
        use_documents=False,
    )

    assert answer.text == "Use the second option from earlier."
    assert answer.memory_hits == ["memory-1"]
    assert captured["limit"] == 2
    assert "Compare the two methods" in provider.prompt
    assert "second method is cheaper" in provider.prompt
    assert "Confirm availability" in provider.prompt
    assert len(provider.prompt) < 5000


def test_simple_arithmetic_skips_web_search_and_document_retrieval() -> None:
    provider = RecordingProvider()
    actions = []

    class UnusedWeb:
        def search(self, *args, **kwargs):
            raise AssertionError("trivial arithmetic should not trigger web search")

    class UnusedRag:
        def retrieve(self, *args, **kwargs):
            raise AssertionError("trivial arithmetic should not query document embeddings")

    answer = ChatAgent(
        provider,
        "model",
        web_search=UnusedWeb(),
        rag_pipeline=UnusedRag(),
    ).answer("What is 2 + 2?", on_step=lambda action, status, detail: actions.append((action, status, detail)))
    assert answer.text == "Use the second option from earlier."
    assert ("Generating a concise answer", "running", "") in actions
    assert any(action == "Generating a concise answer" and status == "complete"
               and detail == "Model: model" for action, status, detail in actions)
