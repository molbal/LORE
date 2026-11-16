"""The OpenRouter gateway (chat completions + embeddings), offline.

Everything here goes through a faked HTTP transport: request shape, strict
schema handling, the model cascade, error mapping, usage, the factory wiring,
the multi-mode refusal, and `[lore.embedding] provider = "openrouter"`.
"""
import asyncio
import json
import os

import httpx
import pytest
from pydantic import BaseModel

from lore.config import EmbeddingConfig, LlmConfig, LoreConfig
from lore.inference import embedding, key_scope
from lore.inference.key_scope import KeyScope
from lore.inference.providers import base, factory
from lore.inference.providers.openrouter_model import OpenRouterTextModel
from lore.inference.providers.schema import strict_schema

BASE = "https://openrouter.ai/api/v1"
KEY = "sk-or-v1-testKEYtestKEYtestKEY-0001"


class Reply(BaseModel):
    answer: str
    note: str | None = None


def completion(content, finish="stop", usage=(11, 5), status=200):
    if status != 200:
        return httpx.Response(status, json={"error": {"message": "nope"}})
    body = {"id": "x", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content}}]}
    if usage is not None:
        body["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1],
                         "total_tokens": usage[0] + usage[1]}
    return httpx.Response(200, json=body)


class Server:
    """A fake OpenRouter: replies in order, records every request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request):
        self.requests.append(request)
        reply = self.replies.pop(0) if self.replies else completion("{}")
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def body(self, i=0):
        return json.loads(self.requests[i].content)


def model(server, key=KEY, models=("minimax/minimax-m3.1-flash",), **kw):
    return OpenRouterTextModel(key, models=models,
                               transport=httpx.MockTransport(server), **kw)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def local_mode(monkeypatch):
    monkeypatch.delenv("LORE_MULTI_USER", raising=False)


# --- request shape --------------------------------------------------------------

def test_json_goes_to_chat_completions_with_a_json_schema():
    server = Server(completion('{"answer": "yes", "note": null}'))
    m = model(server)
    got = run(m.generate_json("Is it?", Reply))
    assert got == Reply(answer="yes")
    request = server.requests[0]
    assert str(request.url) == f"{BASE}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["x-title"] == "LORE"
    body = server.body()
    assert body["model"] == "minimax/minimax-m3.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Is it?"}]
    assert body["temperature"] == 0 and body["stream"] is False
    assert body["max_tokens"] == 8192
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "Reply"
    assert fmt["json_schema"]["schema"] == strict_schema(Reply)
    assert fmt["json_schema"]["strict"] is True
    assert (m.last_model, m.requests_made) == ("minimax/minimax-m3.1-flash", 1)


def test_plain_text_and_usage():
    server = Server(completion("plain words"))
    m = model(server)
    assert run(m.generate("Say it.")) == "plain words"
    assert "response_format" not in server.body()
    assert (m.input_tokens, m.output_tokens) == (11, 5)
    assert KEY not in repr(m)


def test_a_refused_schema_falls_back_to_json_object():
    server = Server(httpx.Response(400, json={"error": {
        "message": "response_format json_schema is not supported by this model"}}),
        completion('{"answer": "yes", "note": null}'))
    m = model(server)
    assert run(m.generate_json("Is it?", Reply)) == Reply(answer="yes")
    assert server.body(1)["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in server.body(1)["messages"][0]["content"]


def test_a_bad_json_reply_is_asked_once_with_the_error():
    server = Server(completion("{\"answer\": 1}"), completion('{"answer": "yes"}'))
    m = model(server)
    assert run(m.generate_json("Is it?", Reply)) == Reply(answer="yes")
    assert len(server.body(1)["messages"]) == 3
    assert "not valid" in server.body(1)["messages"][-1]["content"]


def test_two_bad_replies_are_an_output_error():
    server = Server(completion("{bad"), completion("{bad"))
    m = model(server)
    with pytest.raises(base.ProviderOutputError):
        run(m.generate_json("Is it?", Reply))


def test_the_cascade_moves_to_the_next_model_on_429_and_503():
    server = Server(completion("", status=429), completion("", status=503),
                    completion("answered by the third"))
    m = model(server, models=("a/one", "b/two", "c/three"))
    assert run(m.generate("q")) == "answered by the third"
    assert [server.body(i)["model"] for i in range(3)] == ["a/one", "b/two", "c/three"]
    assert m.last_model == "c/three"


def test_a_401_is_final_and_never_tries_another_model():
    server = Server(completion("", status=401))
    m = model(server, models=("a/one", "b/two"))
    with pytest.raises(base.ProviderKeyError):
        run(m.generate("q"))
    assert len(server.requests) == 1


def test_402_out_of_credit_is_a_final_quota_error():
    server = Server(completion("", status=402))
    with pytest.raises(base.ProviderQuotaError) as caught:
        run(model(server).generate("q"))
    assert "credit" in str(caught.value)
    assert caught.value.daily is True


def test_a_transport_failure_is_mapped():
    server = Server(httpx.ConnectError("down"))
    with pytest.raises(base.ProviderTransportError):
        run(model(server).generate("q"))


def test_a_long_reply_is_an_output_error():
    server = Server(completion("half a sen", finish="length"))
    with pytest.raises(base.ProviderOutputError):
        run(model(server).generate("q"))


def test_a_key_is_required():
    with pytest.raises(base.ProviderKeyError):
        OpenRouterTextModel(None, models=("m",))


# --- keys and the factory -------------------------------------------------------

def _config(tmp_path, **llm):
    return LoreConfig(data_dir=tmp_path / "data", extraction_provider="gemini",
                      selection={"provider": "gemini"},
                      synthesis={"provider": "gemini"},
                      llm=llm or {})


def test_the_key_comes_from_the_environment_then_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert factory.local_key("openrouter") is None
    path = tmp_path / ".config" / "lore" / "openrouter.env"
    path.parent.mkdir(parents=True)
    path.write_text(f"OPENROUTER_API_KEY={KEY}\n")
    assert factory.local_key("openrouter") == KEY
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    assert factory.local_key("openrouter") == "sk-or-env"


def test_credentials_build_and_name_the_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    cfg = _config(tmp_path, provider="openrouter",
                  model="minimax/minimax-m3.1-flash",
                  models=["google/gemini-3.1-flash-lite"], concurrency=3,
                  fast_model="qwen/qwen3.7-flash")
    creds = factory.credentials(cfg)
    assert (creds.provider, creds.key, creds.model) == (
        "openrouter", KEY, "minimax/minimax-m3.1-flash")
    assert creds.models == ("google/gemini-3.1-flash-lite",)
    assert KEY not in repr(creds)
    m = factory.build(creds, cfg, "extract", gemini_models=[])
    assert isinstance(m, OpenRouterTextModel)
    # fast_model first for the short steps; the configured cascade still follows
    assert m.models == ("qwen/qwen3.7-flash", "minimax/minimax-m3.1-flash",
                        "google/gemini-3.1-flash-lite")
    assert m._concurrency == 3


def test_openrouter_is_refused_in_multi_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    monkeypatch.setenv("LORE_MULTI_USER", "1")
    cfg = _config(tmp_path, provider="openrouter", model="m/one")
    with pytest.raises(ValueError, match="multi"):
        factory.credentials(cfg)


def test_a_missing_key_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        factory.credentials(_config(tmp_path, provider="openrouter", model="m/one"))


def test_config_requires_a_model_and_accepts_the_section(tmp_path):
    with pytest.raises(Exception, match="model"):
        LlmConfig(provider="openrouter")
    assert LlmConfig(provider="openrouter", model="m/one").model == "m/one"
    assert EmbeddingConfig().provider == "ollama"
    assert EmbeddingConfig().dimensions == 768


# --- embeddings -----------------------------------------------------------------

def embed_server(vectors, dimension=768):
    def handler(request: httpx.Request):
        body = json.loads(request.content)
        return httpx.Response(200, json={
            "data": [{"embedding": [float(i)] * dimension, "index": i}
                     for i in range(len(body["input"]))]})
    return handler


def test_the_openrouter_embedder_sends_model_input_and_dimensions():
    seen: list[dict] = []

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [
            {"embedding": [0.5] * 768, "index": 1},
            {"embedding": [0.25] * 768, "index": 0}]})

    embedder = embedding.OpenRouterEmbedder(
        KEY, model="qwen/qwen3-embedding-8b",
        transport=httpx.MockTransport(handler))
    vectors = run(embedder.embed(["a", "b"]))
    assert seen[0] == {"model": "qwen/qwen3-embedding-8b", "input": ["a", "b"],
                       "encoding_format": "float", "dimensions": 768}
    # ordered by `index`, not by arrival
    assert vectors[0][0] == 0.25 and vectors[1][0] == 0.5


def test_the_embedder_reports_a_refused_key():
    embedder = embedding.OpenRouterEmbedder(
        "sk-or-bad", model="m", transport=httpx.MockTransport(
            lambda request: httpx.Response(401, json={"error": {"message": "no"}})))
    with pytest.raises(base.ProviderKeyError):
        run(embedder.embed(["a"]))


def test_make_embedder_follows_the_config(tmp_path):
    local = EmbeddingConfig()
    cfg = _config(tmp_path)
    cfg = cfg.model_copy(update={"embedding": local})
    assert isinstance(embedding.make_embedder(cfg), object)
    remote = cfg.model_copy(update={"embedding": EmbeddingConfig(
        provider="openrouter", model="qwen/qwen3-embedding-8b")})
    made = embedding.make_embedder(remote)
    assert isinstance(made, embedding.OpenRouterEmbedder)
    assert embedding.embedder_name(remote) == "qwen/qwen3-embedding-8b"
    assert embedding.embedder_name(cfg) == cfg.extraction.embed_model_ollama
