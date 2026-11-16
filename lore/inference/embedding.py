"""Who turns text into vectors, and the one interface every caller uses.

LORE embeds in five places: book processing (chunks, story beats, codex
entries), the entity resolver (names, predicates), merge adjudication, the
question path (the question and its parts) and the MCP `vector_search` tool.
They all used to build `OllamaProvider` directly, so `nomic-embed-text` on a
local Ollama was the only choice.

`make_embedder(config)` is the single seam now. `[lore.embedding] provider`
picks between:

  * "ollama" (the code default): the local nomic-embed-text, exactly as
    before, in 768 dimensions;
  * "openrouter": `https://openrouter.ai/api/v1/embeddings`, one Bearer key
    (OPENROUTER_API_KEY or ~/.config/lore/openrouter.env), any model
    OpenRouter serves, with `dimensions` sent so the vectors stay the width
    the stores already are (768, `lore.db.lance_store.EMBED_DIM`).

Both return the same `Embedder` surface: `embed(texts, model=None)`,
`healthy()`, `aclose()`. The model name still comes from the caller
(`config.extraction.embed_model_ollama` or `[lore.embedding] model`), and
the prefixes stay where they were (`lore/extract/embed.py`): every site
applies DOCUMENT_PREFIX or QUERY_PREFIX itself, so switching providers
changes only where the text is sent.

The key is read like every other provider key: the environment first, then
~/.config/lore/openrouter.env (mode 600), never lore.toml. It lives only in
the request header.
"""
from __future__ import annotations

import os
import stat
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path

import httpx

from lore.inference.base import ProviderOutputError
from lore.inference.providers import base as providers_base

OPENROUTER = "openrouter"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_VARIABLE = "OPENROUTER_API_KEY"
# The width the LanceDB tables are declared with (lore.db.lance_store.EMBED_DIM).
# Sent as `dimensions` so a swappable model keeps the vectors loadable.
EMBED_DIM = 768

# Characters of error body kept in an exception message.
_BODY_CHARS = 300


class Embedder:
    """What every embedding call site uses: text in, vectors out."""

    model: str

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise NotImplementedError

    async def healthy(self) -> bool:
        raise NotImplementedError

    async def aclose(self) -> None:
        raise NotImplementedError


def openrouter_key_file() -> Path:
    from lore.inference import keyfile
    return keyfile.key_file("openrouter")


def openrouter_key(env: Mapping[str, str] | None = None) -> str | None:
    """The OpenRouter key: OPENROUTER_API_KEY, then ~/.config/lore/openrouter.env.

    A key file whose mode lets anyone but its owner read it is refused, as
    gemini.env is: the message names the file and the fix, never the key.
    """
    env = os.environ if env is None else env
    value = (env.get(OPENROUTER_KEY_VARIABLE) or "").strip()
    if value:
        return value
    path = openrouter_key_file()
    try:
        path.stat()
    except FileNotFoundError:
        return None
    from lore.inference import keyfile
    if not keyfile.mode_is_private(path):
        from lore.inference.gemini import GeminiKeyFileError
        mode = stat.S_IMODE(path.stat().st_mode)
        raise GeminiKeyFileError(
            f"refusing to read {path}: its mode is {oct(mode)[2:]}, which lets"
            f" someone other than its owner read it. Run `chmod 600 {path}`.")
    for line in path.read_text().splitlines():
        name, sep, found = line.partition("=")
        if sep and name.strip() == OPENROUTER_KEY_VARIABLE:
            found = found.strip().strip("'\"").strip()
            if found:
                return found
    return None


class OpenRouterEmbedder(Embedder):
    """Embeddings from OpenRouter's `/embeddings` endpoint.

    The request is the documented OpenAI-shaped one: `{"model", "input",
    "dimensions"}`; the reply is `{"data": [{"embedding": [...], "index": n}]}`.
    Row order is by `index`, so a server that answers out of order still
    lines its vectors up with the texts it was given.
    """

    def __init__(self, api_key: str | None, *, model: str,
                 base_url: str = OPENROUTER_BASE_URL, dimensions: int = EMBED_DIM,
                 timeout: float = 600.0,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        if not model:
            raise ValueError("[lore.embedding] model is required for the OpenRouter embedder")
        self.model = model
        self._dimensions = dimensions
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers,
                                         timeout=timeout, transport=transport)

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        body: dict = {"model": model or self.model, "input": texts,
                      "encoding_format": "float"}
        if self._dimensions:
            body["dimensions"] = self._dimensions
        try:
            response = await self._client.post("/embeddings", json=body)
        except httpx.HTTPError as exc:
            raise providers_base.ProviderTransportError(
                f"could not reach OpenRouter embeddings ({type(exc).__name__}).",
                provider=OPENROUTER) from None
        if response.status_code in (401, 403):
            raise providers_base.ProviderKeyError(
                f"HTTP {response.status_code}: OpenRouter refused the key"
                f" ({OPENROUTER_KEY_VARIABLE} or ~/.config/lore/openrouter.env).",
                provider=OPENROUTER)
        if response.status_code == 429:
            raise providers_base.ProviderQuotaError(
                f"HTTP 429 from OpenRouter embeddings for {body['model']}.",
                provider=OPENROUTER)
        if response.status_code >= 500:
            raise providers_base.ProviderUnavailableError(
                f"HTTP {response.status_code} from OpenRouter embeddings: the"
                " service is failing or overloaded.", provider=OPENROUTER)
        if response.status_code >= 400:
            raise providers_base.ProviderError(
                f"HTTP {response.status_code} from OpenRouter embeddings for"
                f" {body['model']}: {response.text[:_BODY_CHARS].strip()}",
                provider=OPENROUTER)
        try:
            data = response.json()["data"]
            ordered = sorted(data, key=lambda row: int(row.get("index", 0)))
            return [row["embedding"] for row in ordered]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderOutputError(
                "OpenRouter returned a reply that is not an embeddings response",
                response.text[:_BODY_CHARS], body["model"], provider=OPENROUTER) from exc

    async def healthy(self) -> bool:
        """One one-word embedding: the cheapest proof the key and model work."""
        try:
            vectors = await self.embed(["ok"])
        except Exception:  # noqa: BLE001 - a health check reports, never raises
            return False
        return bool(vectors and vectors[0])

    async def aclose(self) -> None:
        await self._client.aclose()


def make_embedder(config, *, timeout: float | None = None):
    """The embedder `[lore.embedding] provider` (and the model) asks for.

    "ollama" is imported here, not at module level, so the OpenRouter path
    does not depend on the local client being installed or running.
    `timeout` applies to the Ollama client only (the question path allows a
    slow CPU longer than processing does).
    """
    embedding = getattr(config, "embedding", None)
    provider = getattr(embedding, "provider", "ollama")
    if provider != OPENROUTER:
        from lore.inference.ollama import OllamaProvider
        if timeout is not None:
            return OllamaProvider(config.ollama, timeout=timeout)
        return OllamaProvider(config.ollama)
    model = embedding.model or config.extraction.embed_model_ollama
    return OpenRouterEmbedder(openrouter_key(), model=model,
                              base_url=embedding.base_url or OPENROUTER_BASE_URL,
                              dimensions=embedding.dimensions)


def embedder_name(config) -> str:
    """How a manifest or a log line names the embedder now in use."""
    embedding = getattr(config, "embedding", None)
    if getattr(embedding, "provider", "ollama") == OPENROUTER:
        return embedding.model or config.extraction.embed_model_ollama
    return config.extraction.embed_model_ollama


def embed_model_for(config) -> str:
    """The model id every embed call passes, whatever the provider."""
    return embedder_name(config)
