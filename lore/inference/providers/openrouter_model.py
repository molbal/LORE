"""OpenRouter (https://openrouter.ai) as the remote model, in local mode.

OpenRouter serves any of its models behind OpenAI's chat completions API
with a Bearer key, so this is plain httpx like the self-hosted provider,
not an SDK: LORE's key format, error mapping and schema handling stay its
own, and no extra dependency is added.

- Structured output: `response_format = {"type": "json_schema", "json_schema":
  {"name", "schema", "strict": true}}` with the same strict schema Claude and
  OpenAI get (`providers/schema.py`). OpenRouter passes it to the upstream
  model and not every model honours it, so EVERY reply is validated against
  the pydantic schema; an invalid one is asked again once with the reply and
  the validation error in the conversation, then reported as an output error
  like every provider's unusable reply.
- A model that refuses `json_schema` outright (an HTTP 400 naming it) is
  asked with `{"type": "json_object"}` and the schema in the prompt instead,
  and remembered for the rest of the process, exactly as the self-hosted
  provider does. So is a 404 "no endpoints support response_format".
- CASCADE: `models` is an ordered list of OpenRouter model ids. A 429 (rate
  limited), a 503 (upstream overloaded) or a 404 (model not available to
  this key) moves to the NEXT model rather than sleeping, the same rule as
  Gemini's cascade (`lore.inference.gemini`). Anything else fails at once.
  402 (out of credit) is final: nothing clears it by trying another model.
- Temperature 0, `max_tokens` from `[lore.llm] max_output_tokens` (default
  8192; 0 sends none).
- Each reply's `usage` tokens are reported like Claude's and OpenAI's; never
  a price (OpenRouter's own dashboard is the bill).
- The key comes from OPENROUTER_API_KEY or ~/.config/lore/openrouter.env
  (mode 600), never lore.toml, and only ever reaches the request header.
"""
from __future__ import annotations

import asyncio
import json
import time
import weakref
from collections.abc import Callable, Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from lore.inference.providers import roles
from lore.inference.providers.base import (
    OPENROUTER,
    Model,
    ProviderError,
    ProviderKeyError,
    ProviderOutputError,
    ProviderQuotaError,
    ProviderTransportError,
    ProviderUnavailableError,
)
from lore.inference.providers.openai_compatible import (
    JSON_INSTRUCTION,
    RETRY_INSTRUCTION,
    _clean,
    _format_name,
    _retry_after,
    _server_says,
    _short_error,
)
from lore.inference.providers.schema import decode, strict_schema

BASE_URL = "https://openrouter.ai/api/v1"
TIMEOUT_S = 1800.0
# How many models in a row may refuse for capacity reasons before the run is
# told the provider is out (a 429/503/404 on every one of them).
_MOVE_ON = (404, 429, 503)
# Per-model, per-loop request gates: one model answering slowly must not be
# asked `llm_concurrency` times at once on a free tier.
_GATES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Semaphore]]" \
    = weakref.WeakKeyDictionary()

Usage = Callable[[str, str, int, int], None]


def _gate(key: str, limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    gates = _GATES.setdefault(loop, {})
    if key not in gates:
        gates[key] = asyncio.Semaphore(max(1, limit))
    return gates[key]


class OpenRouterTextModel:
    """One OpenRouter model (or a short cascade of them) as a TextModel."""

    provider = OPENROUTER

    def __init__(self, api_key: str | None, *, model: str | None = None,
                 models: Sequence[str] = (), base_url: str = BASE_URL,
                 max_tokens: int = 8192, concurrency: int = 4,
                 transport: httpx.AsyncBaseTransport | None = None,
                 timeout: float = TIMEOUT_S, on_usage: Usage | None = None) -> None:
        order = tuple(models) or ((model,) if model else ())
        if not order:
            raise ValueError("an OpenRouter model must be named ([lore.llm] model)")
        if not api_key:
            raise ProviderKeyError(
                "no OpenRouter key: set OPENROUTER_API_KEY, or put one line"
                " `OPENROUTER_API_KEY=...` in ~/.config/lore/openrouter.env with"
                " mode 600.", provider=OPENROUTER)
        self._key = api_key
        # "" (no base_url anywhere) must fall back to OpenRouter's own: an
        # empty httpx base_url raises UnsupportedProtocol on every request.
        self._base_url = (base_url or BASE_URL).rstrip("/")
        self._concurrency = max(1, concurrency)
        self._max_tokens = max_tokens
        self.model = order[0]
        self.models = order
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {api_key}",
                     # OpenRouter's documented attribution headers.
                     "HTTP-Referer": "https://github.com/Jatin-Roopchandani/LORE",
                     "X-Title": "LORE"},
            timeout=timeout, transport=transport)
        self._on_usage = on_usage
        self.last_model: str | None = None
        self.requests_made = 0
        self.attempts: dict[str, int] = {}
        self.input_tokens = 0
        self.output_tokens = 0

    def __repr__(self) -> str:
        return f"OpenRouterTextModel({', '.join(self.models)!r})"

    # --- one request ------------------------------------------------------------

    def _error(self, response: httpx.Response, model: str) -> BaseException:
        status = response.status_code
        where = f"{model} at {self._base_url}"
        if status in (401, 403):
            return ProviderKeyError(
                f"HTTP {status}: OpenRouter refused the key (OPENROUTER_API_KEY"
                " or ~/.config/lore/openrouter.env).", provider=OPENROUTER)
        if status == 402:
            return ProviderQuotaError(
                f"HTTP 402 from {where}: your OpenRouter credit is used up."
                " Top up at openrouter.ai/credits.", provider=OPENROUTER, daily=True)
        if status == 429:
            wait = _retry_after(response)
            hint = f" (retry after {wait:g}s)" if wait is not None else ""
            return ProviderQuotaError(f"HTTP 429 from {where}: rate limited{hint}.",
                                      provider=OPENROUTER, retry_after=wait)
        if status >= 500:
            return ProviderUnavailableError(
                f"HTTP {status} from {where}: the upstream provider is overloaded"
                " or failing.", provider=OPENROUTER)
        return ProviderError(f"HTTP {status} from {where}: {_server_says(response)}",
                             provider=OPENROUTER)

    async def _post(self, model: str, messages: list[dict],
                    response_format: dict | None) -> httpx.Response:
        body: dict[str, Any] = {"model": model, "messages": messages,
                                "temperature": 0, "stream": False, "usage": {"include": True}}
        if self._max_tokens:
            body["max_tokens"] = self._max_tokens
        if response_format is not None:
            body["response_format"] = response_format
        self.requests_made += 1
        self.attempts[model] = self.attempts.get(model, 0) + 1
        key_scope.note_request(model)
        started = time.monotonic()
        try:
            async with _gate(f"{self._base_url}|{model}", self._concurrency):
                response = await self._client.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            error = ProviderTransportError(
                f"could not reach OpenRouter ({type(exc).__name__}). Is this"
                " machine online?", provider=OPENROUTER)
            tracing.note_attempt(model, ok=False,
                                 latency_s=time.monotonic() - started, error=error)
            raise error from None
        ok = response.status_code < 400
        tracing.note_attempt(model, ok=ok, latency_s=time.monotonic() - started,
                             error=None if ok else self._error(response, model))
        return response

    def _read(self, response: httpx.Response, model: str) -> str:
        try:
            payload = response.json()
            choice = payload["choices"][0]
            text = choice["message"].get("content") or ""
            if isinstance(text, list):  # content parts
                text = "".join(part.get("text", "") for part in text
                               if isinstance(part, dict))
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            self.last_model = model
            raise ProviderOutputError(f"{model} sent a reply that is not a chat"
                                      " completion", response.text[:2000], model,
                                      provider=OPENROUTER) from None
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if isinstance(usage, dict):
            tokens_in = int(usage.get("prompt_tokens") or 0)
            tokens_out = int(usage.get("completion_tokens") or 0)
            self.input_tokens += tokens_in
            self.output_tokens += tokens_out
            key_scope.note_usage(OPENROUTER, model, tokens_in, tokens_out)
            if self._on_usage is not None:
                self._on_usage(OPENROUTER, model, tokens_in, tokens_out)
        self.last_model = model
        if choice.get("finish_reason") == "length":
            raise ProviderOutputError(
                f"{model} stopped before finishing (length): LORE's output cap"
                " or the model's context is too small for this request", text, model,
                provider=OPENROUTER)
        if not _clean(text):
            raise ProviderOutputError(f"{model} returned an empty answer",
                                      text, model, provider=OPENROUTER)
        return text

    async def _chat(self, messages: list[dict],
                    response_format: dict | None) -> str:
        """One call through the cascade: the first model that answers wins."""
        last: BaseException | None = None
        for model in self.models:
            response = await self._post(model, messages, response_format)
            if response.status_code == 200:
                return self._read(response, model)
            error = self._error(response, model)
            if isinstance(error, ProviderKeyError):
                key_scope.note_rejection()
                raise error
            if response.status_code in _MOVE_ON:
                last = error
                continue
            raise error
        assert last is not None
        raise last

    # --- TextModel --------------------------------------------------------------

    async def generate(self, prompt: str, *, exhausted_hint: str | None = None) -> str:
        return _clean(await self._chat([{"role": "user", "content": prompt}], None))

    async def generate_json(self, prompt: str, schema: type[Model], *,
                            exhausted_hint: str | None = None,
                            avoid: str | None = None) -> Model:
        strict = strict_schema(schema)
        schema_text = json.dumps(strict)
        messages = [{"role": "user", "content": prompt}]
        fmt: dict | None = {
            "type": "json_schema",
            "json_schema": {"name": _format_name(schema.__name__), "schema": strict,
                            "strict": True}}
        if self.models and all(m in _NO_JSON_SCHEMA for m in self.models):
            fmt = None
        try:
            text = await self._chat(messages, fmt) if fmt is not None else ""
        except ProviderError as exc:
            # A model that cannot take a schema at all: ask in words instead
            # and remember it, so a book's every chunk does not pay for this.
            if not _schema_refused(exc):
                raise
            _NO_JSON_SCHEMA.update(self.models)
            fmt = None
        if fmt is None:
            messages = [{"role": "user",
                         "content": prompt + JSON_INSTRUCTION.format(schema=schema_text)}]
            fmt = {"type": "json_object"}
            text = await self._chat(messages, fmt)
        try:
            return decode(_clean(text), schema)
        except (ValidationError, json.JSONDecodeError) as exc:
            error = _short_error(exc)
        retry = [*messages, {"role": "assistant", "content": text},
                 {"role": "user", "content": RETRY_INSTRUCTION.format(
                     error=error, schema=schema_text)}]
        text = await self._chat(retry, fmt)
        try:
            return decode(_clean(text), schema)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise ProviderOutputError(
                f"{self.last_model or self.model} returned output that is not a"
                f" valid {schema.__name__} JSON object, twice ({_short_error(exc)})."
                f" Raw start: {text[:200]!r}",
                text, self.last_model or "", provider=OPENROUTER) from exc

    async def aclose(self) -> None:
        await self._client.aclose()


# Models whose upstream refused a `json_schema` response_format in this
# process; they are asked with `json_object` and the schema in the prompt.
_NO_JSON_SCHEMA: set[str] = set()


def _schema_refused(exc: BaseException) -> bool:
    """Whether a 400/404 is the upstream refusing `response_format` itself."""
    text = str(exc).lower()
    return "response_format" in text or "json_schema" in text or "json schema" in text


def chat_models(ids: list[str]) -> list[str]:
    """OpenRouter's general chat models, from its `/models` list."""
    return roles.openai_chat_models(ids)


def suggest_model(ids: list[str]) -> str | None:
    """A cheap general chat model from OpenRouter's list (a starting point)."""
    return roles.suggest_openai_model(ids)


from lore import tracing  # noqa: E402
from lore.inference import key_scope  # noqa: E402
