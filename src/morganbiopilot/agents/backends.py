"""Provider backends: what makes the multi-LLM axis of section 7 real.

`LLMPolicy` must not know which SDK it is talking to, otherwise "vary the LLM that
plays the policy" collapses into "vary the Claude model". A backend takes a system
prompt, a user prompt and a JSON schema, and returns text plus token counts. That
is the whole contract.

Model specs
-----------
Backends are selected by a spec string, so the model grid lives on the command
line rather than in code::

    anthropic:claude-opus-5
    openai:<model-id>                 # OpenAI-compatible endpoint

The `openai` form covers open-weight models served through OpenRouter, Together,
or a local vLLM: they all speak the same wire format. Base URL and key come from
`OPENAI_BASE_URL` and `OPENAI_API_KEY`, so pointing at a different host is a
config change, not a code change.

A caution for the paper
-----------------------
**`effort` is not a portable axis.** It is an Anthropic parameter with no
equivalent on an OpenAI-compatible endpoint, where it is accepted and ignored.
Sweeping effort across providers produces a grid whose cells are not comparable:
the Claude rows vary reasoning depth, the open-weight rows repeat the same run.
Either sweep effort within one provider and report it as a separate table, or
state plainly that the column is inapplicable elsewhere. `Backend.supports_effort`
exists so the experiment script can say so rather than quietly pretending.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

from morganbiopilot.core.paths import ROOT_DIR


@dataclass
class Completion:
    """What every backend returns, whatever the provider."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    refused: bool = False


class Backend(Protocol):
    """A model that can answer one structured question."""

    name: str
    supports_effort: bool

    def complete(self, system: str, prompt: str, schema: dict) -> Completion:
        ...


def _load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")


# Local OpenAI-compatible servers, in the order they are probed.
LOCAL_ENDPOINTS = (
    ("ollama", "http://localhost:11434/v1"),
    ("vLLM", "http://localhost:8000/v1"),
    ("LM Studio", "http://localhost:1234/v1"),
)


def detect_local_server() -> Optional[str]:
    """Base URL of a local OpenAI-compatible server, if one answers."""
    import urllib.error
    import urllib.request

    for _, url in LOCAL_ENDPOINTS:
        try:
            urllib.request.urlopen(f"{url}/models", timeout=2)
            return url
        except (urllib.error.URLError, OSError):
            continue
    return None


def resolve_base_url(explicit: Optional[str] = None) -> Optional[str]:
    """Where to send OpenAI-compatible requests. `None` means the SDK default.

    Resolution must live here, not in a single entry point. It used to sit in
    `agents.smoke` only, so every other caller — including the benchmark script —
    silently fell through to api.openai.com and failed with an
    ``AuthenticationError`` that said nothing about the real problem.

    Returning `None` when a real key is present is deliberate: OpenAI's own models
    are reached through the SDK's built-in endpoint, and forcing a base URL would
    make the hosted provider the one case this backend could not serve.
    """
    url = explicit or os.environ.get("OPENAI_BASE_URL")
    if url:
        return url

    local = detect_local_server()
    if local:
        return local

    # No endpoint configured and nothing local: fall back to the SDK's default
    # only if a real key exists to authenticate against it.
    if os.environ.get("OPENAI_API_KEY"):
        return None

    listed = "\n".join(f"    {n}: {u}" for n, u in LOCAL_ENDPOINTS)
    raise RuntimeError(
        "No OpenAI-compatible endpoint and no OPENAI_API_KEY. Either start a local "
        f"server (none answered on the usual ports):\n{listed}\n"
        "or set OPENAI_API_KEY for a hosted provider, optionally with "
        "OPENAI_BASE_URL to point elsewhere."
    )


class AnthropicBackend:
    """Claude, through the official SDK.

    Adaptive thinking is on, `effort` controls its depth, and sampling parameters
    are not sent — this model generation rejects `temperature`/`top_p`/`top_k`
    with a 400, so runs cannot be pinned and must be repeated.
    """

    supports_effort = True

    def __init__(
        self,
        model: str,
        effort: str = "medium",
        max_tokens: int = 16000,
        server_side_fallback: bool = True,
        client=None,
    ):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        # Safety classifiers can decline a request, and biology is one of the
        # categories — foreseeable on a biosynthesis project. Server-side fallback
        # re-runs a declined request on another model inside the same call, so a
        # refusal does not silently degrade the decision into a fallback.
        self.server_side_fallback = server_side_fallback

        if client is None:
            import anthropic

            _load_env()
            client = anthropic.Anthropic()
        self.client = client

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}|{self.effort}"

    def complete(self, system: str, prompt: str, schema: dict) -> Completion:
        request = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            # The system prompt is byte-identical across every decision of a run --
            # only the frontier changes -- so it is worth caching: reads bill at 0.1x.
            # It is ~1000 tokens on the fully tooled surface and well under the
            # minimum cacheable length on the untooled one, so the saving lands on the
            # arms that carry the most context and is simply ignored elsewhere.
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            messages=[{"role": "user", "content": prompt}],
        )

        if self.server_side_fallback:
            try:
                response = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                    **request
                )
            except Exception as exc:                             # noqa: BLE001
                # Not every model accepts `fallbacks`: claude-sonnet-5 rejects it with
                # a 400 saying so by name, which made every decision of a run fail
                # while the prompt itself was fine. An allowlist of models would rot,
                # so the beta switches itself off on that specific refusal -- once per
                # backend, not once per decision -- and the request is reissued on the
                # stable endpoint. Any other 400 is a real error and propagates.
                if not (_is_bad_request(exc) and "fallbacks" in str(exc)):
                    raise
                self.server_side_fallback = False
                response = self.client.messages.create(**request)
        else:
            response = self.client.messages.create(**request)

        usage = response.usage
        if response.stop_reason == "refusal":
            return Completion("", usage.input_tokens, usage.output_tokens, refused=True)

        text = next((b.text for b in response.content if b.type == "text"), "")
        return Completion(
            text=text,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )


def _is_bad_request(exc: Exception) -> bool:
    """True when the server refused the request itself, not the credentials."""
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status == 400
    return type(exc).__name__ in {"BadRequestError", "UnprocessableEntityError"}


class OpenAICompatibleBackend:
    """Any model behind an OpenAI-compatible endpoint — the open-weights path.

    Covers OpenRouter, Together, and a local vLLM server without code changes:
    they share the wire format. `effort` has no equivalent here and is ignored,
    which is why `supports_effort` is False.
    """

    supports_effort = False

    def __init__(
        self,
        model: str,
        # The answer is a schema-constrained JSON object -- an integer and one
        # sentence -- so a few hundred tokens is the real requirement and 1024
        # leaves generous headroom. Keeping it small also keeps the request valid
        # against a short context: a served model advertising 8192 tokens rejects
        # any `max_tokens` that does not fit alongside the prompt, whatever the
        # model would actually have generated. Raise it for a backend that emits
        # reasoning traces before the JSON.
        max_tokens: int = 1024,
        base_url: Optional[str] = None,
        client=None,
        strict_schema: bool = True,
        temperature: Optional[float] = 0.0,
        seed: Optional[int] = 0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        # Unlike the Claude generation used here, which rejects sampling
        # parameters outright, an open-weight served through vLLM accepts them.
        # Defaulting to temperature=0 with a fixed seed makes these rows of the
        # grid *reproducible* — a property the closed-model rows cannot have.
        # Set temperature=None to opt out and sample.
        self.temperature = temperature
        self.seed = seed
        # Not every OpenAI-compatible server implements json_schema response
        # formatting. When one rejects it we retry without and parse the text,
        # recording that the run was unconstrained.
        self.strict_schema = strict_schema
        self.schema_rejected = False

        if client is None:
            import openai

            _load_env()
            resolved = resolve_base_url(base_url)
            kwargs = {"api_key": os.environ.get("OPENAI_API_KEY", "unused")}
            if resolved:            # omit entirely to keep the SDK's own default
                kwargs["base_url"] = resolved
            client = openai.OpenAI(**kwargs)
        self.client = client

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def complete(self, system: str, prompt: str, schema: dict) -> Completion:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, messages=messages)
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed

        if self.strict_schema and not self.schema_rejected:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "choice", "schema": schema, "strict": True},
            }

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Only a *bad request* means the server rejected the schema. Auth,
            # connection and rate-limit errors must propagate: swallowing them
            # here permanently disabled structured outputs on the first transient
            # failure, and hid the real cause behind a formatting symptom.
            if "response_format" not in kwargs or not _is_bad_request(exc):
                raise
            self.schema_rejected = True
            kwargs.pop("response_format")
            response = self.client.chat.completions.create(**kwargs)

        usage = getattr(response, "usage", None)
        return Completion(
            text=response.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


def make_backend(spec: str, effort: str = "medium", **kwargs) -> Backend:
    """Build a backend from a spec string such as ``anthropic:claude-opus-5``.

    A bare model name is treated as Anthropic, so existing commands keep working.
    """
    provider, _, model = spec.partition(":")
    if not model:
        provider, model = "anthropic", provider

    if provider == "anthropic":
        return AnthropicBackend(model=model, effort=effort, **kwargs)
    if provider == "openai":
        kwargs.pop("server_side_fallback", None)
        return OpenAICompatibleBackend(model=model, **kwargs)
    raise ValueError(f"unknown provider {provider!r} in spec {spec!r}; "
                     "use 'anthropic:<model>' or 'openai:<model>'")


def parse_json_choice(text: str) -> Tuple[Optional[int], str]:
    """Extract ``choice`` / ``reason`` from a model answer.

    Tolerates a fenced code block, which unconstrained open-weight models emit
    routinely — without this, half the open-weight decisions would be recorded as
    parse failures and the comparison would measure output formatting rather than
    search quality.
    """
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1]
        if body.startswith("json"):
            body = body[4:]
        body = body.strip()

    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end == -1:
        return None, ""

    try:
        answer = json.loads(body[start:end + 1])
        return int(answer["choice"]), str(answer.get("reason", ""))[:400]
    except Exception:
        return None, ""
