"""tool-call-salvage: catch tool calls a model wrote into its reply as text.

Some models and serving stacks return a tool call as text (``<tool_call>…``, ``<function=…>``,
DSML, Harmony, special tokens) instead of a structured ``tool_calls`` entry. The user sees the
markup and nothing runs.

* ``transform_llm_output`` (always): remove unambiguous leaked call markup that ends the final
  reply. Code blocks and markup followed by more text are never touched. When text is left
  before the markup, a one-line notice naming the tools says they were not run, so an announced
  action never reads as done.
* ``llm_execution`` middleware (``mode: salvage`` only, for models listed in ``salvage_models``):
  turn the leaked text of a Chat Completions response into structured tool calls, but only for
  tools offered in that request (or, with ``deferred_tools``, through Hermes's offered
  ``tool_call`` bridge), so Hermes runs them through its normal path and approvals.

No network, no files, no state, no Hermes imports. Settings are read on each call from
``plugins.entries.tool-call-salvage.settings``.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass
from typing import Any, Callable, Tuple

from . import salvage

logger = logging.getLogger(__name__)

DEFAULT_EMPTY_NOTICE = (
    "The model wrote a tool call as text instead of making it, so nothing was run. "
    "Please ask again.")
# ``{tools}`` becomes the tool names (never arguments). One line, appended after the kept text.
DEFAULT_STRIPPED_NOTICE = "(Not run: {tools}. The model wrote the call as text instead of making it.)"
MAX_NOTICE_NAMES = 8
MODES = ("strip", "salvage")


@dataclass(frozen=True)
class Settings:
    mode: str = "strip"
    salvage_models: Tuple[str, ...] = ()
    bare_json: bool = False
    deferred_tools: bool = False
    max_calls: int = 8
    empty_notice: str = DEFAULT_EMPTY_NOTICE
    stripped_notice: str = DEFAULT_STRIPPED_NOTICE


def read_settings(get: Callable[[str, Any], Any]) -> Settings:
    """Settings from ``ctx.get_config``; a value of the wrong type falls back to its default."""
    d = Settings()
    mode = get("mode", d.mode)
    models = get("salvage_models", list(d.salvage_models))
    bare = get("bare_json", d.bare_json)
    deferred = get("deferred_tools", d.deferred_tools)
    max_calls = get("max_calls", d.max_calls)
    notice = get("empty_notice", d.empty_notice)
    stripped = get("stripped_notice", d.stripped_notice)
    return Settings(
        mode=mode if mode in MODES else d.mode,
        salvage_models=tuple(m for m in models if isinstance(m, str) and m.strip())
        if isinstance(models, (list, tuple)) else d.salvage_models,
        bare_json=bare if isinstance(bare, bool) else d.bare_json,
        deferred_tools=deferred if isinstance(deferred, bool) else d.deferred_tools,
        max_calls=max(1, min(int(max_calls), 32))
        if isinstance(max_calls, int) and not isinstance(max_calls, bool) else d.max_calls,
        empty_notice=notice if isinstance(notice, str) else d.empty_notice,
        stripped_notice=stripped if isinstance(stripped, str) else d.stripped_notice,
    )


def model_listed(patterns: Tuple[str, ...], model: Any, provider: Any) -> bool:
    """Case-insensitive glob match on ``model`` or ``provider/model``."""
    model = str(model or "").lower()
    if not model:
        return False
    names = (model, f"{str(provider or '').lower()}/{model}")
    return any(fnmatch.fnmatchcase(n, p.strip().lower()) for p in patterns for n in names)


def _names(calls) -> str:
    """Tool names for the log (never arguments); a bridged call shows as ``tool_call:<name>``."""
    names = set()
    for c in calls:
        inner = c.arguments.get("calls") if c.name == salvage.BRIDGE_TOOL else None
        if isinstance(inner, list):
            names.update(f"tool_call:{i.get('name')}" for i in inner if isinstance(i, dict))
        else:
            names.add(c.name)
    return ", ".join(sorted(names))


def _user_names(calls) -> str:
    """Tool names for the user, in order, deduplicated; envelopes and bridge calls are unwrapped
    to the tools they name. Names are already restricted to ``[A-Za-z0-9_.-]``; never arguments."""
    names = []
    for c in calls:
        inner = salvage._unwrap_envelope(c)
        for one in inner if inner else [c]:
            if one.name not in names:
                names.append(one.name)
    if len(names) > MAX_NOTICE_NAMES:
        names = names[:MAX_NOTICE_NAMES] + ["…"]
    return ", ".join(names)


def stripped_notice(template: str, calls) -> str:
    """The notice for text that announced calls which were removed; ``""`` when disabled."""
    template = " ".join(template.split())  # always one line
    return template.replace("{tools}", _user_names(calls)) if template else ""


def make_output_hook(get: Callable[[str, Any], Any]):
    def tool_call_salvage_strip(response_text: Any = None, model: Any = None, **kwargs: Any):
        try:
            got = salvage.strip_leak(response_text)
            if got is None:
                return None
            kept, leak = got
            settings = read_settings(get)
            logger.warning(
                "tool-call-salvage: the reply ended with %d tool call(s) written as text (%s: %s); "
                "nothing ran. Removed the markup from the reply (model=%s).",
                len(leak.calls), "/".join(leak.formats), _names(leak.calls), model)
            if kept:
                notice = stripped_notice(settings.stripped_notice, leak.calls)
                return f"{kept}\n\n{notice}" if notice else kept
            if settings.empty_notice:
                return settings.empty_notice
            logger.warning("tool-call-salvage: empty_notice is empty, so the reply is left as it was")
            return None
        except Exception:  # never break a turn
            logger.debug("tool-call-salvage: reply not checked", exc_info=True)
            return None

    return tool_call_salvage_strip


def make_execution_middleware(get: Callable[[str, Any], Any]):
    def tool_call_salvage_execution(**kwargs: Any):
        request = kwargs.get("request")
        response = kwargs["next_call"](request)
        try:
            settings = read_settings(get)
            if settings.mode != "salvage":
                return response
            if not model_listed(settings.salvage_models, kwargs.get("model"), kwargs.get("provider")):
                return response
            got = salvage.salvage_chat_response(
                response, request, bare_json=settings.bare_json, max_calls=settings.max_calls,
                deferred_tools=settings.deferred_tools)
            if got is None:
                return response
            new_response, plan, source = got
            logger.warning(
                "tool-call-salvage: turned %d tool call(s) written as text in the %s into structured "
                "calls (%s: %s; model=%s).",
                len(plan.calls), source, "/".join(plan.formats), _names(plan.calls), kwargs.get("model"))
            return new_response
        except Exception:  # fail open: the provider response goes on unchanged
            logger.debug("tool-call-salvage: response not checked", exc_info=True)
            return response

    return tool_call_salvage_execution


def register(ctx) -> None:
    get = ctx.get_config
    ctx.register_hook("transform_llm_output", make_output_hook(get))
    ctx.register_middleware("llm_execution", make_execution_middleware(get))
