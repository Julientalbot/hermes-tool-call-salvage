"""Pure functions: find a leaked tool call at the end of a reply, strip it, or turn it into a
structured tool call on a Chat Completions response. No I/O, no state, no Hermes imports."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from . import formats
from .formats import Call

# Pseudo tool names some models use to batch real calls: ``<invoke name="tool_call">`` holding a
# JSON array of calls. Unwrapped only when no offered tool has that name.
ENVELOPE_NAMES = frozenset({"tool_call", "tool_calls", "function_calls"})
# Hermes's tool-search bridge: with Tool Search on, deferred tools are reached through
# ``tool_call({"calls": [{"name", "arguments"}]})`` (tools/tool_search.py).
BRIDGE_TOOL = "tool_call"
BRIDGE_NAMES = frozenset({"tool_search", "tool_describe", "tool_call"})

MAX_CANDIDATES = 64
_FENCE_LINE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})", re.MULTILINE)


@dataclass(frozen=True)
class Leak:
    """Leaked calls that run from ``start`` to the end of the text."""

    start: int
    calls: Tuple[Call, ...]
    formats: Tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    """What to put on the response: the text before the calls (may be empty) and the calls."""

    preamble: str
    calls: Tuple[Call, ...]
    formats: Tuple[str, ...]


def _fenced_spans(text: str) -> List[Tuple[int, int]]:
    """``(start, end)`` of every fenced code block; an unclosed fence runs to the end."""
    spans: List[Tuple[int, int]] = []
    opener: Optional[Tuple[int, str]] = None
    for m in _FENCE_LINE.finditer(text):
        mark = m.group(1)
        if opener is None:
            opener = (m.start(), mark)
        elif mark[0] == opener[1][0] and len(mark) >= len(opener[1]):
            spans.append((opener[0], m.end()))
            opener = None
    if opener is not None:
        spans.append((opener[0], len(text)))
    return spans


def _line_start(text: str, pos: int) -> int:
    return text.rfind("\n", 0, pos) + 1


def _may_start(text: str, pos: int, token: bool) -> bool:
    """Plain ASCII markup must open its own line (at most 3 spaces in, so not an indented code
    block). A special token may follow prose on the same line, but not in an indented code block."""
    line = text[_line_start(text, pos):pos]
    if line.startswith(("    ", "\t")):
        return False
    return token or (line.strip() == "" and len(line) <= 3)


def _read_tail(text: str, start: int, strict: bool) -> Optional[Tuple[List[Call], List[str]]]:
    """Read calls from ``start`` to the very end of ``text`` (whitespace between blocks only)."""
    calls: List[Call] = []
    fmts: List[str] = []
    i = start
    while True:
        for reader in formats.TAG_READERS + formats.TOKEN_READERS:
            got = reader(text, i, strict)
            if got is not None and got[0] > i:
                break
        else:
            return None
        i, more, fmt = got
        calls.extend(more)
        if fmt not in fmts:
            fmts.append(fmt)
        i = formats.skip_ws(text, i)
        if i == len(text):
            return calls, fmts


def find_trailing_leak(text: Any, *, strict: bool) -> Optional[Leak]:
    """The leaked tool-call markup that ends ``text``, or ``None``.

    Only markup that ends the reply counts: a model that emits a tool call stops there, while
    markup followed by more prose is an example or an explanation. Markup inside a fenced or
    indented code block is never read: that is how people show tool calls on purpose.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    fences = _fenced_spans(text)
    # Only the last openers can start a trailing leak; bounding them keeps a degenerate reply
    # (thousands of openers) linear instead of quadratic.
    candidates = sorted(
        [(m.start(), False) for m in formats.TAG_OPENER.finditer(text)]
        + [(m.start(), True) for m in formats.TOKEN_OPENER.finditer(text)])[-MAX_CANDIDATES:]
    seen = set()
    for pos, token in candidates:
        if pos in seen:
            continue
        seen.add(pos)
        if any(a <= pos < b for a, b in fences) or not _may_start(text, pos, token):
            continue
        got = _read_tail(text, pos, strict)
        if got is not None:
            return Leak(pos, tuple(got[0]), tuple(got[1]))
    return None


def strip_leak(text: Any) -> Optional[Tuple[str, Leak]]:
    """``(text_without_the_leak, leak)``; the remaining text may be empty. ``None`` = no change."""
    leak = find_trailing_leak(text, strict=False)
    if leak is None:
        return None
    return text[:leak.start].rstrip(), leak


# --- Turning text into structured calls ------------------------------------------------------

def offered_tools(request: Any) -> Dict[str, FrozenSet[str]]:
    """``{tool_name: required_parameter_names}`` from Chat Completions request kwargs."""
    tools = request.get("tools") if isinstance(request, Mapping) else None
    offered: Dict[str, FrozenSet[str]] = {}
    for tool in tools or ():
        fn = tool.get("function") if isinstance(tool, Mapping) else None
        if not isinstance(fn, Mapping) or not isinstance(fn.get("name"), str):
            continue
        params = fn.get("parameters")
        required = params.get("required") if isinstance(params, Mapping) else None
        offered[fn["name"]] = frozenset(r for r in (required if isinstance(required, list) else ()) if isinstance(r, str))
    return offered


def _unwrap_envelope(call: Call) -> Optional[List[Call]]:
    if call.name not in ENVELOPE_NAMES or len(call.arguments) != 1:
        return None
    (value,) = call.arguments.values()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return formats.calls_from_json(value)


def resolve(calls: Sequence[Call], offered: Mapping[str, FrozenSet[str]], max_calls: int, *,
            deferred_tools: bool = False) -> Optional[List[Call]]:
    """The calls to make, or ``None``. Every name must be offered in the request (with the tool's
    required arguments), or, with ``deferred_tools``, be sent through Hermes's offered
    ``tool_call`` bridge, which checks the name against the session's deferred tools itself.
    One call that fits neither refuses all of them: nothing is guessed."""
    bridge = deferred_tools and BRIDGE_TOOL in offered

    def place(call: Call) -> Optional[Call]:
        if call.name in offered:
            return call if offered[call.name] <= set(call.arguments) else None
        if bridge and call.name not in BRIDGE_NAMES:
            return Call(BRIDGE_TOOL, {"calls": [{"name": call.name, "arguments": call.arguments}]})
        return None

    out: List[Call] = []
    for call in calls:
        inner = _unwrap_envelope(call) if call.name not in offered else None
        for one in inner if inner is not None else [call]:
            placed = place(one)
            if placed is None:
                return None
            out.append(placed)
    if not out or len(out) > max_calls:
        return None
    return out


def plan_from_text(text: Any, offered: Mapping[str, FrozenSet[str]], *, bare_json: bool,
                   max_calls: int, deferred_tools: bool = False) -> Optional[Plan]:
    """How to turn ``text`` into structured calls, or ``None`` to leave it alone."""
    if not offered or not isinstance(text, str):
        return None
    leak = find_trailing_leak(text, strict=True)
    if leak is not None:
        calls = resolve(leak.calls, offered, max_calls, deferred_tools=deferred_tools)
        if calls:
            return Plan(text[:leak.start].rstrip(), tuple(calls), leak.formats)
        return None
    if bare_json:
        bare = formats.read_bare_json(text)
        calls = resolve(bare, offered, max_calls, deferred_tools=deferred_tools) if bare else None
        if calls:
            return Plan("", tuple(calls), ("bare_json",))
    return None


def _field(obj: Any, name: str) -> Any:
    """An attribute, falling back to pydantic's ``model_extra`` (where SDKs park unknown fields)."""
    value = getattr(obj, name, None)
    if value is None:
        extra = getattr(obj, "model_extra", None)
        if isinstance(extra, Mapping):
            value = extra.get(name)
    return value


def call_id(response_id: Any, name: str, arguments: str, index: int) -> str:
    """A deterministic call id (prompt caches key on replayed ids, so never random). The tool result
    is paired by this id, so it must be set here: Hermes does not fill a missing one before the tool
    runs. The provider's response id keeps two identical calls in one session apart."""
    seed = f"{response_id or ''}:{name}:{arguments}:{index}"
    return "call_" + hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()[:24]


def salvage_chat_response(response: Any, request: Any, *, bare_json: bool, max_calls: int,
                          deferred_tools: bool = False) -> Optional[Tuple[Any, Plan, str]]:
    """``(new_response, plan, source)`` when a Chat Completions response carries its tool calls as
    text, else ``None``. ``source`` is ``"content"`` or ``"reasoning"``. The provider's response
    object is never mutated: the response, choice and message are shallow copies."""
    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or len(choices) != 1:
        return None
    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None or getattr(message, "tool_calls", None):
        return None
    finish = str(getattr(choice, "finish_reason", "") or "").lower()
    if finish in ("length", "max_tokens", "content_filter"):
        return None  # a cut-off or filtered generation is never executed
    offered = offered_tools(request)
    if not offered:
        return None
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        plan = plan_from_text(content, offered, bare_json=bare_json, max_calls=max_calls,
                              deferred_tools=deferred_tools)
        source = "content"
    elif content is None or content == "":
        reasoning = _field(message, "reasoning_content") or _field(message, "reasoning")
        # A plan in the reasoning channel is taken only when it ends the reasoning and the reply
        # itself is empty: the model stalled with its call written there (#117090, #135008).
        plan = plan_from_text(reasoning, offered, bare_json=False, max_calls=max_calls,
                              deferred_tools=deferred_tools)
        source = "reasoning"
    else:
        return None
    if plan is None:
        return None
    new_message = copy.copy(message)
    new_message.tool_calls = []
    for index, call in enumerate(plan.calls):
        arguments = json.dumps(call.arguments, ensure_ascii=False)
        new_message.tool_calls.append(SimpleNamespace(
            id=call_id(getattr(response, "id", None), call.name, arguments, index), type="function",
            function=SimpleNamespace(name=call.name, arguments=arguments)))
    if source == "content":
        new_message.content = plan.preamble or None
    new_choice = copy.copy(choice)
    new_choice.message = new_message
    new_choice.finish_reason = "tool_calls"
    new_response = copy.copy(response)
    new_response.choices = [new_choice]
    return new_response, plan, source
