"""Readers for tool calls that a model wrote into its reply as text.

Each reader is anchored: it is given the text and the index where an opener starts, and it
returns ``(end, calls, format_id)`` or ``None``. Nothing here searches loosely; the caller
(``salvage.find_trailing_leak``) decides where a reader may start.

``strict`` matters only for the XML-like families. Strict reading (used before a call is
executed) requires every ``</function>`` / ``</invoke>`` closer. Lenient reading (used to
remove markup from a reply) also accepts a block whose last closer is missing at the end of
the text: Hermes's own reply cleanup removes stray ``</function>`` closers before plugins see
the final reply. A parameter must always be closed, so a cut-off value is never read.

Formats, with the template or report they come from:

* ``tool_call``     ``<tool_call>{json}</tool_call>`` (Hermes / Qwen 2.5 chat templates),
                    ``<tool_call><function=…>`` (Qwen3-Coder), ``<minimax:tool_call><invoke …>``.
* ``function_tag``  ``<function=NAME><parameter=KEY>…</parameter></function>`` (Qwen3-Coder,
                    quantized Qwen in reasoning).
* ``invoke``        ``<invoke name="NAME"><parameter name="KEY">…</parameter></invoke>``, bare
                    or inside ``<function_calls>`` / ``<tool_calls>``, with an optional
                    ``ns:`` prefix or DeepSeek's ``｜DSML｜`` prefix.
* ``deepseek``      ``<｜tool▁calls▁begin｜>…<｜tool▁calls▁end｜>`` (DeepSeek V3 / V3.1).
* ``kimi``          ``<|tool_calls_section_begin|>…`` (Kimi K2).
* ``harmony``       ``<|channel|>commentary to=functions.NAME …<|message|>{json}<|call|>`` (gpt-oss).
* ``mistral``       ``[TOOL_CALLS][{json}]`` and ``[TOOL_CALLS]NAME[ARGS]{json}``.
* ``python_tag``    ``<|python_tag|>{json}`` (Llama 3.x).
* ``bare_json``     a reply that is nothing but ``{"name": …, "arguments": …}`` objects,
                    optionally in one code fence (Ollama ``/v1``). Read only on request.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# OpenAI-style function names, plus "." for namespaced MCP tools.
NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]{0,127}")
_DECODER = json.JSONDecoder()
_WS = re.compile(r"\s*")


@dataclass(frozen=True)
class Call:
    """One tool call read from text. ``arguments`` is always a JSON object."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


Parsed = Tuple[int, List[Call], str]
Reader = Callable[[str, int, bool], Optional[Parsed]]


def skip_ws(text: str, i: int) -> int:
    """Index of the first non-whitespace character at or after ``i``."""
    m = _WS.match(text, i)
    return m.end() if m else i


def _json_at(text: str, i: int) -> Optional[Tuple[Any, int]]:
    i = skip_ws(text, i)
    try:
        return _DECODER.raw_decode(text, i)
    except ValueError:
        return None


def _valid_name(name: Any) -> Optional[str]:
    if not isinstance(name, str):
        return None
    name = name.strip()
    return name if NAME_RE.fullmatch(name) else None


def _as_arguments(raw: Any) -> Optional[Dict[str, Any]]:
    """Arguments as a dict: a dict, a JSON string holding one, or nothing (= ``{}``)."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    return raw if isinstance(raw, dict) else None


def call_from_object(obj: Any) -> Optional[Call]:
    """A call from one JSON object, or ``None``.

    Accepted shapes, with no other keys (an extra key means it is data, not a call):
    ``{"name", "arguments"|"parameters", "type": "function"?, "id"?}`` and
    ``{"type": "function"?, "id"?, "function": {"name", "arguments"?}}``.
    """
    if not isinstance(obj, dict):
        return None
    if obj.get("type", "function") != "function":
        return None
    if "function" in obj:
        inner = obj["function"]
        if set(obj) - {"function", "type", "id"} or not isinstance(inner, dict):
            return None
        if set(inner) - {"name", "arguments"}:
            return None
        name, raw_args = inner.get("name"), inner.get("arguments")
    else:
        if set(obj) - {"name", "arguments", "parameters", "type", "id"}:
            return None
        if "arguments" in obj and "parameters" in obj:
            return None
        name, raw_args = obj.get("name"), obj.get("arguments", obj.get("parameters"))
    name = _valid_name(name)
    args = _as_arguments(raw_args)
    if name is None or args is None:
        return None
    return Call(name, args)


def calls_from_json(value: Any) -> Optional[List[Call]]:
    """Calls from a JSON object or a non-empty array of objects; ``None`` if any item is not a call."""
    items = value if isinstance(value, list) else [value]
    if not items:
        return None
    calls = [call_from_object(item) for item in items]
    return None if any(c is None for c in calls) else calls  # type: ignore[return-value]


# --- XML-like families -----------------------------------------------------------------------

# Namespace prefix (``minimax:``, ``antml:``) or DeepSeek's DSML prefix with full-width bars.
_PFX = r"(?:[A-Za-z][\w.-]*:|｜[A-Za-z]+｜)?"

_FUNC_OPEN = re.compile(r'<function(?:=(?P<n1>[^>\s]+)|\s+name\s*=\s*"(?P<n2>[^"]+)")\s*>')
_FUNC_PARAM_OPEN = re.compile(r'<parameter(?:=(?P<k1>[^>\s]+)|\s+name\s*=\s*"(?P<k2>[^"]+)")\s*>')
_FUNC_CLOSE, _FUNC_PARAM_CLOSE = "</function>", "</parameter>"

_INVOKE_OPEN = re.compile(rf'<(?P<p>{_PFX})invoke\s+name\s*=\s*"(?P<n>[^"]+)"\s*>')
_INVOKE_PARAM_OPEN = re.compile(
    rf'<(?P<p>{_PFX})parameter\s+name\s*=\s*"(?P<k>[^"]+)"(?P<attrs>(?:\s+[\w-]+\s*=\s*"[^"]*")*)\s*>')
_STRING_ATTR = re.compile(r'\bstring\s*=\s*"(true|false)"')

_WRAPPER_OPEN = re.compile(rf"<(?P<tag>{_PFX}(?:tool_calls?|function_calls))>")


def _function_tag(text: str, i: int, strict: bool) -> Optional[Parsed]:
    """``<function=NAME>`` / ``<function name="NAME">`` with ``<parameter=KEY>`` children."""
    m = _FUNC_OPEN.match(text, i)
    if not m:
        return None
    name = _valid_name(m.group("n1") or m.group("n2"))
    if name is None:
        return None
    args: Dict[str, Any] = {}
    j = m.end()
    while True:
        j = skip_ws(text, j)
        if text.startswith(_FUNC_CLOSE, j):
            return j + len(_FUNC_CLOSE), [Call(name, args)], "function_tag"
        if j == len(text):
            return None if strict else (j, [Call(name, args)], "function_tag")
        pm = _FUNC_PARAM_OPEN.match(text, j)
        if not pm:
            return None
        key = (pm.group("k1") or pm.group("k2") or "").strip()
        end = text.find(_FUNC_PARAM_CLOSE, pm.end())
        if not key or key in args or end < 0:
            return None
        value = text[pm.end():end]
        if _FUNC_PARAM_OPEN.search(value) or _FUNC_CLOSE in value:
            return None  # an earlier parameter was never closed
        # The Qwen3-Coder template puts each value on its own lines.
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        args[key] = value
        j = end + len(_FUNC_PARAM_CLOSE)


def _invoke(text: str, i: int, strict: bool) -> Optional[Parsed]:
    """``<invoke name="NAME">`` with ``<parameter name="KEY">`` children (optional prefix)."""
    m = _INVOKE_OPEN.match(text, i)
    if not m:
        return None
    name = _valid_name(m.group("n"))
    if name is None:
        return None
    prefix = m.group("p")
    close = f"</{prefix}invoke>"
    args: Dict[str, Any] = {}
    j = m.end()
    while True:
        j = skip_ws(text, j)
        if text.startswith(close, j):
            return j + len(close), [Call(name, args)], "invoke"
        if j == len(text):
            return None if strict else (j, [Call(name, args)], "invoke")
        pm = _INVOKE_PARAM_OPEN.match(text, j)
        if not pm or pm.group("p") != prefix:
            return None
        key = pm.group("k").strip()
        param_close = f"</{prefix}parameter>"
        end = text.find(param_close, pm.end())
        if not key or key in args or end < 0:
            return None
        value: Any = text[pm.end():end]
        if _INVOKE_PARAM_OPEN.search(value) or close in value:
            return None  # an earlier parameter was never closed
        string_attr = _STRING_ATTR.search(pm.group("attrs") or "")
        if string_attr and string_attr.group(1) == "false":
            # DSML marks non-string values: they must be JSON.
            try:
                value = json.loads(value)
            except ValueError:
                return None
        args[key] = value
        j = end + len(param_close)


def _xml_blocks(text: str, i: int, strict: bool) -> Optional[Parsed]:
    """One or more ``<function=…>`` / ``<invoke …>`` blocks separated by whitespace."""
    calls: List[Call] = []
    fmt = ""
    j = i
    while True:
        got = _function_tag(text, j, strict) or _invoke(text, j, strict)
        if got is None:
            break
        j, more, fmt = got
        calls.extend(more)
        nxt = skip_ws(text, j)
        if not (_FUNC_OPEN.match(text, nxt) or _INVOKE_OPEN.match(text, nxt)):
            break
        j = nxt
    return (j, calls, fmt) if calls else None


def _wrapper(text: str, i: int, strict: bool) -> Optional[Parsed]:
    """``<tool_call>`` / ``<tool_calls>`` / ``<function_calls>`` (optionally prefixed) holding
    JSON call objects or XML call blocks. The closing tag may be missing at the end of the text
    (it is often a stop token); the calls inside must still be complete."""
    m = _WRAPPER_OPEN.match(text, i)
    if not m:
        return None
    close = f"</{m.group('tag')}>"
    j = skip_ws(text, m.end())
    got_json = _json_at(text, j) if text[j:j + 1] in ("{", "[") else None
    if got_json is not None:
        calls = calls_from_json(got_json[0])
        if not calls:
            return None
        k, fmt = got_json[1], "tool_call"
    else:
        inner = _xml_blocks(text, j, strict)
        if inner is None:
            return None
        k, calls, fmt = inner
        fmt = "tool_call" if fmt == "function_tag" else fmt
    k = skip_ws(text, k)
    if text.startswith(close, k):
        return k + len(close), calls, fmt
    return (k, calls, fmt) if k == len(text) else None


# --- Special-token families ------------------------------------------------------------------

_BAR = "[｜|]"
_DS_CALLS_BEGIN = re.compile(rf"<{_BAR}tool▁calls▁begin{_BAR}>")
_DS_CALL_BEGIN = re.compile(rf"\s*<{_BAR}tool▁call▁begin{_BAR}>")
_DS_SEP = re.compile(rf"<{_BAR}tool▁sep{_BAR}>")
_DS_CALL_END = re.compile(rf"\s*<{_BAR}tool▁call▁end{_BAR}>")
_DS_CALLS_END = re.compile(rf"\s*<{_BAR}tool▁calls▁end{_BAR}>")
_DS_FENCE_OPEN = re.compile(r"\s*```(?:json)?[ \t]*\n?")
_DS_FENCE_CLOSE = re.compile(r"\s*```")


def _deepseek(text: str, i: int, strict: bool) -> Optional[Parsed]:
    """V3: ``<｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME\\n```json\\n{…}\\n```<｜tool▁call▁end｜>``;
    V3.1: ``<｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{…}<｜tool▁call▁end｜>``."""
    m = _DS_CALLS_BEGIN.match(text, i)
    if not m:
        return None
    calls: List[Call] = []
    j = m.end()
    while True:
        cb = _DS_CALL_BEGIN.match(text, j)
        if not cb:
            break
        sep = _DS_SEP.search(text, cb.end())
        if not sep:
            return None
        head = text[cb.end():sep.start()].strip()
        k = sep.end()
        if head == "function":  # V3: the name follows the separator, then a fenced JSON block
            nl = text.find("\n", k)
            if nl < 0:
                return None
            name = _valid_name(text[k:nl])
            fence = _DS_FENCE_OPEN.match(text, nl)
            k = fence.end() if fence else nl
        else:
            name, fence = _valid_name(head), None
        got = _json_at(text, k)
        if name is None or got is None:
            return None
        args = _as_arguments(got[0])
        if args is None:
            return None
        k = got[1]
        if fence:
            fc = _DS_FENCE_CLOSE.match(text, k)
            if not fc:
                return None
            k = fc.end()
        end = _DS_CALL_END.match(text, k)
        if not end:
            if strict or skip_ws(text, k) != len(text):
                return None
            calls.append(Call(name, args))
            return len(text), calls, "deepseek"
        calls.append(Call(name, args))
        j = end.end()
    if not calls:
        return None
    end_all = _DS_CALLS_END.match(text, j)
    if end_all:
        return end_all.end(), calls, "deepseek"
    return (j, calls, "deepseek") if skip_ws(text, j) == len(text) else None


_KIMI_SECTION_BEGIN = re.compile(r"<\|tool_calls_section_begin\|>")
_KIMI_CALL = re.compile(
    r"\s*<\|tool_call_begin\|>\s*(?:functions\.)?(?P<n>[A-Za-z0-9_.\-]+?)(?::\d+)?\s*"
    r"<\|tool_call_argument_begin\|>")
_KIMI_CALL_END = re.compile(r"\s*<\|tool_call_end\|>")
_KIMI_SECTION_END = re.compile(r"\s*<\|tool_calls_section_end\|>")


def _kimi(text: str, i: int, strict: bool) -> Optional[Parsed]:
    """``<|tool_calls_section_begin|><|tool_call_begin|>functions.NAME:0<|tool_call_argument_begin|>
    {…}<|tool_call_end|><|tool_calls_section_end|>``."""
    m = _KIMI_SECTION_BEGIN.match(text, i)
    if not m:
        return None
    calls: List[Call] = []
    j = m.end()
    while True:
        cm = _KIMI_CALL.match(text, j)
        if not cm:
            break
        name, got = _valid_name(cm.group("n")), _json_at(text, cm.end())
        if name is None or got is None or (args := _as_arguments(got[0])) is None:
            return None
        end = _KIMI_CALL_END.match(text, got[1])
        if not end:
            return None
        calls.append(Call(name, args))
        j = end.end()
    if not calls:
        return None
    end_all = _KIMI_SECTION_END.match(text, j)
    if end_all:
        return end_all.end(), calls, "kimi"
    return (j, calls, "kimi") if skip_ws(text, j) == len(text) else None


_HARMONY = re.compile(
    r"(?:<\|start\|>assistant)?"
    r"(?:[ \t]*to=functions\.(?P<n1>[A-Za-z0-9_.\-]+))?"
    r"[ \t]*<\|channel\|>[ \t]*commentary"
    r"(?:[ \t]+to=functions\.(?P<n2>[A-Za-z0-9_.\-]+))?"
    r"[ \t]*(?:<\|constrain\|>)?[ \t]*(?:json)?[ \t]*<\|message\|>")
_HARMONY_CALL = re.compile(r"\s*<\|call\|>")


def _harmony(text: str, i: int, strict: bool) -> Optional[Parsed]:
    """gpt-oss: ``<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{…}<|call|>``,
    also with ``<|start|>assistant`` and the recipient before the channel. ``<|call|>`` is a stop
    token, so it may be missing at the end of the text."""
    m = _HARMONY.match(text, i)
    if not m or bool(m.group("n1")) == bool(m.group("n2")):
        return None
    name = _valid_name(m.group("n1") or m.group("n2"))
    got = _json_at(text, m.end())
    if name is None or got is None or (args := _as_arguments(got[0])) is None:
        return None
    call_end = _HARMONY_CALL.match(text, got[1])
    return (call_end.end() if call_end else got[1]), [Call(name, args)], "harmony"


_MISTRAL = re.compile(r"\[TOOL_CALLS\]")
_MISTRAL_NAMED = re.compile(
    r"\s*(?P<n>[A-Za-z0-9_.\-]+)\s*(?:\[CALL_ID\]\s*[A-Za-z0-9_\-]+\s*)?\[ARGS\]")


def _mistral(text: str, i: int, strict: bool) -> Optional[Parsed]:
    """``[TOOL_CALLS][{"name": …, "arguments": …}]`` or ``[TOOL_CALLS]NAME[ARGS]{…}`` (repeatable)."""
    m = _MISTRAL.match(text, i)
    if not m:
        return None
    j = m.end()
    named = _MISTRAL_NAMED.match(text, j)
    if named:
        name, got = _valid_name(named.group("n")), _json_at(text, named.end())
        if name is None or got is None or (args := _as_arguments(got[0])) is None:
            return None
        return got[1], [Call(name, args)], "mistral"
    got = _json_at(text, j)
    calls = calls_from_json(got[0]) if got else None
    return (got[1], calls, "mistral") if got and calls else None


_PYTHON_TAG = re.compile(r"<\|python_tag\|>")
_LLAMA_END = re.compile(r"\s*<\|(?:eom_id|eot_id)\|>")


def _python_tag(text: str, i: int, strict: bool) -> Optional[Parsed]:
    """Llama 3.x: ``<|python_tag|>{"name": …, "parameters": …}`` (``;``-separated), then an
    optional ``<|eom_id|>``."""
    m = _PYTHON_TAG.match(text, i)
    if not m:
        return None
    calls: List[Call] = []
    j = m.end()
    while True:
        got = _json_at(text, j)
        if got is None or not (more := calls_from_json(got[0])):
            return None
        calls.extend(more)
        j = skip_ws(text, got[1])
        if not text.startswith(";", j):
            break
        j += 1
    end = _LLAMA_END.match(text, j)
    return (end.end() if end else j), calls, "python_tag"


# Readers in the order they are tried, keyed by whether their opener is plain ASCII markup
# (``tag``: must start a line) or a special token nobody types in prose (``token``).
TAG_READERS: Tuple[Reader, ...] = (_wrapper, _function_tag, _invoke)
TOKEN_READERS: Tuple[Reader, ...] = (_deepseek, _kimi, _harmony, _mistral, _python_tag, _wrapper, _invoke)

# Where a reader may start. ``tag`` openers are plain ASCII; ``token`` openers contain a
# special-token marker (``<|…|>``, ``｜``, ``▁``, ``[TOOL_CALLS]``).
TAG_OPENER = re.compile(
    r"<(?:[A-Za-z][\w.-]*:)?(?:tool_calls?|function_calls)>|<function[=\s]|<(?:[A-Za-z][\w.-]*:)?invoke\s")
TOKEN_OPENER = re.compile(
    r"<｜[A-Za-z]+｜(?:tool_calls?|function_calls|invoke)\b|<[｜|]tool▁calls▁begin[｜|]>"
    r"|<\|tool_calls_section_begin\|>|<\|start\|>assistant|<\|channel\|>|\[TOOL_CALLS\]|<\|python_tag\|>")


def read_bare_json(text: str) -> Optional[List[Call]]:
    """Calls from a reply that is only JSON call objects (several may follow one another), or the
    same inside one code fence spanning the whole reply. Any other text means ``None``."""
    body = text.strip()
    fence = re.fullmatch(r"```[ \t]*(?:json)?[ \t]*\n(?P<b>.*?)\n?[ \t]*(?:```)?", body, re.DOTALL)
    if fence:
        body = fence.group("b")
    calls: List[Call] = []
    i = 0
    while True:
        i = skip_ws(body, i)
        if i == len(body):
            return calls or None
        if body[i] not in "{[":
            return None
        got = _json_at(body, i)
        if got is None or not (more := calls_from_json(got[0])):
            return None
        calls.extend(more)
        i = got[1]
