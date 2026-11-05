"""Unit tests for reading, stripping and salvaging leaked tool calls."""

import json
import logging
from types import SimpleNamespace

import pytest

import tool_call_salvage as plugin
from tool_call_salvage import salvage
from corpus import BARE_JSON, FALSE_POSITIVES, LEAKS


def _pairs(calls):
    return [(c.name, c.arguments) for c in calls]


def _tool(name, required=()):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {
        "type": "object", "properties": {k: {"type": "string"} for k in required}, "required": list(required)}}}


def _request(*tools):
    return {"model": "m", "messages": [], "tools": list(tools)}


def _chat(content, *, reasoning=None, finish="stop", tool_calls=None):
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls, reasoning_content=reasoning)
    return SimpleNamespace(id="r1", model="m", usage=None,
                           choices=[SimpleNamespace(index=0, message=message, finish_reason=finish)])


ALL_TOOLS = _request(*(_tool(n) for n in (
    "terminal", "read_file", "execute_code", "web_search", "get_weather", "get_time")))


# --- reading -----------------------------------------------------------------------------------

@pytest.mark.parametrize("case", LEAKS, ids=[c["id"] for c in LEAKS])
def test_strict_reading_returns_the_calls(case):
    leak = salvage.find_trailing_leak(case["text"], strict=True)
    assert leak is not None, case["source"]
    assert _pairs(leak.calls) == case["calls"]


@pytest.mark.parametrize("case", LEAKS, ids=[c["id"] for c in LEAKS])
def test_strip_keeps_only_the_text_before_the_calls(case):
    kept, _ = salvage.strip_leak(case["text"])
    assert kept == case["kept"]


@pytest.mark.parametrize("case", FALSE_POSITIVES + BARE_JSON, ids=[c["id"] for c in FALSE_POSITIVES + BARE_JSON])
def test_strip_leaves_legitimate_text_and_bare_json_alone(case):
    assert salvage.strip_leak(case["text"]) is None


@pytest.mark.parametrize("case", FALSE_POSITIVES, ids=[c["id"] for c in FALSE_POSITIVES])
def test_salvage_never_converts_a_false_positive(case):
    response = _chat(case["text"])
    assert salvage.salvage_chat_response(response, ALL_TOOLS, bare_json=True, max_calls=8) is None


@pytest.mark.parametrize("case", BARE_JSON, ids=[c["id"] for c in BARE_JSON])
def test_bare_json_is_read_only_when_enabled(case):
    assert salvage.plan_from_text(case["text"], salvage.offered_tools(ALL_TOOLS), bare_json=False, max_calls=8) is None
    plan = salvage.plan_from_text(case["text"], salvage.offered_tools(ALL_TOOLS), bare_json=True, max_calls=8)
    assert plan is not None and _pairs(plan.calls) == case["calls"]


def test_lenient_reading_accepts_a_closer_removed_by_hermes_cleanup():
    # Hermes's reply cleanup drops stray </function> closers before transform_llm_output runs.
    text = "Listing.\n<function=terminal>\n<parameter=command>\nls\n</parameter>\n"
    assert salvage.find_trailing_leak(text, strict=True) is None
    assert salvage.strip_leak(text)[0] == "Listing."


def test_a_degenerate_reply_is_read_quickly():
    import time
    # Thousands of unclosed blocks at line starts: each would rescan the rest of the reply.
    text = '<invoke name="x">\n<parameter name="y">\n' * 20_000 + "\n" + LEAKS[0]["text"]
    started = time.monotonic()
    assert salvage.strip_leak(text)[0].endswith("I will list the directory.")
    assert time.monotonic() - started < 1.0  # about 4 s without the candidate bound


# --- salvaging a Chat Completions response -------------------------------------------------------

def test_salvage_rewrites_a_copy_of_the_response_into_structured_calls():
    original = _chat('Checking.\n<tool_call>\n{"name": "web_search", "arguments": {"query": "x"}}\n</tool_call>')
    got = salvage.salvage_chat_response(original, _request(_tool("web_search", ["query"])), bare_json=False, max_calls=8)
    assert got is not None
    new, plan, source = got
    choice = new.choices[0]
    assert source == "content" and choice.finish_reason == "tool_calls"
    assert choice.message.content == "Checking."
    (call,) = choice.message.tool_calls
    assert call.type == "function"
    assert call.id == salvage.call_id("r1", "web_search", call.function.arguments, 0)  # deterministic
    assert call.id.startswith("call_") and len(call.id) == 29
    assert call.function.name == "web_search" and json.loads(call.function.arguments) == {"query": "x"}
    # The provider's object is untouched.
    assert original.choices[0].message.tool_calls is None
    assert original.choices[0].finish_reason == "stop"


def test_salvage_unwraps_an_envelope_into_the_offered_tools():
    case = next(c for c in LEAKS if c["id"] == "envelope")
    new, plan, _ = salvage.salvage_chat_response(_chat(case["text"]), ALL_TOOLS, bare_json=False, max_calls=8)
    calls = new.choices[0].message.tool_calls
    assert [c.function.name for c in calls] == ["terminal", "web_search"]
    assert len({c.id for c in calls}) == 2  # each call gets its own id


def test_deferred_tools_go_through_the_offered_bridge_only_when_enabled():
    request = _request(_tool("tool_search", ["queries"]), _tool("tool_call", ["calls"]), _tool("terminal"))
    text = '<function=todo_list>\n</function>\n<function=terminal>\n<parameter=command>\nls\n</parameter>\n</function>'
    assert salvage.salvage_chat_response(_chat(text), request, bare_json=False, max_calls=8) is None
    new, _, _ = salvage.salvage_chat_response(_chat(text), request, bare_json=False, max_calls=8,
                                              deferred_tools=True)
    calls = [(c.function.name, json.loads(c.function.arguments)) for c in new.choices[0].message.tool_calls]
    assert calls == [("tool_call", {"calls": [{"name": "todo_list", "arguments": {}}]}),
                     ("terminal", {"command": "ls"})]
    # Without the bridge in the request there is nothing to route through.
    assert salvage.salvage_chat_response(_chat(text), ALL_TOOLS, bare_json=False, max_calls=8,
                                         deferred_tools=True) is None


def test_salvage_reads_a_call_left_in_the_reasoning_when_the_reply_is_empty():
    reasoning = "I need the date.\n<function=terminal>\n<parameter=command>\ndate\n</parameter>\n</function>"
    new, plan, source = salvage.salvage_chat_response(_chat(None, reasoning=reasoning), ALL_TOOLS,
                                                      bare_json=False, max_calls=8)
    assert source == "reasoning"
    assert new.choices[0].message.reasoning_content == reasoning  # reasoning is kept as is
    assert new.choices[0].message.content is None
    assert new.choices[0].message.tool_calls[0].function.name == "terminal"


@pytest.mark.parametrize("label,response,request_", [
    ("unknown tool", _chat('<tool_call>{"name": "delete_everything", "arguments": {}}</tool_call>'), ALL_TOOLS),
    ("missing required argument", _chat('<tool_call>{"name": "web_search", "arguments": {}}</tool_call>'),
     _request(_tool("web_search", ["query"]))),
    ("no tools offered", _chat('<tool_call>{"name": "web_search", "arguments": {}}</tool_call>'), _request()),
    ("already has structured calls", _chat('<tool_call>{"name": "web_search", "arguments": {}}</tool_call>',
                                           tool_calls=[SimpleNamespace(id="c1")]), ALL_TOOLS),
    ("cut off by max tokens", _chat('<tool_call>{"name": "web_search", "arguments": {}}</tool_call>',
                                    finish="length"), ALL_TOOLS),
    ("missing invoke closer", _chat('<invoke name="terminal">\n<parameter name="command">ls</parameter>\n'),
     ALL_TOOLS),
    ("too many calls", _chat("".join('<tool_call>{"name": "get_time"}</tool_call>\n' for _ in range(9))), ALL_TOOLS),
    ("reasoning leak but visible reply", _chat("Here you go.", reasoning='<tool_call>{"name": "get_time"}</tool_call>'),
     ALL_TOOLS),
])
def test_salvage_refuses(label, response, request_):
    assert salvage.salvage_chat_response(response, request_, bare_json=False, max_calls=8) is None, label


def test_salvage_ignores_other_response_shapes():
    responses_api = SimpleNamespace(output=[SimpleNamespace(type="message", content=[])])
    anthropic = SimpleNamespace(content=[SimpleNamespace(type="text", text="<tool_call>{}</tool_call>")])
    for response in (responses_api, anthropic, None):
        assert salvage.salvage_chat_response(response, ALL_TOOLS, bare_json=False, max_calls=8) is None


def test_salvage_works_on_openai_sdk_objects():
    types = pytest.importorskip("openai.types.chat")
    response = types.ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": '<tool_call>{"name": "get_time", "arguments": {}}</tool_call>'}}]})
    new, _, _ = salvage.salvage_chat_response(response, ALL_TOOLS, bare_json=False, max_calls=8)
    assert new.choices[0].message.tool_calls[0].function.name == "get_time"
    assert new.choices[0].message.content is None
    assert response.choices[0].message.content.startswith("<tool_call>")


# --- plugin wiring ---------------------------------------------------------------------------------

def _getter(**settings):
    return lambda key, default=None: settings.get(key, default)


def test_settings_fall_back_to_safe_defaults():
    s = plugin.read_settings(_getter(mode="execute-everything", salvage_models="qwen*", bare_json="yes",
                                     max_calls=10_000, empty_notice=None, stripped_notice=3))
    assert s == plugin.Settings(max_calls=32)
    assert plugin.read_settings(_getter()) == plugin.Settings()


def test_model_listing_matches_model_or_provider_slash_model():
    assert plugin.model_listed(("qwen*",), "Qwen3-Coder-30B", "custom")
    assert plugin.model_listed(("ollama/*",), "llama3.1:8b", "ollama")
    assert not plugin.model_listed((), "qwen3", "custom")
    assert not plugin.model_listed(("qwen*",), "", "custom")


def test_output_hook_strips_and_logs(caplog):
    hook = plugin.make_output_hook(_getter())
    with caplog.at_level(logging.WARNING):
        out = hook(response_text=LEAKS[0]["text"], session_id="s", model="m", platform="cli")
    assert out.startswith(LEAKS[0]["kept"] + "\n\n")
    assert "terminal" in caplog.text and "ls -la" not in caplog.text  # names logged, arguments never


def test_output_hook_replaces_a_reply_that_was_only_markup_with_the_notice():
    hook = plugin.make_output_hook(_getter())
    assert hook(response_text='<invoke name="x">\n</invoke>') == plugin.DEFAULT_EMPTY_NOTICE
    assert plugin.make_output_hook(_getter(empty_notice=""))(response_text='<invoke name="x">\n</invoke>') is None


# --- the notice when text is left before stripped markup -------------------------------------

def test_text_before_stripped_markup_gets_a_notice_naming_the_tool():
    out = plugin.make_output_hook(_getter())(response_text=LEAKS[0]["text"])
    assert out == "I will list the directory.\n\n(Not run: terminal. The model wrote the call as text instead of making it.)"
    assert out.count("\n") == 2  # the notice is one line, on its own paragraph


def test_a_reply_that_was_only_markup_gets_the_empty_notice_only():
    hook = plugin.make_output_hook(_getter(stripped_notice="STRIPPED {tools}"))
    out = hook(response_text='<invoke name="x">\n</invoke>')
    assert out == plugin.DEFAULT_EMPTY_NOTICE and "STRIPPED" not in out
    assert plugin.make_output_hook(_getter(empty_notice="", stripped_notice="S {tools}"))(
        response_text='<invoke name="x">\n</invoke>') is None


def test_an_empty_stripped_notice_keeps_the_0_1_behaviour():
    out = plugin.make_output_hook(_getter(stripped_notice=""))(response_text=LEAKS[0]["text"])
    assert out == LEAKS[0]["kept"]


@pytest.mark.parametrize("case", [c for c in LEAKS if c["kept"].strip()], ids=lambda c: c["id"])
def test_every_kept_text_gets_the_notice_and_never_the_arguments(case):
    names = []
    for name, arguments in case["calls"]:
        inner = salvage._unwrap_envelope(salvage.Call(name, arguments))
        names += [c.name for c in inner or [salvage.Call(name, arguments)] if c.name not in names]
    out = plugin.make_output_hook(_getter())(response_text=case["text"])
    # Exactly the kept text and the template with the names: no argument can reach the user.
    assert out == case["kept"] + "\n\n" + plugin.DEFAULT_STRIPPED_NOTICE.replace("{tools}", ", ".join(names))


def test_the_notice_lists_several_tools_once_each_in_order():
    text = ("Reading both.\n<function=read_file>\n<parameter=path>/etc/secret\n</parameter>\n</function>\n"
            "<function=terminal>\n<parameter=command>rm -rf /tmp/x\n</parameter>\n</function>\n"
            "<function=read_file>\n<parameter=path>/b\n</parameter>\n</function>")
    out = plugin.make_output_hook(_getter(stripped_notice="[{tools}]"))(response_text=text)
    assert out == "Reading both.\n\n[read_file, terminal]"


def test_the_notice_names_the_tools_inside_an_envelope():
    envelope = next(c for c in LEAKS if c["id"] == "envelope")["text"]
    out = plugin.make_output_hook(_getter(stripped_notice="{tools}"))(response_text="On it.\n" + envelope)
    assert out == "On it.\n\nterminal, web_search"


def test_a_custom_notice_without_placeholder_or_on_several_lines_stays_one_line():
    hook = plugin.make_output_hook(_getter(stripped_notice="Nothing\nran."))
    assert hook(response_text=LEAKS[0]["text"]) == LEAKS[0]["kept"] + "\n\nNothing ran."


def test_a_long_list_of_tools_is_cut():
    text = "Many.\n" + "".join(f'<tool_call>{{"name": "t{i}"}}</tool_call>\n' for i in range(12))
    out = plugin.make_output_hook(_getter(stripped_notice="{tools}"))(response_text=text)
    assert out == "Many.\n\n" + ", ".join(f"t{i}" for i in range(8)) + ", …"


def test_salvaged_calls_leave_no_markup_so_the_hook_adds_no_notice():
    text = 'Checking.\n<tool_call>\n{"name": "web_search", "arguments": {"query": "x"}}\n</tool_call>'
    settings = {"mode": "salvage", "salvage_models": ["qwen*"]}
    out = _run_middleware(settings, _chat(text))
    message = out.choices[0].message
    assert message.tool_calls and message.content == "Checking."
    assert plugin.make_output_hook(_getter(**settings))(response_text=message.content) is None


def test_a_call_salvage_declines_is_stripped_with_the_notice():
    text = 'Deleting.\n<tool_call>\n{"name": "delete_everything", "arguments": {"path": "/"}}\n</tool_call>'
    settings = {"mode": "salvage", "salvage_models": ["qwen*"]}
    response = _chat(text)
    assert _run_middleware(settings, response) is response  # tool not offered: not converted
    out = plugin.make_output_hook(_getter(**settings))(response_text=text)
    assert out == "Deleting.\n\n(Not run: delete_everything. The model wrote the call as text instead of making it.)"


@pytest.mark.parametrize("case", FALSE_POSITIVES, ids=[c["id"] for c in FALSE_POSITIVES])
def test_output_hook_returns_none_for_false_positives(case):
    assert plugin.make_output_hook(_getter())(response_text=case["text"]) is None


def _run_middleware(settings, response, model="qwen3-coder", provider="custom"):
    calls = []

    def next_call(request):
        calls.append(request)
        return response

    out = plugin.make_execution_middleware(_getter(**settings))(
        request=ALL_TOOLS, next_call=next_call, model=model, provider=provider, session_id="s")
    assert len(calls) == 1  # the provider is called exactly once
    return out


def test_middleware_is_off_by_default_and_needs_a_listed_model():
    response = _chat('<tool_call>{"name": "get_time"}</tool_call>')
    assert _run_middleware({}, response) is response
    assert _run_middleware({"mode": "salvage"}, response) is response
    assert _run_middleware({"mode": "salvage", "salvage_models": ["llama*"]}, response) is response
    out = _run_middleware({"mode": "salvage", "salvage_models": ["qwen*"]}, response)
    assert out is not response and out.choices[0].message.tool_calls[0].function.name == "get_time"


def test_middleware_fails_open():
    weird = SimpleNamespace(choices=object())
    assert _run_middleware({"mode": "salvage", "salvage_models": ["*"]}, weird) is weird


def test_register_declares_exactly_what_the_manifest_lists():
    registered = []
    ctx = SimpleNamespace(get_config=_getter(),
                          register_hook=lambda name, cb, **kw: registered.append(("hook", name)),
                          register_middleware=lambda kind, cb: registered.append(("middleware", kind)))
    plugin.register(ctx)
    assert registered == [("hook", "transform_llm_output"), ("middleware", "llm_execution")]
