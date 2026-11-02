"""Shared test corpus.

LEAKS: replies that end with a tool call written as text, in the shapes reported upstream
(NousResearch/hermes-agent issue or PR number in ``source``) or produced by the model's own chat
template. ``kept`` is the reply after stripping; ``calls`` is what strict reading must return.

FALSE_POSITIVES: replies that must come out byte-identical, in every mode.
"""

DSML = "\uff5cDSML\uff5c"
DS_CALLS_BEGIN, DS_CALLS_END = "<\uff5ctool\u2581calls\u2581begin\uff5c>", "<\uff5ctool\u2581calls\u2581end\uff5c>"
DS_CALL_BEGIN, DS_CALL_END = "<\uff5ctool\u2581call\u2581begin\uff5c>", "<\uff5ctool\u2581call\u2581end\uff5c>"
DS_SEP = "<\uff5ctool\u2581sep\uff5c>"

LEAKS = [
    dict(
        id="qwen3_coder_wrapped", source="#129060, #137775 (Qwen3-Coder template)",
        text="I will list the directory.\n<tool_call>\n<function=terminal>\n<parameter=command>\nls -la\n"
             "</parameter>\n</function>\n</tool_call>",
        kept="I will list the directory.",
        calls=[("terminal", {"command": "ls -la"})],
    ),
    dict(
        id="qwen_function_plain", source="#135008 (quantized Qwen, plain <function=…>)",
        text="I should check the repo state first.\n\n<function=terminal>\n<parameter=command>\n"
             "cd ~/.hermes/hermes-agent && git status --short | head\n</parameter>\n</function>\n",
        kept="I should check the repo state first.",
        calls=[("terminal", {"command": "cd ~/.hermes/hermes-agent && git status --short | head"})],
    ),
    dict(
        id="qwen_function_two_calls", source="#135008",
        text="Lass mich die Datei lesen.\n\n<function=read_file>\n<parameter=path>/tmp/a.py\n</parameter>\n"
             "<parameter=offset>120\n</parameter>\n</function>\n<function=terminal>\n<parameter=command>"
             "echo hi\n</parameter>\n</function>",
        kept="Lass mich die Datei lesen.",
        calls=[("read_file", {"path": "/tmp/a.py", "offset": "120"}), ("terminal", {"command": "echo hi"})],
    ),
    dict(
        id="qwen_function_multiline_value", source="#135008",
        text="<function=execute_code>\n<parameter=code>from hermes_tools import terminal\nr = terminal('ls')\n"
             "print(r['output'])\n</parameter>\n</function>",
        kept="",
        calls=[("execute_code", {"code": "from hermes_tools import terminal\nr = terminal('ls')\nprint(r['output'])"})],
    ),
    dict(
        id="hermes_json_tool_call", source="#129060 (Hermes / Qwen 2.5 template)",
        text='<tool_call>\n{"name": "web_search", "arguments": {"query": "hermes agent plugins"}}\n</tool_call>',
        kept="",
        calls=[("web_search", {"query": "hermes agent plugins"})],
    ),
    dict(
        id="hermes_json_tool_call_no_closer", source="closing tag sent as a stop token",
        text='Searching.\n<tool_call>\n{"name": "web_search", "arguments": {"query": "x"}}',
        kept="Searching.",
        calls=[("web_search", {"query": "x"})],
    ),
    dict(
        id="dsml", source="#119264, #117090 (DeepSeek DSML)",
        text=f"Before <{DSML}tool_calls>\n<{DSML}invoke name=\"terminal\">\n"
             f"<{DSML}parameter name=\"command\" string=\"true\">echo hello</{DSML}parameter>\n"
             f"<{DSML}parameter name=\"timeout\" string=\"false\">8</{DSML}parameter>\n"
             f"</{DSML}invoke>\n</{DSML}tool_calls>",
        kept="Before",
        calls=[("terminal", {"command": "echo hello", "timeout": 8})],
    ),
    dict(
        id="invoke_bare", source="#74723, #63282 (standalone invoke)",
        text='I will search now.\n\n<invoke name="web_search">\n<parameter name="query">hermes</parameter>\n</invoke>',
        kept="I will search now.",
        calls=[("web_search", {"query": "hermes"})],
    ),
    dict(
        id="invoke_wrapped", source="function_calls wrapper",
        text='<function_calls>\n<invoke name="read_file">\n<parameter name="path">/tmp/x</parameter>\n</invoke>\n'
             '<invoke name="terminal">\n<parameter name="command">pwd</parameter>\n</invoke>\n</function_calls>',
        kept="",
        calls=[("read_file", {"path": "/tmp/x"}), ("terminal", {"command": "pwd"})],
    ),
    dict(
        id="minimax_wrapped", source="MiniMax-M2 template",
        text='<minimax:tool_call>\n<invoke name="web_search">\n<parameter name="query">weather</parameter>\n'
             '</invoke>\n</minimax:tool_call>',
        kept="",
        calls=[("web_search", {"query": "weather"})],
    ),
    dict(
        id="envelope", source="#117090 (calls batched in a synthetic tool_call invoke)",
        text='<invoke name="tool_call">\n<parameter name="calls">[{"name": "terminal", "arguments": '
             '{"command": "date"}}, {"name": "web_search", "arguments": {"query": "x"}}]</parameter>\n</invoke>',
        kept="",
        calls=[("tool_call", {"calls": '[{"name": "terminal", "arguments": {"command": "date"}}, '
                                       '{"name": "web_search", "arguments": {"query": "x"}}]'})],
    ),
    dict(
        id="deepseek_v3", source="DeepSeek-V3 template",
        text=f"Let me check.{DS_CALLS_BEGIN}{DS_CALL_BEGIN}function{DS_SEP}get_weather\n```json\n"
             f'{{"location": "Paris"}}\n```{DS_CALL_END}{DS_CALLS_END}',
        kept="Let me check.",
        calls=[("get_weather", {"location": "Paris"})],
    ),
    dict(
        id="deepseek_v31", source="DeepSeek-V3.1 template",
        text=f'{DS_CALLS_BEGIN}{DS_CALL_BEGIN}get_weather{DS_SEP}{{"location": "Paris"}}{DS_CALL_END}'
             f'{DS_CALL_BEGIN}get_time{DS_SEP}{{}}{DS_CALL_END}{DS_CALLS_END}',
        kept="",
        calls=[("get_weather", {"location": "Paris"}), ("get_time", {})],
    ),
    dict(
        id="kimi_k2", source="Kimi K2 template",
        text='<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>'
             '{"city": "Paris"}<|tool_call_end|><|tool_calls_section_end|>',
        kept="",
        calls=[("get_weather", {"city": "Paris"})],
    ),
    dict(
        id="harmony", source="gpt-oss Harmony format (#125472 is the Codex variant)",
        text='Checking.<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json'
             '<|message|>{"location": "Tokyo"}<|call|>',
        kept="Checking.",
        calls=[("get_weather", {"location": "Tokyo"})],
    ),
    dict(
        id="harmony_recipient_first_no_call_token", source="gpt-oss, <|call|> eaten as stop token",
        text='<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{"location": "Tokyo"}',
        kept="",
        calls=[("get_weather", {"location": "Tokyo"})],
    ),
    dict(
        id="mistral_array", source="Mistral [TOOL_CALLS] (v3 tokenizer)",
        text='[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Paris"}}]',
        kept="",
        calls=[("get_weather", {"city": "Paris"})],
    ),
    dict(
        id="mistral_named", source="Mistral [TOOL_CALLS]NAME[ARGS] (v11 tokenizer)",
        text='[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}[TOOL_CALLS]get_time[ARGS]{}',
        kept="",
        calls=[("get_weather", {"city": "Paris"}), ("get_time", {})],
    ),
    dict(
        id="llama_python_tag", source="Llama 3.x <|python_tag|>",
        text='<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}<|eom_id|>',
        kept="",
        calls=[("get_weather", {"city": "Paris"})],
    ),
]

# Ollama /v1 shapes (#87900): read only with bare_json, never stripped.
BARE_JSON = [
    dict(id="ollama_single", text='{"name": "web_search", "arguments": {"query": "x"}}',
         calls=[("web_search", {"query": "x"})]),
    dict(id="ollama_type_function", text='{"type": "function", "name": "web_search", "arguments": {"query": "x"}}',
         calls=[("web_search", {"query": "x"})]),
    dict(id="ollama_two_no_array", text='{"name": "get_time"}\n{"name": "web_search", "arguments": {"query": "x"}}',
         calls=[("get_time", {}), ("web_search", {"query": "x"})]),
    dict(id="ollama_fenced", text='```json\n{"name": "web_search", "arguments": {"query": "x"}}\n```',
         calls=[("web_search", {"query": "x"})]),
    dict(id="ollama_fenced_unclosed", text='```json\n{"name": "web_search", "arguments": {"query": "x"}}',
         calls=[("web_search", {"query": "x"})]),
    dict(id="openai_wrapper", text='{"type": "function", "function": {"name": "web_search", '
                                   '"arguments": "{\\"query\\": \\"x\\"}"}}',
         calls=[("web_search", {"query": "x"})]),
]

FALSE_POSITIVES = [
    dict(id="fenced_xml_example", why="a developer asked what a Qwen tool call looks like",
         text="Qwen writes tool calls like this:\n\n```xml\n<tool_call>\n<function=terminal>\n<parameter=command>\n"
              "ls\n</parameter>\n</function>\n</tool_call>\n```"),
    dict(id="fenced_json_example", why="OpenAI tool-call example in a json fence, after prose",
         text='An OpenAI tool call looks like:\n```json\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n```'),
    dict(id="fenced_unclosed", why="unclosed fence at the end is still code",
         text="Example:\n```xml\n<invoke name=\"web_search\">\n<parameter name=\"query\">x</parameter>\n</invoke>"),
    dict(id="tilde_fence", why="~~~ fences count too",
         text="~~~\n<function=terminal>\n<parameter=command>\nls\n</parameter>\n</function>\n~~~"),
    dict(id="indented_code", why="4-space indented code block",
         text="Example:\n\n    <invoke name=\"web_search\">\n    <parameter name=\"query\">x</parameter>\n    </invoke>"),
    dict(id="inline_prose_tags", why="prose naming the formats",
         text="Qwen emits `<tool_call>` tags, DeepSeek uses `<\uff5ctool\u2581calls\u2581begin\uff5c>` tokens "
              "and Mistral starts with `[TOOL_CALLS]`."),
    dict(id="inline_example_mid_line", why="#74723 negative: inline example in a sentence",
         text='For example: <invoke name="memory"></invoke> is transcript text.'),
    dict(id="inline_example_line_end", why="inline example closing the reply, not on its own line",
         text='The leaked form is: <invoke name="memory"></invoke>'),
    dict(id="example_then_explanation", why="#74723 negative: markup followed by more text",
         text='<invoke name="memory">\n<parameter name="action">add</parameter>\n</invoke>\nThis is an example.'),
    dict(id="json_answer", why="ordinary JSON answer",
         text='{"name": "Alice", "age": 30}'),
    dict(id="json_payload_in_prose", why="#87900 negative: JSON call object after prose",
         text='Sure, here is the payload: {"name": "search", "arguments": {"q": "x"}}'),
    dict(id="json_extra_key", why="#87900 negative: an extra key means data",
         text='{"name": "search", "arguments": {"q": "x"}, "priority": 2}'),
    dict(id="truncated_parameter", why="#117090: a cut-off value is never read",
         text="<function=terminal>\n<parameter=command>\nrm -rf /tmp/half"),
    dict(id="ascii_dsml_lookalike", why="#119264 control: ASCII bars are prose",
         text="Use <|DSML|tool_calls> as a literal example, not a tool call."),
    dict(id="blockquote", why="quoted markup",
         text='> <invoke name="web_search">\n> <parameter name="query">x</parameter>\n> </invoke>'),
    dict(id="xml_document", why="an XML answer with <function> elements",
         text='<functions>\n  <function name="handler">\n    <parameter name="x">1</parameter>\n  </function>\n'
              '</functions>'),
    dict(id="harmony_in_fence", why="Harmony tokens shown in a code block",
         text='```\n<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"a": 1}<|call|>\n```'),
    dict(id="plain_text_about_tool_calls", why="talking about tool calls",
         text="I called the web_search tool with the query \"hermes\" and found three results."),
    dict(id="empty", why="nothing", text=""),
]
