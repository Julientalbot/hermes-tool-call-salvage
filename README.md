# tool-call-salvage

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin for tool calls that a model
writes into its reply as text instead of making them.

## The problem

Some models and serving stacks sometimes return a tool call as plain text: `<tool_call>…`,
`<function=…>`, `<invoke name="…">`, DeepSeek DSML or special tokens, Harmony `to=functions.…`,
`[TOOL_CALLS]`, or a bare `{"name": …, "arguments": …}`. The `tool_calls` field is empty, so the
user sees markup and nothing runs. Reports upstream include NousResearch/hermes-agent #117090,
#135008, #74723, #87900, #63282, #129060, #119264, #137775 and #125472. Hermes already removes
closed `<tool_call>` blocks from the final reply and recovers leaks from the Codex Responses API;
this plugin covers the other shapes and, if you ask for it, runs the call.

## Default behaviour (`mode: strip`)

When the final reply **ends** with complete tool-call markup, the plugin removes that markup and
logs a warning that names the format and the tools (never the arguments). Nothing runs. If the
reply was only markup, the user gets a short notice instead (`empty_notice`).

It never touches:

- fenced (```` ``` ````, `~~~`, an unclosed fence included) or indented code blocks, so a
  developer who asks what a tool call looks like gets the example;
- markup followed by more text (an example or an explanation), or plain markup in the middle of
  a line (only special tokens such as `<|python_tag|>` or `<｜tool▁calls▁begin｜>` may follow
  prose on the same line);
- JSON. A bare JSON object is never stripped, because it is also an ordinary answer;
- incomplete markup: every parameter must be closed.

## Optional: run the call (`mode: salvage`, off by default)

For the models listed in `salvage_models`, the plugin reads each Chat Completions response
before Hermes does. If the response has no structured call and its text **ends** with complete
calls, the plugin turns them into structured calls, and Hermes runs them through its normal path
(argument checks, guardrails, approvals). Text before the calls stays as the reply. All of these
must hold, or the response is left as it was:

- every tool name is offered **in that request**, and each call has the tool's required arguments
  (with `deferred_tools: true`, a tool kept behind Hermes's Tool Search is reached through the
  offered `tool_call` bridge instead, which checks the name against the session's tools);
- the generation was not cut off (`finish_reason` is not `length`);
- at most `max_calls` calls;
- for bare JSON (Ollama `/v1`): `bare_json: true`, and the reply is nothing but call objects
  with no extra keys.

If the reply is empty and the reasoning ends with complete calls (a model that stalled with its
call written there), the calls are taken from the reasoning, which is kept as it is.

## Settings

In `~/.hermes/config.yaml` (also shown in the Desktop app under Settings → Plugins):

```yaml
plugins:
  enabled: [tool-call-salvage]
  entries:
    tool-call-salvage:
      settings:
        mode: strip            # strip | salvage
        salvage_models: []     # globs on the model or provider/model, e.g. ["qwen*", "ollama/*"]
        bare_json: false       # salvage only: accept a reply made only of JSON call objects
        deferred_tools: false  # salvage only: reach Tool Search (deferred) tools via tool_call
        max_calls: 8           # more calls than this in one reply: nothing is converted (1-32)
        empty_notice: "The model wrote a tool call as text instead of making it, so nothing was run. Please ask again."
```

An empty `salvage_models` list means nothing is converted, even in `salvage` mode. Settings are
read on every call, so a change applies without a restart. `empty_notice: ""` leaves a reply that
was only markup unchanged.

## Formats

| Family | Shape | Seen with |
|---|---|---|
| `tool_call` | `<tool_call>{json}</tool_call>`, `<tool_call><function=…>`, `<minimax:tool_call>` | Hermes and Qwen templates, MiniMax |
| `function_tag` | `<function=NAME><parameter=KEY>…</parameter></function>` | Qwen3-Coder, quantized Qwen |
| `invoke` | `<invoke name="NAME"><parameter name="KEY">…`, bare or in `<function_calls>`; `<｜DSML｜tool_calls>` | DeepSeek DSML, others |
| `deepseek` | `<｜tool▁calls▁begin｜>…` | DeepSeek V3, V3.1 |
| `kimi` | `<\|tool_calls_section_begin\|>…` | Kimi K2 |
| `harmony` | `<\|channel\|>commentary to=functions.NAME … <\|message\|>{json}` | gpt-oss |
| `mistral` | `[TOOL_CALLS][{json}]`, `[TOOL_CALLS]NAME[ARGS]{json}` | Mistral |
| `python_tag` | `<\|python_tag\|>{json}` | Llama 3.x |
| `bare_json` | `{"name": …, "arguments": …}` (salvage only, opt-in) | Ollama `/v1` |

## Disclosure

No network, no files, no shell, no subprocesses, no telemetry, no stored state. The plugin reads
its own settings and changes only the current reply. In `strip` mode it removes leaked tool-call
markup from the final reply. In `salvage` mode, for listed models only, text the model wrote can
become a real tool call: only tools offered in that request (and, with `deferred_tools`, the
session's deferred tools through Hermes's `tool_call` bridge), run through Hermes's normal
approvals.

## Limits

- With streaming on, the user may see the markup arrive before the final reply replaces it.
- Salvage reads Chat Completions responses only. Anthropic Messages and Responses API replies
  are only stripped (Hermes already recovers Codex leaks).
- Markup that does not end the reply, and markup in the commentary of a step that also made a
  real call, are left alone.
- A reply that is nothing but XML in exactly the shape of a call, on its own lines and outside a
  code block (for example `<function name="x"><parameter name="y">1</parameter></function>`),
  is treated as a leak.
- The proper fix belongs in Hermes's response intake, where several core PRs propose one. This
  plugin is a stopgap: remove it once Hermes handles your model's format.

## Development

```sh
python -m pytest tests                                            # unit tests
HERMES_E2E_BIN=$(command -v hermes) python -m pytest tests/test_e2e_hermes.py
hermes plugins validate .
```

The end-to-end test runs `hermes -z` in a throwaway HOME against a scripted local provider.

## License

MIT
