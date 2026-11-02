"""End to end with a real Hermes: the plugin is installed into a throwaway HOME / HERMES_HOME,
``hermes -z`` talks to a scripted local provider that writes its tool call as text, and the test
reads what the user gets and what Hermes sent back to the provider.

Runs only when ``HERMES_E2E_BIN`` names a ``hermes`` executable. ``HERMES_E2E_TMP`` sets the
parent folder for the throwaway home (default: pytest's tmp_path). HOME is replaced too, so the
run never touches the real profile or a gateway already serving it; the gateway this run starts
for the throwaway home is stopped afterwards with ``hermes gateway stop``.
"""

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from fake_provider import FakeProvider

HERMES = os.environ.get("HERMES_E2E_BIN", "")
ROOT = Path(__file__).resolve().parent.parent
PLUGIN_FILES = ("plugin.yaml", "__init__.py", "formats.py", "salvage.py", "README.md", "LICENSE")

pytestmark = pytest.mark.skipif(not HERMES, reason="set HERMES_E2E_BIN to a hermes executable")

NOTES = "milk, eggs, coffee"
DONE = "Your notes list milk, eggs and coffee."


def _leak(user_home, wrapped):
    call = f"<function=read_file>\n<parameter=path>\n{user_home / 'notes.txt'}\n</parameter>\n</function>"
    return f"<tool_call>\n{call}\n</tool_call>" if wrapped else f"Sure.\n{call}"


def _env(user_home):
    return {"HOME": str(user_home), "HERMES_HOME": str(user_home / ".hermes"), "PATH": "/usr/bin:/bin",
            "HERMES_DISABLE_LAZY_INSTALLS": "1", "LANG": "C.UTF-8"}


@pytest.fixture
def home(tmp_path):
    base = os.environ.get("HERMES_E2E_TMP")
    root = Path(tempfile.mkdtemp(prefix="tcs-", dir=base)) if base else tmp_path
    user_home = root / "home"
    (user_home / ".hermes").mkdir(parents=True)
    (user_home / "notes.txt").write_text(NOTES)
    yield user_home
    _stop_gateway(user_home)
    if base:
        shutil.rmtree(root, ignore_errors=True)


def _stop_gateway(user_home):
    """Stop the gateway ``hermes -z`` started for this throwaway home (it would idle-exit later)."""
    state = user_home / ".hermes" / "gateway_state.json"
    try:
        pid = json.loads(state.read_text())["pid"]
        if json.loads(state.read_text()).get("hermes_home") != str(user_home / ".hermes"):
            return
        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError, KeyError):
        return
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.1)


def _install(user_home, provider, enabled, settings=None, defer=None):
    hermes_home = user_home / ".hermes"
    dest = hermes_home / "plugins" / "tool-call-salvage"
    dest.mkdir(parents=True)
    for name in PLUGIN_FILES:
        shutil.copy2(ROOT / name, dest / name)
    plugins: dict = {"enabled": ["tool-call-salvage"] if enabled else []}
    if settings:
        plugins["entries"] = {"tool-call-salvage": {"settings": settings}}
    config = {"model": {"provider": "custom", "base_url": provider.base_url, "default": provider.model,
                        "api_key": "test", "context_length": 131072},
              "plugins": plugins}
    if defer is not None:  # which tools Tool Search keeps behind its tool_call bridge
        config["tools"] = {"tool_search": {"defer": defer}}
    (hermes_home / "config.yaml").write_text(json.dumps(config, indent=2))  # JSON is valid YAML


def _chat(user_home):
    proc = subprocess.run([HERMES, "-z", "What is in notes.txt?", "-t", "file"], env=_env(user_home),
                          capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
                          cwd=user_home)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    return proc.stdout


def _tool_messages(body):
    return [m for m in body.get("messages", []) if m.get("role") == "tool"]


def test_without_the_plugin_the_user_sees_the_markup(home):
    with FakeProvider(lambda body: _leak(home, wrapped=False)) as provider:
        _install(home, provider, enabled=False)
        out = _chat(home)
    assert "<function=read_file>" in out
    assert not any(_tool_messages(b) for b in provider.main_requests())


def test_strip_mode_removes_the_markup_and_runs_nothing(home):
    with FakeProvider(lambda body: _leak(home, wrapped=False)) as provider:
        _install(home, provider, enabled=True)
        out = _chat(home)
    assert out.strip() == "Sure."
    assert "<function" not in out
    assert not any(_tool_messages(b) for b in provider.main_requests())


def _salvage_run(home, **install):
    def script(body):
        return DONE if _tool_messages(body) else _leak(home, wrapped=True)

    with FakeProvider(script) as provider:
        _install(home, provider, enabled=True, **install)
        out = _chat(home)
    assert DONE in out and "<tool_call>" not in out and "<function" not in out
    second = provider.main_requests()[-1]
    (tool_message,) = _tool_messages(second)
    assistant = next(m for m in second["messages"] if m.get("role") == "assistant" and m.get("tool_calls"))
    (call,) = assistant["tool_calls"]
    assert tool_message["tool_call_id"] == call["id"]  # Hermes paired the result with the salvaged call
    assert NOTES in str(tool_message["content"])  # the tool really ran
    assert "<tool_call>" not in (assistant.get("content") or "")  # the markup is not replayed
    return call["function"], [t["function"]["name"] for t in second["tools"]]


def test_salvage_mode_runs_the_written_call_through_hermes(home):
    function, offered = _salvage_run(home, settings={"mode": "salvage", "salvage_models": ["fake-*"]})
    assert "read_file" in offered
    assert function["name"] == "read_file"
    assert json.loads(function["arguments"]) == {"path": str(home / "notes.txt")}


def test_salvage_mode_reaches_a_deferred_tool_through_the_bridge(home):
    function, offered = _salvage_run(
        home, defer=["read_file"], settings={"mode": "salvage", "salvage_models": ["fake-*"], "deferred_tools": True})
    assert "read_file" not in offered and "tool_call" in offered
    assert function["name"] == "tool_call"
    assert json.loads(function["arguments"]) == {
        "calls": [{"name": "read_file", "arguments": {"path": str(home / "notes.txt")}}]}
