"""Load the plugin directory as the package ``tool_call_salvage`` (its folder name has dashes)."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for ``import corpus``


def _load():
    if "tool_call_salvage" in sys.modules:
        return sys.modules["tool_call_salvage"]
    spec = importlib.util.spec_from_file_location(
        "tool_call_salvage", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["tool_call_salvage"] = module
    spec.loader.exec_module(module)
    return module


_load()
