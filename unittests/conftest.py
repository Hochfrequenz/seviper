import builtins
import importlib
import sys

import pytest


@pytest.fixture(scope="function")
def trigger_aiostream_import_error():
    realimport = builtins.__import__

    def myimport(name, global_vars, local_vars, fromlist, level):
        if name.startswith("aiostream"):
            raise ImportError
        return realimport(name, global_vars, local_vars, fromlist, level)

    builtins.__import__ = myimport

    modules_to_reload = [
        "error_handler._extra",
        "error_handler.stream",
        "error_handler.pipe",
        "error_handler",
    ]
    for module in modules_to_reload:
        importlib.reload(importlib.import_module(module))

    yield myimport

    builtins.__import__ = realimport

    for module in modules_to_reload:
        sys.modules.pop(module)
        importlib.reload(importlib.import_module(module))
