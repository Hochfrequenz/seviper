"""
This package contains a little framework to conveniently handle errors in async and sync code.
It also provides pipable operators to handle errors inside an aiostream pipeline.
"""

import importlib

from .context_manager import context_manager
from .decorator import decorator, retry_on_error
from .types import (
    ERRORED,
    UNSET,
    AsyncFunctionType,
    ErroredType,
    FunctionType,
    SecuredAsyncFunctionType,
    SecuredFunctionType,
    UnsetType,
)

stream = importlib.import_module("error_handler.stream")
pipe = importlib.import_module("error_handler.pipe")
