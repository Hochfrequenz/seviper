"""
This package contains a little framework to conveniently handle errors in async and sync code.
It also provides pipable operators to handle errors inside an aiostream pipeline.
"""

import importlib
from typing import TYPE_CHECKING

from .context_manager import context_manager
from .decorator import decorator, decorator_as_result, retry_on_error
from .result import NegativeResult, PositiveResult, ResultType
from .types import UNSET, AsyncFunctionType, FunctionType, SecuredAsyncFunctionType, SecuredFunctionType, UnsetType

if TYPE_CHECKING:
    from . import pipe, stream
else:
    stream = importlib.import_module("error_handler.stream")
    pipe = importlib.import_module("error_handler.pipe")
