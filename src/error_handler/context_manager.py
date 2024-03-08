"""
This module provides a context manager to handle errors in a convenient way.
"""

from contextlib import contextmanager
from typing import Any, Callable, Iterator

from .callback import Callback, ErrorCallback
from .core import Catcher, ContextCatcher
from .types import UnsetType


# pylint: disable=unsubscriptable-object
@contextmanager
def context_manager(
    *,
    on_success: Callable[[], Any] | None = None,
    on_error: Callable[[BaseException], Any] | None = None,
    on_finalize: Callable[[], Any] | None = None,
    suppress_recalling_on_error: bool = True,
) -> Iterator[Catcher[UnsetType]]:
    """
    This context manager catches all errors inside the context and calls the corresponding callbacks.
    It is a shorthand for creating a Catcher instance and using its secure_context method.
    If the context raises an error, the on_error callback will be called.
    If the context does not raise an error, the on_success callback will be called.
    The on_finalize callback will be called in both cases and after the other callbacks.
    If reraise is True, the error will be reraised after the callbacks were called.
    If suppress_recalling_on_error is True, the on_error callable will not be called if the error were already
    caught by a previous catcher.
    """
    catcher = ContextCatcher(
        Callback.from_callable(on_success, return_type=Any) if on_success is not None else None,
        ErrorCallback.from_callable(on_error, return_type=Any) if on_error is not None else None,
        Callback.from_callable(on_finalize, return_type=Any) if on_finalize is not None else None,
        suppress_recalling_on_error=suppress_recalling_on_error,
    )
    with catcher.secure_context():
        yield catcher
    catcher.handle_result_and_call_callbacks(catcher.result)
