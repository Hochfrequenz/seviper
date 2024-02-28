"""
This module provides a context manager to handle errors in a convenient way.
"""

from typing import Any, Callable, ContextManager

from .core import Catcher


# pylint: disable=unsubscriptable-object
def context_manager(
    on_success: Callable[[], Any] | None = None,
    on_error: Callable[[Exception], Any] | Callable[[], Any] | None = None,
    on_finalize: Callable[[], Any] | None = None,
    suppress_recalling_on_error: bool = True,
) -> ContextManager[Catcher[None]]:
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
    catcher = Catcher[None](on_success, on_error, on_finalize, suppress_recalling_on_error=suppress_recalling_on_error)
    return catcher.secure_context()
