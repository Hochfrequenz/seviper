"""
This module contains decorators to secure an async or sync callable and handle its errors.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, TypeGuard

from .core import Catcher
from .types import (
    ERRORED,
    AsyncFunctionType,
    ErroredType,
    FunctionType,
    NegativeResult,
    P,
    PositiveResult,
    ResultType,
    SecuredAsyncFunctionType,
    SecuredFunctionType,
    T,
)


def iscoroutinefunction(callable_: FunctionType[P, T] | AsyncFunctionType[P, T]) -> TypeGuard[AsyncFunctionType[P, T]]:
    """
    This function checks if the given callable is a coroutine function.
    """
    return asyncio.iscoroutinefunction(callable_)


# pylint: disable=too-many-arguments
def decorator(
    on_success: Callable[[T], Any] | Callable[[], Any] | None = None,
    on_error: Callable[[Exception], Any] | Callable[[], Any] | None = None,
    on_finalize: Callable[[], Any] | None = None,
    on_error_return_always: T | ErroredType = ERRORED,
    suppress_recalling_on_error: bool = True,
) -> Callable[
    [FunctionType[P, T] | AsyncFunctionType[P, T]], SecuredFunctionType[P, T] | SecuredAsyncFunctionType[P, T]
]:
    """
    This decorator secures a callable (sync or async) and handles its errors.
    If the callable raises an error, the on_error callback will be called and the value if on_error_return_always
    will be returned.
    If the callable does not raise an error, the on_success callback will be called (the return value will be
    provided to the callback if it receives an argument) and the return value will be returned.
    The on_finalize callback will be called in both cases and after the other callbacks.
    If reraise is True, the error will be reraised after the callbacks were called.
    If suppress_recalling_on_error is True, the on_error callable will not be called if the error were already
    caught by a previous catcher.
    """
    # pylint: disable=unsubscriptable-object
    catcher = Catcher[T](on_success, on_error, on_finalize, on_error_return_always, suppress_recalling_on_error)

    def decorator_inner(
        callable_to_secure: FunctionType[P, T] | AsyncFunctionType[P, T]
    ) -> SecuredFunctionType[P, T] | SecuredAsyncFunctionType[P, T]:
        if iscoroutinefunction(callable_to_secure):

            @functools.wraps(callable_to_secure)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | ErroredType:
                return (
                    await catcher.secure_await(callable_to_secure(*args, **kwargs))
                ).result  # type: ignore[return-value]

                # Incompatible return value type (got "object", expected "T")  [return-value]
                # Seems like mypy isn't good enough for this.

        else:

            @functools.wraps(callable_to_secure)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | ErroredType:
                return catcher.secure_call(  # type: ignore[return-value]
                    callable_to_secure,  # type: ignore[arg-type]
                    *args,
                    **kwargs,
                ).result
                # Incompatible return value type (got "object", expected "T")  [return-value]
                # Seems like mypy isn't good enough for this.

        wrapper.__catcher__ = catcher  # type: ignore[attr-defined]
        wrapper.__original_callable__ = callable_to_secure  # type: ignore[attr-defined]
        return wrapper

    return decorator_inner


# pylint: disable=too-many-arguments
def retry_on_error(
    on_error: Callable[[Exception, int], bool],
    retry_stepping_func: Callable[[int], float] = lambda retry_count: 1.71**retry_count,
    # <-- with max_retries = 10 the whole decorator may wait up to 5 minutes.
    # because sum(1.71seconds**i for i in range(10)) == 5minutes
    max_retries: int = 10,
    on_success: Callable[[T, int], Any] | None = None,
    on_fail: Callable[[Exception, int], Any] | None = None,
    on_finalize: Callable[[int], Any] | None = None,
    logger: logging.Logger = logging.getLogger(__name__),
) -> Callable[[FunctionType[P, T]], FunctionType[P, T]]:
    """
    This decorator retries a callable (sync or async) on error.
    The retry_stepping_func is called with the retry count and should return the time to wait until the next retry.
    The max_retries parameter defines how often the callable will be retried at max.
    If the decorated function raises an error, the on_error callback will be called and the return value of the callback
    will be used to decide if the function should be retried.
    The function fails immediately, if the on_error callback returns False or if the max_retries are reached.
    In this case, the on_fail callback will be called and the respective error will be raised.
    You can additionally use the normal decorator on top of that if you don't want an exception to be raised.
    """
    # pylint: disable=unsubscriptable-object
    catcher = Catcher[T]()

    def handle_result(result: ResultType[T], retry_count: int) -> bool:
        if isinstance(result, NegativeResult):
            if not on_error(result.error, retry_count):
                if on_fail is not None:
                    on_fail(result.error, retry_count)
                if on_finalize is not None:
                    on_finalize(retry_count)
                raise result.error
            return True
        if on_success is not None:
            on_success(result.result, retry_count)
        if on_finalize is not None:
            on_finalize(retry_count)
        return False

    def too_many_retries_error_handler(callback_name: str, max_retries: int) -> Exception:
        too_many_retries_error = RuntimeError(f"Too many retries ({max_retries}) for {callback_name}")
        if on_fail is not None:
            on_fail(too_many_retries_error, max_retries)
        if on_finalize is not None:
            on_finalize(max_retries)
        return too_many_retries_error

    def decorator_inner(
        callable_to_secure: FunctionType[P, T] | AsyncFunctionType[P, T]
    ) -> SecuredFunctionType[P, T] | SecuredAsyncFunctionType[P, T]:
        if iscoroutinefunction(callable_to_secure):

            @functools.wraps(callable_to_secure)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                for retry_count in range(max_retries):
                    result = await catcher.secure_await(callable_to_secure(*args, **kwargs))
                    if handle_result(result, retry_count):
                        # Should retry
                        await asyncio.sleep(retry_stepping_func(retry_count))
                        continue
                    # Should not retry because the result is positive
                    assert isinstance(result, PositiveResult), "Internal error: NegativeResult was not handled properly"
                    return result.result

                raise too_many_retries_error_handler(callable_to_secure.__name__, max_retries)

        else:
            logger.warning(
                "Sync retry decorator is dangerous as it uses time.sleep() for retry logic. "
                "Combined with asyncio code it could lead to deadlocks and other unexpected behaviour. "
                "Please consider decorating an async function instead."
            )

            @functools.wraps(callable_to_secure)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                for retry_count in range(max_retries):
                    result = catcher.secure_call(callable_to_secure, *args, **kwargs)  # type: ignore[arg-type]
                    if handle_result(result, retry_count):
                        # Should retry
                        time.sleep(retry_stepping_func(retry_count))
                        continue
                    # Should not retry because the result is positive
                    assert isinstance(result, PositiveResult), "Internal error: NegativeResult was not handled properly"
                    return result.result

                too_many_retries_error = RuntimeError(
                    f"Too many retries ({max_retries}) for {callable_to_secure.__name__}"
                )
                if on_fail is not None:
                    on_fail(too_many_retries_error, max_retries)
                raise too_many_retries_error

        wrapper.__catcher__ = catcher  # type: ignore[attr-defined]
        wrapper.__original_callable__ = callable_to_secure  # type: ignore[attr-defined]
        return wrapper

    return decorator_inner  # type: ignore[return-value]
