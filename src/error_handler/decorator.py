"""
This module contains decorators to secure an async or sync callable and handle its errors.
"""

import asyncio
import functools
import inspect
import logging
import time
from typing import Any, Callable, Concatenate, Generator, ParamSpec, TypeGuard, TypeVar, cast

from .callback import Callback, ErrorCallback, SuccessCallback
from .core import Catcher
from .result import CallbackResultType, PositiveResult, ResultType
from .types import (
    ERRORED,
    AsyncFunctionType,
    ErroredType,
    FunctionType,
    SecuredAsyncFunctionType,
    SecuredFunctionType,
    UnsetType,
)

_P = ParamSpec("_P")
_T = TypeVar("_T")


def iscoroutinefunction(
    callable_: FunctionType[_P, _T] | AsyncFunctionType[_P, _T]
) -> TypeGuard[AsyncFunctionType[_P, _T]]:
    """
    This function checks if the given callable is a coroutine function.
    """
    return asyncio.iscoroutinefunction(callable_)


# pylint: disable=too-many-arguments
def decorator(
    on_success: Callable[Concatenate[_T, _P], Any] | None = None,
    on_error: Callable[Concatenate[Exception, _P], Any] | None = None,
    on_finalize: Callable[_P, Any] | None = None,
    on_error_return_always: _T | ErroredType = ERRORED,
    suppress_recalling_on_error: bool = True,
) -> Callable[
    [FunctionType[_P, _T] | AsyncFunctionType[_P, _T]], SecuredFunctionType[_P, _T] | SecuredAsyncFunctionType[_P, _T]
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

    def decorator_inner(
        callable_to_secure: FunctionType[_P, _T] | AsyncFunctionType[_P, _T]
    ) -> SecuredFunctionType[_P, _T] | SecuredAsyncFunctionType[_P, _T]:
        sig = inspect.signature(callable_to_secure)
        catcher = Catcher[_T](
            SuccessCallback.from_callable(on_success, sig, return_type=Any) if on_success is not None else None,
            ErrorCallback.from_callable(on_error, sig, return_type=Any) if on_error is not None else None,
            Callback.from_callable(on_finalize, sig, return_type=Any) if on_finalize is not None else None,
            on_error_return_always,
            suppress_recalling_on_error,
        )
        if iscoroutinefunction(callable_to_secure):

            @functools.wraps(callable_to_secure)
            async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T | ErroredType:
                result = await catcher.secure_await(callable_to_secure(*args, **kwargs))
                catcher.handle_result_and_call_callbacks(result, *args, **kwargs)
                assert not isinstance(result.result, UnsetType), "Internal error: result is unset"
                return result.result

        else:

            @functools.wraps(callable_to_secure)
            def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T | ErroredType:
                result = catcher.secure_call(
                    callable_to_secure,  # type: ignore[arg-type]
                    *args,
                    **kwargs,
                )
                catcher.handle_result_and_call_callbacks(result, *args, **kwargs)
                assert not isinstance(result.result, UnsetType), "Internal error: result is unset"
                return result.result

        return_func = cast(SecuredFunctionType[_P, _T] | SecuredAsyncFunctionType[_P, _T], wrapper)
        return_func.__catcher__ = catcher
        return_func.__original_callable__ = callable_to_secure
        return return_func

    return decorator_inner


# pylint: disable=too-many-arguments, too-many-locals
def retry_on_error(
    on_error: Callable[Concatenate[Exception, int, _P], bool],
    retry_stepping_func: Callable[[int], float] = lambda retry_count: 1.71**retry_count,
    # <-- with max_retries = 10 the whole decorator may wait up to 5 minutes.
    # because sum(1.71seconds**i for i in range(10)) == 5minutes
    max_retries: int = 10,
    on_success: Callable[Concatenate[_T, int, _P], Any] | None = None,
    on_fail: Callable[Concatenate[Exception, int, _P], Any] | None = None,
    on_finalize: Callable[Concatenate[int, _P], Any] | None = None,
    logger: logging.Logger = logging.getLogger(__name__),
) -> Callable[
    [FunctionType[_P, _T] | AsyncFunctionType[_P, _T]], SecuredFunctionType[_P, _T] | SecuredAsyncFunctionType[_P, _T]
]:
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

    def decorator_inner(
        callable_to_secure: FunctionType[_P, _T] | AsyncFunctionType[_P, _T]
    ) -> SecuredFunctionType[_P, _T] | SecuredAsyncFunctionType[_P, _T]:
        sig = inspect.signature(callable_to_secure)
        sig = sig.replace(
            parameters=[
                inspect.Parameter("retries", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
                *sig.parameters.values(),
            ],
        )
        on_error_callback: ErrorCallback[Concatenate[Exception, int, _P], bool] = ErrorCallback.from_callable(
            on_error, sig, return_type=bool
        )
        on_success_callback: SuccessCallback[Concatenate[_T, int, _P], Any] | None = (
            SuccessCallback.from_callable(on_success, sig, return_type=Any) if on_success is not None else None
        )
        on_fail_callback: ErrorCallback[Concatenate[Exception, int, _P], Any] | None = (
            ErrorCallback.from_callable(on_fail, sig, return_type=Any) if on_fail is not None else None
        )
        on_finalize_callback: Callback[Concatenate[int, _P], Any] | None = (
            Callback.from_callable(on_finalize, sig, return_type=Any) if on_finalize is not None else None
        )

        # pylint: disable=unsubscriptable-object
        catcher_executor = Catcher[_T](on_error=on_error_callback)
        catcher_retrier = Catcher[_T](
            on_success=on_success_callback,
            on_error=on_fail_callback,
            on_finalize=on_finalize_callback,
            suppress_recalling_on_error=False,
        )
        retry_count = 0

        def retry_generator(*args: _P.args, **kwargs: _P.kwargs) -> Generator[int, ResultType[_T], _T]:
            nonlocal retry_count
            for retry_count_i in range(max_retries):
                result: ResultType[_T] = yield retry_count_i
                retry_count = retry_count_i
                if isinstance(result, PositiveResult):
                    assert not isinstance(result.result, UnsetType), "Internal error: result is unset"
                    return result.result
                callback_summary = catcher_executor.handle_result_and_call_callbacks(
                    result, retry_count_i, *args, **kwargs
                )
                assert (
                    callback_summary.callback_result_types.error == CallbackResultType.SUCCESS
                ), "Internal error: on_error callback was not successful but didn't raise exception"
                if callback_summary.callback_return_values.error is True:
                    yield retry_count_i
                    continue
                # Should not retry
                raise result.error

            retry_count = max_retries
            error = RuntimeError(f"Too many retries ({max_retries}) for {callable_to_secure.__name__}")
            raise error

        def handle_result_and_call_callbacks(result: ResultType[_T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
            if isinstance(result, PositiveResult):
                assert not isinstance(result.result, UnsetType), "Internal error: result is unset"
                catcher_retrier.handle_success_case(
                    result.result,
                    retry_count,
                    *args,
                    **kwargs,
                )
                return result.result

            catcher_retrier.handle_error_case(result.error, retry_count, *args, **kwargs)
            raise result.error

        if iscoroutinefunction(callable_to_secure):

            async def retry_function_async(*args: _P.args, **kwargs: _P.kwargs) -> _T:
                generator = retry_generator(*args, **kwargs)
                while True:
                    next(generator)
                    try:
                        retry_count_ = generator.send(
                            await catcher_executor.secure_await(callable_to_secure(*args, **kwargs))
                        )
                    except StopIteration as stop_iteration:
                        return stop_iteration.value
                    await asyncio.sleep(retry_stepping_func(retry_count_))

            @functools.wraps(callable_to_secure)
            async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
                result = await catcher_retrier.secure_await(retry_function_async(*args, **kwargs))
                return handle_result_and_call_callbacks(result, *args, **kwargs)

        else:
            logger.warning(
                "Sync retry decorator is dangerous as it uses time.sleep() for retry logic. "
                "Combined with asyncio code it could lead to deadlocks and other unexpected behaviour. "
                "Please consider decorating an async function instead."
            )

            def retry_function_sync(*args: _P.args, **kwargs: _P.kwargs) -> _T:
                generator = retry_generator(*args, **kwargs)
                while True:
                    next(generator)
                    try:
                        retry_count_ = generator.send(
                            catcher_executor.secure_call(callable_to_secure, *args, **kwargs)  # type: ignore[arg-type]
                        )
                    except StopIteration as stop_iteration:
                        return stop_iteration.value
                    time.sleep(retry_stepping_func(retry_count_))

            @functools.wraps(callable_to_secure)
            def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
                result = catcher_retrier.secure_call(retry_function_sync, *args, **kwargs)
                return handle_result_and_call_callbacks(result, *args, **kwargs)

        return_func = cast(SecuredFunctionType[_P, _T] | SecuredAsyncFunctionType[_P, _T], wrapper)
        return_func.__catcher__ = catcher_retrier
        return_func.__original_callable__ = callable_to_secure
        return return_func

    return decorator_inner
