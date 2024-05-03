"""
This module contains decorators to secure an async or sync callable and handle its errors.
"""

import asyncio
import functools
import inspect
import logging
import time
from typing import Any, Callable, Concatenate, Generator, ParamSpec, Protocol, TypeGuard, TypeVar, cast, overload

from .callback import Callback, ErrorCallback, SuccessCallback
from .core import Catcher
from .result import CallbackResultType, PositiveResult, ResultType
from .types import UNSET, AsyncFunctionType, FunctionType, SecuredAsyncFunctionType, SecuredFunctionType, UnsetType

_P = ParamSpec("_P")
_T = TypeVar("_T")


def iscoroutinefunction(
    callable_: FunctionType[_P, _T] | AsyncFunctionType[_P, _T]
) -> TypeGuard[AsyncFunctionType[_P, _T]]:
    """
    This function checks if the given callable is a coroutine function.
    """
    return asyncio.iscoroutinefunction(callable_)


# pylint: disable=too-few-public-methods
class SecureDecorator(Protocol[_P, _T]):
    """
    This protocol represents a decorator that secures a callable and returns a ResultType[T].
    """

    @overload
    def __call__(  # type: ignore[overload-overlap]
        # This error happens, because Callable[..., Awaitable[T]] is a subtype of Callable[..., T] and
        # therefore the overloads are overlapping. This leads to problems with the type checker if you use it like this:
        #
        # async def some_coroutine_function() -> None: ...
        # callable_to_secure: FunctionType[_P, _T] = some_coroutine_function
        # reveal_type(decorator(callable_to_secure))
        #
        # Revealed type is 'SecuredAsyncFunctionType[_P, _T]' but mypy will think it is 'SecuredFunctionType[_P, _T]'.
        # Since it is not possible to 'negate' types (e.g. something like 'Callable[..., T \ Awaitable[T]]'),
        # we have no other choice than to ignore this error. Anyway, it should be fine if you are plainly decorating
        # your functions, so it's ok.
        # Reference: https://stackoverflow.com/a/74567241/21303427
        self,
        callable_to_secure: AsyncFunctionType[_P, _T],
    ) -> SecuredAsyncFunctionType[_P, _T]: ...

    @overload
    def __call__(self, callable_to_secure: FunctionType[_P, _T]) -> SecuredFunctionType[_P, _T]: ...

    def __call__(
        self, callable_to_secure: FunctionType[_P, _T] | AsyncFunctionType[_P, _T]
    ) -> SecuredFunctionType[_P, _T] | SecuredAsyncFunctionType[_P, _T]: ...


# pylint: disable=too-few-public-methods
class Decorator(Protocol[_P, _T]):
    """
    This protocol represents a decorator that secures a callable but does not change the return type.
    """

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self,
        callable_to_secure: AsyncFunctionType[_P, _T],
    ) -> AsyncFunctionType[_P, _T]: ...

    @overload
    def __call__(self, callable_to_secure: FunctionType[_P, _T]) -> FunctionType[_P, _T]: ...

    def __call__(
        self,
        callable_to_secure: FunctionType[_P, _T] | AsyncFunctionType[_P, _T],
    ) -> FunctionType[_P, _T] | AsyncFunctionType[_P, _T]: ...


# pylint: disable=too-many-arguments
def decorator_as_result(
    *,
    on_success: Callable[Concatenate[_T, _P], Any] | None = None,
    on_error: Callable[Concatenate[BaseException, _P], Any] | None = None,
    on_finalize: Callable[_P, Any] | None = None,
    suppress_recalling_on_error: bool = True,
) -> SecureDecorator[_P, _T]:
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

    @overload
    def decorator_inner(  # type: ignore[overload-overlap] # See above
        callable_to_secure: AsyncFunctionType[_P, _T],
    ) -> SecuredAsyncFunctionType[_P, _T]: ...

    @overload
    def decorator_inner(callable_to_secure: FunctionType[_P, _T]) -> SecuredFunctionType[_P, _T]: ...

    def decorator_inner(
        callable_to_secure: FunctionType[_P, _T] | AsyncFunctionType[_P, _T]
    ) -> SecuredFunctionType[_P, _T] | SecuredAsyncFunctionType[_P, _T]:
        sig = inspect.signature(callable_to_secure)
        catcher = Catcher[_T](
            SuccessCallback.from_callable(on_success, sig, return_type=Any) if on_success is not None else None,
            ErrorCallback.from_callable(on_error, sig, return_type=Any) if on_error is not None else None,
            Callback.from_callable(on_finalize, sig, return_type=Any) if on_finalize is not None else None,
            suppress_recalling_on_error,
        )
        if iscoroutinefunction(callable_to_secure):

            @functools.wraps(callable_to_secure)
            async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> ResultType[_T]:
                result = await catcher.secure_await(callable_to_secure(*args, **kwargs))
                catcher.handle_result_and_call_callbacks(result, *args, **kwargs)
                return result

        else:

            @functools.wraps(callable_to_secure)
            def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> ResultType[_T]:
                result = catcher.secure_call(
                    callable_to_secure,  # type: ignore[arg-type]
                    *args,
                    **kwargs,
                )
                catcher.handle_result_and_call_callbacks(result, *args, **kwargs)
                return result

        return_func = cast(SecuredFunctionType[_P, _T] | SecuredAsyncFunctionType[_P, _T], wrapper)
        return_func.__catcher__ = catcher
        return_func.__original_callable__ = callable_to_secure
        return return_func

    return decorator_inner


# pylint: disable=too-many-arguments, too-many-locals
def retry_on_error(
    *,
    on_error: Callable[Concatenate[BaseException, int, _P], bool],
    retry_stepping_func: Callable[[int], float] = lambda retry_count: 1.71**retry_count,
    # <-- with max_retries = 10 the whole decorator may wait up to 5 minutes.
    # because sum(1.71seconds**i for i in range(10)) == 5minutes
    max_retries: int = 10,
    on_success: Callable[Concatenate[_T, int, _P], Any] | None = None,
    on_fail: Callable[Concatenate[BaseException, int, _P], Any] | None = None,
    on_finalize: Callable[Concatenate[int, _P], Any] | None = None,
    logger: logging.Logger = logging.getLogger(__name__),
) -> Decorator[_P, _T]:
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
    ) -> FunctionType[_P, _T] | AsyncFunctionType[_P, _T]:
        sig = inspect.signature(callable_to_secure)
        sig = sig.replace(
            parameters=[
                inspect.Parameter("retries", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
                *sig.parameters.values(),
            ],
        )
        on_error_callback: ErrorCallback[Concatenate[BaseException, int, _P], bool] = ErrorCallback.from_callable(
            on_error, sig, return_type=bool
        )
        on_success_callback: SuccessCallback[Concatenate[_T, int, _P], Any] | None = (
            SuccessCallback.from_callable(on_success, sig, return_type=Any) if on_success is not None else None
        )
        on_fail_callback: ErrorCallback[Concatenate[BaseException, int, _P], Any] | None = (
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

        return_func = cast(FunctionType[_P, _T] | AsyncFunctionType[_P, _T], wrapper)
        return_func.__catcher__ = catcher_retrier  # type: ignore[union-attr]
        return_func.__original_callable__ = callable_to_secure  # type: ignore[union-attr]
        return return_func

    return decorator_inner  # type: ignore[return-value]


_C = TypeVar("_C", bound=Callable)


def decorator(
    *,
    on_success: Callable[Concatenate[_T, _P], Any] | None = None,
    on_error: Callable[Concatenate[BaseException, _P], Any] | None = None,
    on_finalize: Callable[_P, Any] | None = None,
    suppress_recalling_on_error: bool = True,
    on_error_return_always: _T | UnsetType = UNSET,
) -> Callable[[_C], _C]:
    """
    Returns a callback that converts the result of a secured function back to the original return type.
    To make this work, you need to define which value should be returned in error cases.
    Otherwise, if the secured function returns an error result, the error will be raised.
    """

    def decorator_inner(
        func: FunctionType[_P, _T] | AsyncFunctionType[_P, _T]
    ) -> FunctionType[_P, _T] | AsyncFunctionType[_P, _T]:
        secured_func = decorator_as_result(
            on_success=on_success,
            on_error=on_error,
            on_finalize=on_finalize,
            suppress_recalling_on_error=suppress_recalling_on_error,
        )(func)

        def handle_result(result: ResultType[_T]) -> _T:
            if isinstance(result, PositiveResult):
                return result.result
            if isinstance(on_error_return_always, UnsetType):
                raise result.error
            return on_error_return_always

        if iscoroutinefunction(secured_func):

            @functools.wraps(secured_func)
            async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
                return handle_result(await secured_func(*args, **kwargs))

        else:

            @functools.wraps(secured_func)
            def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
                return handle_result(secured_func(*args, **kwargs))  # type: ignore[arg-type]

        return cast(FunctionType[_P, _T] | AsyncFunctionType[_P, _T], wrapper)

    return decorator_inner  # type: ignore[return-value]
