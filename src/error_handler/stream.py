"""
This module contains pipable operators that are used to handle errors in aiostream pipelines.
"""

import asyncio
import logging
import sys
from typing import Any, AsyncIterable, AsyncIterator, Awaitable, Callable, Coroutine

from . import NegativeResult, PositiveResult, ResultType
from ._extra import IS_AIOSTREAM_INSTALLED
from .decorator import decorator_as_result
from .types import is_secured

if IS_AIOSTREAM_INSTALLED:
    import aiostream
    from aiostream.stream.combine import T, U

    # pylint: disable=too-many-arguments, redefined-builtin
    @aiostream.pipable_operator
    def map(
        source: AsyncIterable[T],
        func: Callable[[T], Coroutine[None, None, U]] | Callable[[T], U],
        *more_sources: AsyncIterable[T],
        ordered: bool = True,
        task_limit: int | None = None,
        on_success: Callable[[U, T], Any] | None = None,
        on_error: Callable[[BaseException, T], Any] | None = None,
        on_finalize: Callable[[T], Any] | None = None,
        wrap_secured_function: bool = False,
        suppress_recalling_on_error: bool = True,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> AsyncIterator[U]:
        """
        This operator does mostly the same as stream.map of aiostream.
        Additionally, it catches all errors, calls the corresponding callbacks and filters out errored results.
        If suppress_recalling_on_error is True, the on_error callable will not be called if the error were already
        caught by a previous catcher.
        """
        if not wrap_secured_function and is_secured(func):
            if (
                on_success is not None
                or on_error is not None
                or on_finalize is not None
                or not suppress_recalling_on_error
            ):
                raise ValueError(
                    "The given function is already secured. "
                    "Please do not set on_success, on_error, on_finalize as they would be ignored. "
                    "You can set wrap_secured_function=True to wrap the secured function with another catcher."
                )
            logger.debug(
                f"The given function {func.__original_callable__.__name__} is already secured. Using it as is."
            )
            secured_func = func
        else:
            # pylint: disable=duplicate-code
            secured_func = decorator_as_result(  # type: ignore[assignment]
                on_success=on_success,
                on_error=on_error,
                on_finalize=on_finalize,
                suppress_recalling_on_error=suppress_recalling_on_error,
            )(
                func  # type: ignore[arg-type]
            )
            # Ignore that T | ErroredType is not compatible with T. All ErroredType results are filtered out
            # in a subsequent step.
        results: AsyncIterator[ResultType[U]] = aiostream.stream.map.raw(
            source, secured_func, *more_sources, ordered=ordered, task_limit=task_limit  # type: ignore[arg-type]
        )
        positive_results: AsyncIterator[PositiveResult[U]] = aiostream.stream.filter.raw(
            results,  # type: ignore[arg-type]
            # mypy can't successfully narrow the type here.
            lambda result: not isinstance(result, NegativeResult),
        )
        result_values: AsyncIterator[U] = aiostream.stream.map.raw(
            positive_results,
            lambda result: (  # type: ignore[arg-type, misc]
                result.result if isinstance(result, PositiveResult) else result
            ),
        )
        return result_values

    # pylint: disable=too-many-positional-arguments
    @aiostream.pipable_operator
    def action(
        source: AsyncIterable[T],
        func: Callable[[T], Awaitable[Any] | Any],
        ordered: bool = True,
        task_limit: int | None = None,
        on_success: Callable[[T], Any] | None = None,
        on_error: Callable[[BaseException, T], Any] | None = None,
        on_finalize: Callable[[T], Any] | None = None,
        wrap_secured_function: bool = False,
        suppress_recalling_on_error: bool = True,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> AsyncIterator[T]:
        """
        This operator does mostly the same as stream.action of aiostream.
        Additionally, it catches all errors and filters out errored results.
        """
        innerfunc: Callable[[T], Coroutine[None, None, T]] | Callable[[T], T]
        if asyncio.iscoroutinefunction(func):

            async def innerfunc(arg: T, *_: object) -> T:
                awaitable = func(arg)
                assert isinstance(awaitable, Awaitable)
                await awaitable
                return arg

        else:

            def innerfunc(arg: T, *_: object) -> T:
                func(arg)
                return arg

        return map.raw(
            source,
            innerfunc,
            ordered=ordered,
            task_limit=task_limit,
            on_success=lambda u, t: on_success(t) if on_success is not None else None,
            on_error=on_error,
            on_finalize=on_finalize,
            wrap_secured_function=wrap_secured_function,
            suppress_recalling_on_error=suppress_recalling_on_error,
            logger=logger,
        )

else:
    from ._extra import _NotInstalled

    sys.modules[__name__] = _NotInstalled()  # type: ignore[assignment]
