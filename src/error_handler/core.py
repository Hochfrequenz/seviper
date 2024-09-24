"""
This module contains the core logic of the error_handler package. It contains the Catcher class, which implements the
methods to surround statements with try-except blocks and calls corresponding callbacks.
"""

# pylint: disable=undefined-variable
# Seems like pylint doesn't like the new typing features. It has a problem with the generic T of class Catcher.
import inspect
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Generic, Iterator, ParamSpec, Self, TypeVar

from .callback import Callback
from .result import (
    CallbackResultType,
    CallbackResultTypes,
    CallbackSummary,
    NegativeResult,
    PositiveResult,
    ResultType,
    ReturnValues,
)
from .types import UNSET, T, UnsetType

_T = TypeVar("_T")
_U = TypeVar("_U")
_P = ParamSpec("_P")


_CALLBACK_ERROR_PARAM = inspect.Parameter("error", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=BaseException)


# pylint: disable=too-many-instance-attributes
class Catcher(Generic[T]):
    """
    After defining callbacks and other options for an instance, you can use the secure_call and secure_await methods
    to call or await corresponding objects in a secure context. I.e. errors will be caught and the callbacks will be
    called accordingly.
    """

    # pylint: disable=too-many-positional-arguments, too-many-arguments
    def __init__(
        self,
        on_success: Callback | None = None,
        on_error: Callback | None = None,
        on_finalize: Callback | None = None,
        suppress_recalling_on_error: bool = True,
        raise_callback_errors: bool = True,
        no_wrap_exception_group_when_reraise: bool = True,
    ):
        self.on_success = on_success
        self.on_error = on_error
        self.on_finalize = on_finalize
        self.suppress_recalling_on_error = suppress_recalling_on_error
        """
        If this flag is set, the framework won't call the callbacks if the caught exception was already caught by
        another catcher.
        This is especially useful if you have nested catchers (e.g. due to nested context managers / function calls)
        which are re-raising the error.
        """
        self._result: ResultType[T] | None = None
        self.raise_callback_errors = raise_callback_errors
        self.no_wrap_exception_group_when_reraise = no_wrap_exception_group_when_reraise

    @property
    def result(self) -> ResultType[T]:
        """
        This method returns the result of the last execution. If the catcher has not been executed yet, a ValueError
        will be raised.
        """
        if self._result is None:
            raise ValueError("The catcher has not been executed yet.")
        return self._result

    def _mark_exception(self, error: BaseException) -> None:
        """
        This method marks the given exception as handled by the catcher.
        """
        if not hasattr(error, "__caught_by_catcher__"):
            error.__caught_by_catcher__ = []  # type: ignore[attr-defined]
        error.__caught_by_catcher__.append(self)  # type: ignore[attr-defined]

    @staticmethod
    def _call_callback(callback: Callback | None, *args: Any, **kwargs: Any) -> tuple[Any, CallbackResultType]:
        callback_result = CallbackResultType.SKIPPED
        callback_return_value: Any = UNSET
        if callback is not None:
            try:
                callback_return_value = callback(*args, **kwargs)
                callback_result = CallbackResultType.SUCCESS
            except BaseException as callback_error:  # pylint: disable=broad-exception-caught
                callback_return_value = callback_error
                callback_result = CallbackResultType.ERROR
        return callback_return_value, callback_result

    def _raise_callback_errors_if_set(self, result: CallbackSummary, raise_from: BaseException | None = None) -> None:
        if not self.raise_callback_errors:
            return
        excs = []
        if result.callback_result_types.success == CallbackResultType.ERROR:
            excs.append(result.callback_return_values.success)
        if result.callback_result_types.error == CallbackResultType.ERROR:
            excs.append(result.callback_return_values.error)
        if result.callback_result_types.finalize == CallbackResultType.ERROR:
            excs.append(result.callback_return_values.finalize)

        if self.no_wrap_exception_group_when_reraise and len(excs) == 1 and raise_from is excs[0]:
            raise raise_from
        if len(excs) > 0:
            exc_group = BaseExceptionGroup("There were one or more errors while calling the callback functions.", excs)
            if raise_from is not None:
                exc_group.__context__ = raise_from
            raise exc_group

    def _handle_error_callback(self, error: BaseException, *args: Any, **kwargs: Any) -> tuple[Any, CallbackResultType]:
        """
        This method handles the given exception.
        """
        return_value = UNSET
        result = CallbackResultType.SKIPPED
        caught_before = hasattr(error, "__caught_by_catcher__")
        self._mark_exception(error)
        if not (caught_before and self.suppress_recalling_on_error):
            return_value, result = self._call_callback(self.on_error, error, *args, **kwargs)
            if result == CallbackResultType.ERROR and return_value is error:
                assert self.on_error is not None, "Internal error: on_error is None but result is ERROR"
                error.add_note(f"This error was reraised by on_error callback {self.on_error.callback.__name__}")

        return return_value, result

    def _handle_success_callback(self, *args: Any, **kwargs: Any) -> tuple[Any, CallbackResultType]:
        """
        This method handles the given result.
        """
        return self._call_callback(self.on_success, *args, **kwargs)

    def _handle_finalize_callback(self, *args: Any, **kwargs: Any) -> tuple[Any, CallbackResultType]:
        """
        This method handles the finalize case.
        """
        return self._call_callback(self.on_finalize, *args, **kwargs)

    def handle_success_case(self, result: T | UnsetType, *args: Any, **kwargs: Any) -> CallbackSummary:
        """
        This method handles the success case.
        """
        if result is UNSET:
            success_return_value, success_result = self._handle_success_callback(*args, **kwargs)
        else:
            success_return_value, success_result = self._handle_success_callback(result, *args, **kwargs)
        finalize_return_value, finalize_result = self._handle_finalize_callback(*args, **kwargs)
        callback_result = CallbackSummary(
            callback_result_types=CallbackResultTypes(
                success=success_result,
                finalize=finalize_result,
            ),
            callback_return_values=ReturnValues(
                success=success_return_value,
                finalize=finalize_return_value,
            ),
        )
        self._raise_callback_errors_if_set(callback_result)
        return callback_result

    def handle_error_case(self, error: BaseException, *args: Any, **kwargs: Any) -> CallbackSummary:
        """
        This method handles the error case.
        """
        error_return_value, error_result = self._handle_error_callback(error, *args, **kwargs)
        finalize_return_value, finalize_result = self._handle_finalize_callback(*args, **kwargs)
        callback_result = CallbackSummary(
            callback_result_types=CallbackResultTypes(
                error=error_result,
                finalize=finalize_result,
            ),
            callback_return_values=ReturnValues(
                error=error_return_value,
                finalize=finalize_return_value,
            ),
        )
        self._raise_callback_errors_if_set(callback_result, error)
        return callback_result

    def handle_result_and_call_callbacks(self, result: ResultType[T], *args: Any, **kwargs: Any) -> CallbackSummary:
        """
        This method handles the last case.
        """
        if isinstance(result, PositiveResult):
            return self.handle_success_case(result.result, *args, **kwargs)
        return self.handle_error_case(result.error, *args, **kwargs)

    def secure_call(  # type: ignore[return]  # Because mypy is stupid, idk.
        self,
        callable_to_secure: Callable[_P, T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> ResultType[T]:
        """
        This method calls the given callable with the given arguments and handles its errors.
        If the callable raises an error, the on_error callback will be called and the value if on_error_return_always
        will be returned.
        If the callable does not raise an error, the on_success callback will be called (the return value will be
        provided to the callback if it receives an argument) and the return value will be propagated.
        The on_finalize callback will be called in both cases and after the other callbacks.
        """
        try:
            result = callable_to_secure(*args, **kwargs)
            self._result = PositiveResult(result=result)
        except BaseException as error:  # pylint: disable=broad-exception-caught
            self._result = NegativeResult(error=error)
        return self.result

    async def secure_await(  # type: ignore[return]  # Because mypy is stupid, idk.
        self,
        awaitable_to_secure: Awaitable[T],
    ) -> ResultType[T]:
        """
        This method awaits the given awaitable and handles its errors.
        If the awaitable raises an error, the on_error callback will be called and the value if on_error_return_always
        will be returned.
        If the awaitable does not raise an error, the on_success callback will be called (the return value will be
        provided to the callback if it receives an argument) and the return value will be propagated.
        The on_finalize callback will be called in both cases and after the other callbacks.
        """
        try:
            result = await awaitable_to_secure
            self._result = PositiveResult(result=result)
        except BaseException as error:  # pylint: disable=broad-exception-caught
            self._result = NegativeResult(error=error)
        return self.result


class ContextCatcher(Catcher[UnsetType]):
    """
    This class is a special case of the Catcher class. It is meant to use the context manager.
    """

    @contextmanager
    def secure_context(self) -> Iterator[Self]:
        """
        This context manager catches all errors inside the context and calls the corresponding callbacks.
        If the context raises an error, the on_error callback will be called.
        If the context does not raise an error, the on_success callback will be called.
        The on_finalize callback will be called in both cases and after the other callbacks.
        If reraise is True, the error will be reraised after the callbacks were called.
        Note: When using this context manager, the on_success callback cannot receive arguments.
        If the callback has an argument, a ValueError will be raised.
        """
        try:
            yield self
            self._result = PositiveResult(result=UNSET)
        except BaseException as error:  # pylint: disable=broad-exception-caught
            self._result = NegativeResult(error=error)
