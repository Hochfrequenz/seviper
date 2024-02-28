"""
This module contains the core logic of the error_handler package. It contains the Catcher class, which implements the
methods to surround statements with try-except blocks and calls corresponding callbacks.
"""

# pylint: disable=undefined-variable
# Seems like pylint doesn't like the new typing features. It has a problem with the generic T of class Catcher.
import inspect
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Generic, Iterator, ParamSpec, Self, TypeVar, overload

from .types import ERRORED, UNSET, ErroredType, NegativeResult, PositiveResult, ResultType, T, UnsetType

_T = TypeVar("_T")
_U = TypeVar("_U")
_P = ParamSpec("_P")


@overload
def _call_callback(
    callback: Callable[[_T], _U] | Callable[[], _U],
    optional_arg: _T,
    raise_if_arg_present: bool = False,
) -> _U: ...


@overload
def _call_callback(
    callback: Callable[[], _U],
    optional_arg: UnsetType = UNSET,
    raise_if_arg_present: bool = False,
) -> _U: ...


def _call_callback(
    callback: Callable[[_T], _U] | Callable[[], _U],
    optional_arg: _T | UnsetType = UNSET,
    raise_if_arg_present: bool = False,
) -> _U:
    signature = inspect.signature(callback)
    if len(signature.parameters) == 0:
        return callback()  # type: ignore[call-arg]
    if len(signature.parameters) == 1 and raise_if_arg_present:
        raise ValueError(f"Callback {callback.__name__} cannot receive arguments when using for a context manager.")
    assert optional_arg is not UNSET, "Internal error: optional_arg is UNSET but should be."
    return callback(optional_arg)  # type: ignore[call-arg,arg-type]


class Catcher(Generic[T]):
    """
    After defining callbacks and other options for an instance, you can use the secure_call and secure_await methods
    to call or await corresponding objects in a secure context. I.e. errors will be caught and the callbacks will be
    called accordingly.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        on_success: Callable[[T], Any] | Callable[[], Any] | None = None,
        on_error: Callable[[Exception], Any] | Callable[[], Any] | None = None,
        on_finalize: Callable[[], Any] | None = None,
        on_error_return_always: T | ErroredType = ERRORED,
        suppress_recalling_on_error: bool = True,
    ):
        self.on_success = on_success
        self.on_error = on_error
        self.on_finalize = on_finalize
        self.on_error_return_always = on_error_return_always
        self.suppress_recalling_on_error = suppress_recalling_on_error
        """
        If this flag is set, the framework won't call the callbacks if the caught exception was already caught by
        another catcher.
        This is especially useful if you have nested catchers (e.g. due to nested context managers / function calls)
        which are re-raising the error.
        """

    def mark_exception(self, error: Exception) -> None:
        """
        This method marks the given exception as handled by the catcher.
        """
        if not hasattr(error, "__caught_by_catcher__"):
            error.__caught_by_catcher__ = []  # type: ignore[attr-defined]
        error.__caught_by_catcher__.append(self)  # type: ignore[attr-defined]

    @staticmethod
    def _ensure_exception_in_cause_propagation(error_base: Exception, error_cause: Exception) -> None:
        """
        This method ensures that the given error_cause is in the cause chain of the given error_base.
        """
        if error_base is error_cause:
            return
        if error_base.__cause__ is None:
            error_base.__cause__ = error_cause
        else:
            assert isinstance(error_base.__cause__, Exception), "Internal error: __cause__ is not an Exception"
            Catcher._ensure_exception_in_cause_propagation(error_base.__cause__, error_cause)

    def handle_error_case(self, error: Exception) -> ResultType[T]:
        """
        This method handles the given exception.
        """
        caught_before = hasattr(error, "__caught_by_catcher__")
        self.mark_exception(error)
        if self.on_error is not None and not (caught_before and self.suppress_recalling_on_error):
            try:
                _call_callback(self.on_error, error)
            except Exception as callback_error:  # pylint: disable=broad-exception-caught
                self._ensure_exception_in_cause_propagation(callback_error, error)
                raise callback_error
        return NegativeResult(error=error, result=self.on_error_return_always)

    def handle_success_case(self, result: T, raise_if_arg_present: bool = False) -> ResultType[T]:
        """
        This method handles the given result.
        """
        if self.on_success is not None:
            _call_callback(self.on_success, result, raise_if_arg_present=raise_if_arg_present)  # type: ignore[arg-type]
        return PositiveResult(result=result)

    def handle_finalize_case(self) -> None:
        """
        This method handles the finalize case.
        """
        if self.on_finalize is not None:
            self.on_finalize()

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
            return self.handle_success_case(result)
        except Exception as error:  # pylint: disable=broad-exception-caught
            return self.handle_error_case(error)
        finally:
            self.handle_finalize_case()

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
            return self.handle_success_case(result)
        except Exception as error:  # pylint: disable=broad-exception-caught
            return self.handle_error_case(error)
        finally:
            self.handle_finalize_case()

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
            self.handle_success_case(None, raise_if_arg_present=True)  # type: ignore[arg-type]
        except Exception as error:  # pylint: disable=broad-exception-caught
            self.handle_error_case(error)
        finally:
            self.handle_finalize_case()
