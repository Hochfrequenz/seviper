"""
This module contains the core logic of the error_handler package. It contains the Catcher class, which implements the
methods to surround statements with try-except blocks and calls corresponding callbacks.
"""

# pylint: disable=undefined-variable
# Seems like pylint doesn't like the new typing features. It has a problem with the generic T of class Catcher.
import inspect
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Generic, Iterator, ParamSpec, Self, Sequence, TypeVar, overload

from .callback import Callback
from .types import ERRORED, UNSET, ErroredType, NegativeResult, PositiveResult, ResultType, T, _UnsetType

_T = TypeVar("_T")
_U = TypeVar("_U")
_P = ParamSpec("_P")


_CALLBACK_ERROR_PARAM = inspect.Parameter("error", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Exception)


class Catcher(Generic[T]):
    """
    After defining callbacks and other options for an instance, you can use the secure_call and secure_await methods
    to call or await corresponding objects in a secure context. I.e. errors will be caught and the callbacks will be
    called accordingly.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        on_success: Callable[..., Any] | Callback | None = None,
        on_error: Callable[..., Any] | Callback | None = None,
        on_finalize: Callable[..., Any] | Callback | None = None,
        on_error_return_always: T | ErroredType = ERRORED,
        suppress_recalling_on_error: bool = True,
    ):
        self.on_success = (
            on_success
            if isinstance(on_success, Callback)
            else Callback(on_success, inspect.Signature(return_annotation=Any)) if on_success is not None else None
        )
        self.on_error = (
            on_error
            if isinstance(on_error, Callback)
            else Callback(on_error, inspect.Signature(return_annotation=Any)) if on_error is not None else None
        )
        self.on_finalize = (
            on_finalize
            if isinstance(on_finalize, Callback)
            else Callback(on_finalize, inspect.Signature(return_annotation=Any)) if on_finalize is not None else None
        )
        self.on_error_return_always = on_error_return_always
        self.suppress_recalling_on_error = suppress_recalling_on_error
        """
        If this flag is set, the framework won't call the callbacks if the caught exception was already caught by
        another catcher.
        This is especially useful if you have nested catchers (e.g. due to nested context managers / function calls)
        which are re-raising the error.
        """

    def _auto_set_expected_signature_of_error_callback(self, base_signature: inspect.Signature | None = None):
        if self.on_error is not None:
            if base_signature is None:
                base_signature = self.on_error.expected_signature
            self.on_error.expected_signature = base_signature.replace(
                parameters=[_CALLBACK_ERROR_PARAM, *base_signature.parameters.values()], return_annotation=Any
            )

    def _auto_set_expected_signature_of_finalize_callback(self, base_signature: inspect.Signature | None = None):
        if self.on_finalize is not None:
            if base_signature is not None:
                self.on_finalize.expected_signature = base_signature

    def _auto_set_expected_signature_of_success_callback(
        self, base_signature: inspect.Signature | None = None, provide_return_annotation_as_param: bool = True
    ):
        if self.on_success is not None:
            if base_signature is None:
                base_signature = self.on_success.expected_signature
            if provide_return_annotation_as_param:
                add_param = inspect.Parameter(
                    "result",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=base_signature.return_annotation,
                )
                self.on_success.expected_signature = base_signature.replace(
                    parameters=[add_param, *base_signature.parameters.values()], return_annotation=Any
                )
            else:
                self.on_success.expected_signature = base_signature.replace(return_annotation=Any)

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

    def handle_error_case(self, error: Exception, *args: Any, **kwargs: Any) -> ResultType[T]:
        """
        This method handles the given exception.
        """
        caught_before = hasattr(error, "__caught_by_catcher__")
        self.mark_exception(error)
        if self.on_error is not None and not (caught_before and self.suppress_recalling_on_error):
            try:
                self.on_error(error, *args, **kwargs)
            except Exception as callback_error:  # pylint: disable=broad-exception-caught
                self._ensure_exception_in_cause_propagation(callback_error, error)
                raise callback_error
        return NegativeResult(error=error, result=self.on_error_return_always)

    def handle_success_case(self, result: T, *args: Any, **kwargs: Any) -> ResultType[T]:
        """
        This method handles the given result.
        """
        if self.on_success is not None:
            self.on_success(result, *args, **kwargs)
        return PositiveResult(result=result)

    def handle_finalize_case(self, *args: Any, **kwargs: Any) -> None:
        """
        This method handles the finalize case.
        """
        if self.on_finalize is not None:
            self.on_finalize(*args, **kwargs)

    def secure_call(  # type: ignore[return]  # Because mypy is stupid, idk.
        self,
        callable_to_secure: Callable[_P, T],
        *args: _P.args,
        __auto_set_expected_signatures__: bool = True,
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

            if __auto_set_expected_signatures__:
                self._auto_set_expected_signature_of_success_callback(inspect.signature(callable_to_secure))
            return self.handle_success_case(
                result,
                *args,
                **kwargs,
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            if __auto_set_expected_signatures__:
                self._auto_set_expected_signature_of_error_callback(inspect.signature(callable_to_secure))
            return self.handle_error_case(
                error,
                *args,
                **kwargs,
            )
        finally:
            if __auto_set_expected_signatures__:
                self._auto_set_expected_signature_of_finalize_callback(inspect.signature(callable_to_secure))
            self.handle_finalize_case(
                *args,
                **kwargs,
            )

    async def secure_await(  # type: ignore[return]  # Because mypy is stupid, idk.
        self,
        awaitable_to_secure: Awaitable[T],
        __auto_set_expected_signatures__: bool = True,
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

            if __auto_set_expected_signatures__:
                self._auto_set_expected_signature_of_success_callback()
            return self.handle_success_case(result)
        except Exception as error:  # pylint: disable=broad-exception-caught
            if __auto_set_expected_signatures__:
                self._auto_set_expected_signature_of_error_callback()
            return self.handle_error_case(error)
        finally:
            if __auto_set_expected_signatures__:
                self._auto_set_expected_signature_of_finalize_callback()
            self.handle_finalize_case()

    async def secure_call_coroutine(  # type: ignore[return]  # Because mypy is stupid, idk.
        self,
        callable_to_secure: Callable[_P, Awaitable[T]],
        *args: _P.args,
        __auto_set_expected_signatures__: bool = True,
        **kwargs: _P.kwargs,
    ) -> ResultType[T]:
        """
        This method calls and awaits the given coroutine with the given arguments and handles its errors.
        If the coroutine raises an error, the on_error callback will be called and the value if on_error_return_always
        will be returned.
        If the coroutine does not raise an error, the on_success callback will be called (the return value will be
        provided to the callback if it receives an argument) and the return value will be propagated.
        The on_finalize callback will be called in both cases and after the other callbacks.
        """
        try:
            result = await callable_to_secure(*args, **kwargs)

            if __auto_set_expected_signatures__:
                self._auto_set_expected_signature_of_success_callback(inspect.signature(callable_to_secure))
            return self.handle_success_case(
                result,
                *args,
                **kwargs,
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            if __auto_set_expected_signatures__:
                self._auto_set_expected_signature_of_error_callback(inspect.signature(callable_to_secure))
            return self.handle_error_case(
                error,
                *args,
                **kwargs,
            )
        finally:
            if __auto_set_expected_signatures__:
                self._auto_set_expected_signature_of_finalize_callback(inspect.signature(callable_to_secure))
            self.handle_finalize_case(
                *args,
                **kwargs,
            )

    @contextmanager
    def secure_context(self, __auto_set_expected_signatures__: bool = True) -> Iterator[Self]:
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
            if __auto_set_expected_signatures__:
                self._auto_set_expected_signature_of_success_callback(provide_return_annotation_as_param=False)
            if self.on_success is not None:
                self.on_success()
        except Exception as error:  # pylint: disable=broad-exception-caught
            if __auto_set_expected_signatures__:
                self._auto_set_expected_signature_of_error_callback()
            self.handle_error_case(error)
        finally:
            if __auto_set_expected_signatures__:
                self._auto_set_expected_signature_of_finalize_callback()
            self.handle_finalize_case()
