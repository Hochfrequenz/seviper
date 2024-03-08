"""
This module contains the Callback class, which is used to wrap a callable and its expected signature.
The expected signature is only used to give nicer error messages when the callback is called with the wrong
arguments. Just in case that the type checker is not able to spot callback functions with wrong signatures.
"""

import inspect
from typing import Any, Callable, Generic, ParamSpec, Sequence, TypeVar, cast

from .types import UNSET

_P = ParamSpec("_P")
_T = TypeVar("_T")
_CallbackT = TypeVar("_CallbackT", bound="Callback")
_ErrorCallbackT = TypeVar("_ErrorCallbackT", bound="ErrorCallback")
_SuccessCallbackT = TypeVar("_SuccessCallbackT", bound="SuccessCallback")


class Callback(Generic[_P, _T]):
    """
    This class wraps a callable and its expected signature.
    """

    def __init__(self, callback: Callable[_P, _T], expected_signature: inspect.Signature):
        self.callback = callback
        self.expected_signature = expected_signature
        self._actual_signature: inspect.Signature | None = None

    @property
    def actual_signature(self) -> inspect.Signature:
        """
        The actual signature of the callback
        """
        if self._actual_signature is None:
            self._actual_signature = inspect.signature(self.callback)
        return self._actual_signature

    @property
    def expected_signature_str(self) -> str:
        """
        The expected signature as string
        """
        return str(self.expected_signature)

    @property
    def actual_signature_str(self) -> str:
        """
        The actual signature as string
        """
        return str(self.actual_signature)

    @classmethod
    def from_callable(
        cls: type[_CallbackT],
        callback: Callable,
        signature_from_callable: Callable[..., Any] | inspect.Signature | None = None,
        add_params: Sequence[inspect.Parameter] | None = None,
        return_type: Any = UNSET,
    ) -> _CallbackT:
        """
        Create a new Callback instance from a callable. The expected signature will be taken from the
        signature_from_callable. You can add additional parameters or change the return type for the
        expected signature.
        """
        if signature_from_callable is None:
            sig = inspect.Signature()
        elif isinstance(signature_from_callable, inspect.Signature):
            sig = signature_from_callable
        else:
            sig = inspect.signature(signature_from_callable)
        if add_params is not None or return_type is not None:
            params = list(sig.parameters.values())
            if add_params is not None:
                params = [*add_params, *params]
            if return_type is UNSET:
                return_type = sig.return_annotation
            sig = sig.replace(parameters=params, return_annotation=return_type)
        return cls(callback, sig)

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        """
        Call the callback with the given arguments and keyword arguments. The arguments will be checked against the
        expected signature. If the callback does not match the expected signature, a TypeError explaining which
        signature was expected will be raised.
        """
        try:
            filled_signature = self.actual_signature.bind(*args, **kwargs)
        except TypeError:
            # pylint: disable=raise-missing-from
            # I decided to leave this out because the original exception is less helpful and spams the stack trace.
            # Please read: https://docs.python.org/3/library/exceptions.html#BaseException.__suppress_context__
            raise TypeError(
                f"Arguments do not match signature of callback {self.callback.__name__}{self.actual_signature_str}. "
                f"Callback function must match signature: {self.callback.__name__}{self.expected_signature_str}"
            ) from None
        return self.callback(*filled_signature.args, **filled_signature.kwargs)


class ErrorCallback(Callback[_P, _T]):
    """
    This class wraps an error callback. It is a subclass of Callback and adds the error parameter to the expected
    signature.
    """

    _CALLBACK_ERROR_PARAM = inspect.Parameter(
        "error", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=BaseException
    )

    @classmethod
    def from_callable(
        cls: type[_ErrorCallbackT],
        callback: Callable,
        signature_from_callable: Callable[..., Any] | inspect.Signature | None = None,
        add_params: Sequence[inspect.Parameter] | None = None,
        return_type: Any = UNSET,
    ) -> _ErrorCallbackT:
        if add_params is None:
            add_params = []
        inst = cast(
            _ErrorCallbackT,
            super().from_callable(
                callback, signature_from_callable, [cls._CALLBACK_ERROR_PARAM, *add_params], return_type
            ),
        )
        return inst


class SuccessCallback(Callback[_P, _T]):
    """
    This class wraps a success callback. It is a subclass of Callback and adds the result parameter to the expected
    signature. The annotation type is taken from the return annotation of the `signature_from_callable`.
    """

    @classmethod
    def from_callable(
        cls: type[_SuccessCallbackT],
        callback: Callable,
        signature_from_callable: Callable[..., Any] | inspect.Signature | None = None,
        add_params: Sequence[inspect.Parameter] | None = None,
        return_type: Any = UNSET,
    ) -> _SuccessCallbackT:
        inst = cast(_SuccessCallbackT, super().from_callable(callback, signature_from_callable, add_params))
        add_param = inspect.Parameter(
            "result", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=inst.expected_signature.return_annotation
        )
        if return_type is UNSET:
            return_type = inst.expected_signature.return_annotation
        inst.expected_signature = inst.expected_signature.replace(
            parameters=[add_param, *inst.expected_signature.parameters.values()],
            return_annotation=return_type,
        )
        return inst
