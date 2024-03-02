"""
This module contains the Callback class, which is used to wrap a callable and its expected signature.
"""

import inspect
from typing import Any, Callable, Generic, ParamSpec, Sequence, TypeVar

from .types import UNSET

_P = ParamSpec("_P")
_T = TypeVar("_T")
_CallbackT = TypeVar("_CallbackT", bound=Callable)


class Callback(Generic[_P, _T]):
    """
    This class wraps a callable and its expected signature.
    """

    def __init__(self, callback: Callable[_P, _T], expected_signature: inspect.Signature):
        self.callback = callback
        self.expected_signature = expected_signature
        self._actual_signature: inspect.Signature | None = None
        self._args_to_inject: dict[int, Any] | None = None
        self._kwargs_to_inject: dict[str, Any] | None = None

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

    def inject_parameters(self, *args: tuple[int, Any], **kwargs):
        """
        Partially bind the given arguments and keyword arguments to the expected signature.
        This will not raise an error if the arguments do not match the signature yet.
        It also allows to inject positional arguments at specific indices which is not supported by
        `inspect.bind_partial`.
        """
        if self._args_to_inject is None:
            self._args_to_inject = {}
        if self._kwargs_to_inject is None:
            self._kwargs_to_inject = {}
        self._args_to_inject.update(dict(args))
        self._kwargs_to_inject.update(kwargs)

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        """
        Call the callback with the given arguments and keyword arguments. The arguments will be checked against the
        expected signature. If the callback does not match the expected signature, a TypeError explaining which
        signature was expected will be raised.
        """
        if self._args_to_inject is not None:
            args = list(args)
            for index, value in self._args_to_inject.items():
                args.insert(index, value)
        if self._kwargs_to_inject is not None:
            kwargs.update(self._kwargs_to_inject)
        try:
            filled_signature = self.actual_signature.bind(*args, **kwargs)
        except TypeError:
            # pylint: disable=raise-missing-from
            # I decided to leave this out because the original exception is less helpful and spams the stack trace.
            # Please read: https://docs.python.org/3/library/exceptions.html#BaseException.__suppress_context__
            raise TypeError(
                f"Arguments do not match signature of callback '{self.callback.__name__}'. "
                f"Callback function must match signature: {self.callback.__name__}{self.expected_signature_str}"
            ) from None
        return self.callback(*filled_signature.args, **filled_signature.kwargs)


class ErrorCallback(Callback[_P, _T]):
    _CALLBACK_ERROR_PARAM = inspect.Parameter("error", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Exception)

    @classmethod
    def from_callable(
        cls: type["_CallbackT"],
        callback: Callable,
        signature_from_callable: Callable[..., Any] | inspect.Signature | None = None,
        add_params: Sequence[inspect.Parameter] | None = None,
        return_type: Any = UNSET,
    ) -> "_CallbackT":
        if add_params is None:
            add_params = []
        inst = super().from_callable(
            callback, signature_from_callable, [cls._CALLBACK_ERROR_PARAM, *add_params], return_type
        )
        return inst


class SuccessCallback(Callback[_P, _T]):
    @classmethod
    def from_callable(
        cls: type["_CallbackT"],
        callback: Callable,
        signature_from_callable: Callable[..., Any] | inspect.Signature | None = None,
        add_params: Sequence[inspect.Parameter] | None = None,
        return_type: Any = UNSET,
    ) -> "_CallbackT":
        inst = super().from_callable(callback, signature_from_callable, add_params)
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
