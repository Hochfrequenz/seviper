import inspect
from typing import Any, Callable, Generic, ParamSpec, Sequence, TypeVar

from .types import UNSET

_P = ParamSpec("_P")
_T = TypeVar("_T")


class Callback(Generic[_P, _T]):
    def __init__(self, callback: Callable[_P, _T], expected_signature: inspect.Signature):
        self.callback = callback
        self.expected_signature = expected_signature
        self._actual_signature: inspect.Signature | None = None

    @property
    def actual_signature(self) -> inspect.Signature:
        if self._actual_signature is None:
            self._actual_signature = inspect.signature(self.callback)
        return self._actual_signature

    @property
    def expected_signature_str(self) -> str:
        return str(self.expected_signature)

    @property
    def actual_signature_str(self) -> str:
        return str(self.actual_signature)

    @classmethod
    def from_callable(
        cls,
        callback: Callable,
        signature_from_callable: Callable[..., Any] | None = None,
        add_params: Sequence[inspect.Parameter] = None,
        return_type: Any = UNSET,
    ) -> "Callback":
        if signature_from_callable is None:
            sig = inspect.Signature()
        else:
            sig = inspect.signature(signature_from_callable)
        if add_params is not None or return_type is not None:
            params = list(sig.parameters.values())
            if add_params is not None:
                params = [add_params, *params]
            if return_type is not UNSET:
                return_type = sig.return_annotation
            sig = sig.replace(parameters=params, return_annotation=return_type)
        return cls(callback, sig)

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            filled_signature = self.actual_signature.bind(*args, **kwargs)
        except TypeError:
            raise TypeError(
                f"Arguments do not match signature of callback '{self.callback.__name__}'. "
                f"Callback function must match signature: {self.callback.__name__}{self.expected_signature_str}"
            )
        return self.callback(*filled_signature.args, **filled_signature.kwargs)
