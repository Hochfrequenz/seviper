"""
This module defines the types used in the error_handler module.
"""

import inspect
from typing import TYPE_CHECKING, Any, Callable, Coroutine, ParamSpec, Protocol, TypeAlias, TypeGuard, TypeVar

if TYPE_CHECKING:
    from .core import Catcher
    from .result import ResultType

T = TypeVar("T")
P = ParamSpec("P")


class SingletonMeta(type):
    """
    A metaclass implementing the singleton pattern.
    """

    def __new__(mcs, name, bases, attrs):
        if "__init__" in attrs:
            mcs.check_init(attrs["__init__"], name)
        attrs["__singleton_instance__"] = None
        attrs["__new__"] = mcs.get_singleton_new(attrs.get("__new__", object.__new__))
        return super().__new__(mcs, name, bases, attrs)

    @staticmethod
    def check_init(init_method, cls_name):
        """
        Raises an error if the __init__ method of the class receives arguments.
        It is contrary to the singleton pattern. If you really need to do this, you should instead overwrite the __new__
        method.
        """
        signature = inspect.signature(init_method)
        if len(signature.parameters) > 1:
            raise AttributeError(
                f"__init__ method of {cls_name} cannot receive arguments. This is contrary to the singleton pattern."
            )

    @staticmethod
    def get_singleton_new(old_new):
        """
        Returns a new __new__ method for the class that uses the Singleton metaclass.
        """

        def __singleton_new__(cls, *args, **kwargs):
            if cls.__singleton_instance__ is None:
                cls.__singleton_instance__ = old_new(cls, *args, **kwargs)
            return cls.__singleton_instance__

        return __singleton_new__


# pylint: disable=too-few-public-methods
class UnsetType(metaclass=SingletonMeta):
    """
    This type is meant to be used as singleton. Do not instantiate it on your own.
    The instance below represents an unset value. It is needed as default value since the respective
    parameters can be of any type (including None).
    """


UNSET = UnsetType()
"""
Represents an unset value. It is used as default value for parameters that can be of any type.
"""


FunctionType: TypeAlias = Callable[P, T]
AsyncFunctionType: TypeAlias = Callable[P, Coroutine[Any, Any, T]]


class SecuredFunctionType(Protocol[P, T]):
    """
    This type represents a secured function.
    """

    __catcher__: "Catcher[T]"
    __original_callable__: FunctionType[P, T]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> "ResultType[T]": ...


class SecuredAsyncFunctionType(Protocol[P, T]):
    """
    This type represents a secured async function.
    """

    __catcher__: "Catcher[T]"
    __original_callable__: AsyncFunctionType[P, T]

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> "ResultType[T]": ...


def is_secured(
    func: FunctionType[P, T] | SecuredFunctionType[P, T] | AsyncFunctionType[P, T] | SecuredAsyncFunctionType[P, T]
) -> TypeGuard[SecuredFunctionType[P, T] | SecuredAsyncFunctionType[P, T]]:
    """
    Returns True if the given function is secured and returns a ResultType[T]. False otherwise.
    """
    return hasattr(func, "__catcher__") and hasattr(func, "__original_callable__")
