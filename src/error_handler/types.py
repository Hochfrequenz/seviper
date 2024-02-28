"""
This module defines the types used in the error_handler module.
"""

import inspect
from abc import ABC
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, ParamSpec, TypeAlias, TypeVar

T = TypeVar("T")
P = ParamSpec("P")


class Singleton(type):
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
class ErroredType(metaclass=Singleton):
    """
    This type is meant to be used as singleton. Do not instantiate it on your own.
    The instance below represents an errored result.
    """


# pylint: disable=too-few-public-methods
class UnsetType(metaclass=Singleton):
    """
    This type is meant to be used as singleton. Do not instantiate it on your own.
    The instance below represents an unset value. It is needed as default value since the respective
    parameters can be of any type (including None).
    """


@dataclass
class Result(Generic[T], ABC):
    """
    Represents a result of a function call.
    """


@dataclass
class PositiveResult(Result[T]):
    """
    Represents a successful result.
    """

    result: T


@dataclass
class NegativeResult(Result[T]):
    """
    Represents an errored result.
    """

    result: T | ErroredType
    error: Exception


UNSET = UnsetType()
"""
Represents an unset value. It is used as default value for parameters that can be of any type.
"""
ERRORED = ErroredType()
"""
Represents an errored result. It is used to be able to return something in error cases. See Catcher.secure_call
for more information.
"""

FunctionType: TypeAlias = Callable[P, T]
AsyncFunctionType: TypeAlias = Callable[P, Awaitable[T]]
SecuredFunctionType: TypeAlias = Callable[P, T | ErroredType]
SecuredAsyncFunctionType: TypeAlias = Callable[P, Awaitable[T | ErroredType]]
ResultType: TypeAlias = PositiveResult[T] | NegativeResult[T]
