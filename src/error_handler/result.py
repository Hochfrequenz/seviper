"""
This module contains classes to encapsulate some information about the result of a secured context and its callbacks.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, TypeAlias

from error_handler.types import UNSET, ErroredType, T, UnsetType


class CallbackResultType(StrEnum):
    """
    Determines whether a callback was successful or errored or not called aka skipped.
    """

    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class CallbackResultTypes:
    """
    Contains the information for each callback whether a callback was successful or errored or not called aka skipped.
    """

    success: CallbackResultType = CallbackResultType.SKIPPED
    error: CallbackResultType = CallbackResultType.SKIPPED
    finalize: CallbackResultType = CallbackResultType.SKIPPED


@dataclass(frozen=True)
class ReturnValues:
    """
    Contains the return values of each callback.
    If a callback was not called, the value is UNSET.
    If a callback errored, the value is the error.
    """

    success: Any = UNSET
    error: Any = UNSET
    finalize: Any = UNSET


@dataclass(frozen=True)
class CallbackSummary:
    """
    Contains the information of the result of a secured context and its callbacks.
    """

    callback_result_types: CallbackResultTypes
    callback_return_values: ReturnValues


@dataclass(frozen=True)
class PositiveResult(Generic[T]):
    """
    Represents a successful result.
    """

    result: T | UnsetType


@dataclass(frozen=True)
class NegativeResult(Generic[T]):
    """
    Represents an errored result.
    """

    result: T | ErroredType
    error: BaseException


ResultType: TypeAlias = PositiveResult[T] | NegativeResult[T]
