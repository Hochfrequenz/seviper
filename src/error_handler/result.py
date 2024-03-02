from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, TypeAlias

from error_handler.types import UNSET, ErroredType, T, _UnsetType


class CallbackResultType(StrEnum):
    """
    Represents the result of a callback.
    """

    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class CallbackResultTypes:
    success: CallbackResultType = CallbackResultType.SKIPPED
    error: CallbackResultType = CallbackResultType.SKIPPED
    finalize: CallbackResultType = CallbackResultType.SKIPPED


@dataclass(frozen=True)
class ReturnValues:
    success: Any = UNSET
    error: Any = UNSET
    finalize: Any = UNSET


@dataclass(frozen=True)
class CallbackSummary:
    """
    Represents a result of a function call.
    """

    callback_result_types: CallbackResultTypes
    callback_return_values: ReturnValues


@dataclass(frozen=True)
class PositiveResult(Generic[T]):
    """
    Represents a successful result.
    """

    result: T | _UnsetType


@dataclass(frozen=True)
class NegativeResult(Generic[T]):
    """
    Represents an errored result.
    """

    result: T | ErroredType
    error: BaseException


ResultType: TypeAlias = PositiveResult[T] | NegativeResult[T]
