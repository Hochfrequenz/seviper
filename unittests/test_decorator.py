from typing import Any, Callable

import pytest

import error_handler


def assert_not_called(*args, **kwargs):
    raise ValueError("This should not be called")


def create_callback_tracker(
    additional_callback: Callable = lambda *args: None,
) -> tuple[Callable, list[tuple[Any, ...]]]:
    """
    Creates a callback function taking any arguments and a tracker which stores all calls to the callback in a list.
    The callback function will call the additional_callback with the same arguments and return its return value.
    It will also store the arguments as tuple in the call_args list.
    Returns the callback and the call_args list.
    """
    call_args = []

    def callback(*args):
        call_args.append(args)
        return additional_callback(*args)

    return callback, call_args


def retry_stepping_func(_: int) -> float:
    """To speed things up a bit"""
    return 0.001


class TestErrorHandlerDecorator:
    async def test_decorator_coroutine_error_case(self):
        error_callback, error_tracker = create_callback_tracker()
        finalize_callback, finalize_tracker = create_callback_tracker()

        @error_handler.decorator(
            on_error=error_callback,
            on_finalize=finalize_callback,
            on_success=assert_not_called,
            on_error_return_always=error_handler.ERRORED,
        )
        async def async_function(hello: str) -> None:
            raise ValueError(f"This is a test error {hello}")

        awaitable = async_function("world")
        result = await awaitable
        assert str(error_tracker[0][0]) == "This is a test error world"
        assert result == error_handler.ERRORED
        assert finalize_tracker == [()]

    async def test_decorator_coroutine_success_case(self):
        on_success_callback, success_tracker = create_callback_tracker()
        on_finalize_callback, finalize_tracker = create_callback_tracker()

        @error_handler.decorator(
            on_success=on_success_callback,
            on_finalize=on_finalize_callback,
            on_error=assert_not_called,
        )
        async def async_function(hello: str) -> str:
            return f"Hello {hello}"

        awaitable = async_function("World!")
        result = await awaitable
        assert result == success_tracker[0][0] == "Hello World!"
        assert finalize_tracker == [()]

    def test_decorator_function_error_case(self):
        catched_error: Exception | None = None

        def store_error(error: Exception):
            nonlocal catched_error
            catched_error = error

        @error_handler.decorator(on_error=store_error)
        def func(hello: str) -> None:
            raise ValueError(f"This is a test error {hello}")

        result = func("world")
        assert isinstance(catched_error, ValueError)
        assert str(catched_error) == "This is a test error world"
        assert result == error_handler.ERRORED

    def test_decorator_function_success_case(self):
        return_value: str | None = None

        def store_return_value(value: str):
            nonlocal return_value
            return_value = value

        @error_handler.decorator(
            on_success=store_return_value,
        )
        def async_function(hello: str) -> str:
            return f"Hello {hello}"

        result = async_function("World!")
        assert result == return_value == "Hello World!"

    async def test_decorator_reraise_coroutine(self):
        catched_error: Exception | None = None

        def store_error(error: Exception):
            nonlocal catched_error
            catched_error = error
            raise error

        @error_handler.decorator(on_error=store_error)
        async def async_function(hello: str) -> None:
            raise ValueError(f"This is a test error {hello}")

        awaitable = async_function("world")
        with pytest.raises(ValueError) as error:
            _ = await awaitable

        assert isinstance(catched_error, ValueError)
        assert catched_error is error.value
        assert str(catched_error) == "This is a test error world"

    def test_decorator_reraise_function(self):
        catched_error: Exception | None = None

        def store_error(error: Exception):
            nonlocal catched_error
            catched_error = error
            raise error

        @error_handler.decorator(on_error=store_error)
        def func(hello: str) -> None:
            raise ValueError(f"This is a test error {hello}")

        with pytest.raises(ValueError) as error:
            _ = func("world")

        assert isinstance(catched_error, ValueError)
        assert catched_error is error.value
        assert str(catched_error) == "This is a test error world"

    async def test_retry_coroutine_return_after_retries(self):
        retry_counter = 0

        def error_callback(_: Exception, retry_count: int) -> bool:
            nonlocal retry_counter
            assert retry_count == retry_counter
            retry_counter += 1
            return True

        error_callback, error_tracker = create_callback_tracker(additional_callback=error_callback)
        success_callback, success_tracker = create_callback_tracker()
        finalize_callback, finalize_tracker = create_callback_tracker()

        @error_handler.retry_on_error(
            on_error=error_callback,
            on_success=success_callback,
            on_finalize=finalize_callback,
            on_fail=assert_not_called,
            retry_stepping_func=retry_stepping_func,
        )
        async def async_function(hello: str) -> str:
            nonlocal retry_counter
            if retry_counter < 2:
                raise ValueError(retry_counter)
            return f"Hello {hello}"

        awaitable = async_function("world")
        result = await awaitable

        assert retry_counter == 2
        assert [error.args[0] for error, _ in error_tracker] == [0, 1]
        assert result == "Hello world"
        assert success_tracker == [("Hello world", 2)]
        assert finalize_tracker == [(2,)]

    async def test_retry_coroutine_return_without_retries(self):
        @error_handler.retry_on_error(on_error=assert_not_called, retry_stepping_func=retry_stepping_func)
        async def async_function(hello: str) -> str:
            return f"Hello {hello}"

        awaitable = async_function("world")
        result = await awaitable

        assert result == "Hello world"

    async def test_retry_coroutine_fail_too_many_retries(self):
        retry_counter = 0

        def error_callback(_: Exception, retry_count: int) -> bool:
            nonlocal retry_counter
            assert retry_count == retry_counter
            retry_counter += 1
            return True

        error_callback, error_tracker = create_callback_tracker(additional_callback=error_callback)
        fail_callback, fail_tracker = create_callback_tracker()
        finalize_callback, finalize_tracker = create_callback_tracker()

        @error_handler.retry_on_error(
            on_error=error_callback,
            on_success=assert_not_called,
            on_finalize=finalize_callback,
            on_fail=fail_callback,
            max_retries=2,
            retry_stepping_func=retry_stepping_func,
        )
        async def async_function(_: str) -> str:
            raise ValueError(retry_counter)

        awaitable = async_function("world")
        with pytest.raises(RuntimeError) as error:
            _ = await awaitable

        assert str(error.value) == "Too many retries (2) for async_function"
        assert [error.args[0] for error, _ in error_tracker] == [0, 1]
        assert fail_tracker == [(error.value, 2)]
        assert finalize_tracker == [(2,)]

    async def test_retry_coroutine_fail_callback_returns_false(self):
        retry_counter = 0

        def error_callback(_: Exception, retry_count: int) -> bool:
            nonlocal retry_counter
            assert retry_count == retry_counter
            if retry_counter >= 2:
                return False
            retry_counter += 1
            return True

        error_callback, error_tracker = create_callback_tracker(additional_callback=error_callback)
        fail_callback, fail_tracker = create_callback_tracker()
        finalize_callback, finalize_tracker = create_callback_tracker()

        @error_handler.retry_on_error(
            on_error=error_callback,
            on_success=assert_not_called,
            on_finalize=finalize_callback,
            on_fail=fail_callback,
            retry_stepping_func=retry_stepping_func,
        )
        async def async_function(_: str) -> str:
            raise ValueError(retry_counter)

        awaitable = async_function("world")
        with pytest.raises(ValueError) as error:
            _ = await awaitable

        assert error.value.args[0] == 2
        assert [error.args[0] for error, _ in error_tracker] == [0, 1, 2]
        assert fail_tracker == [(error.value, 2)]
        assert finalize_tracker == [(2,)]

    def test_retry_function_return_after_retries(self):
        retry_counter = 0

        def error_callback(_: Exception, retry_count: int) -> bool:
            nonlocal retry_counter
            assert retry_count == retry_counter
            retry_counter += 1
            return True

        error_callback, error_tracker = create_callback_tracker(additional_callback=error_callback)
        success_callback, success_tracker = create_callback_tracker()
        finalize_callback, finalize_tracker = create_callback_tracker()

        @error_handler.retry_on_error(
            on_error=error_callback,
            on_success=success_callback,
            on_finalize=finalize_callback,
            on_fail=assert_not_called,
            retry_stepping_func=retry_stepping_func,
        )
        def sync_function(hello: str) -> str:
            nonlocal retry_counter
            if retry_counter < 2:
                raise ValueError(retry_counter)
            return f"Hello {hello}"

        result = sync_function("world")

        assert retry_counter == 2
        assert [error.args[0] for error, _ in error_tracker] == [0, 1]
        assert result == "Hello world"
        assert success_tracker == [("Hello world", 2)]
        assert finalize_tracker == [(2,)]
