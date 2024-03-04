import asyncio
from math import floor

import error_handler

from .utils import assert_not_called, create_callback_tracker


class TestConcurrency:
    async def test_concurrency_decorator_callback_order(self):
        counter = 0
        counter_list = []

        def log_counter(_: str, __: str):
            nonlocal counter
            counter_list.append(counter)

        on_success_callback, success_tracker = create_callback_tracker(log_counter)
        on_finalize_callback, finalize_tracker = create_callback_tracker()

        @error_handler.decorator(
            on_success=on_success_callback,
            on_finalize=on_finalize_callback,
            on_error=assert_not_called,
        )
        async def async_function(hello: str) -> str:
            nonlocal counter
            counter += 1
            await asyncio.sleep(0.1)
            return f"Hello {hello}"

        results = await asyncio.gather(async_function("World!"), async_function("world..."))
        assert set(results) == {"Hello World!", "Hello world..."}
        assert success_tracker == [(("Hello World!", "World!"), {}), (("Hello world...", "world..."), {})]
        assert finalize_tracker == [(("World!",), {}), (("world...",), {})]
        assert counter_list == [2, 2]

    async def test_concurrency_retry(self):
        retry_counter = 0

        def error_callback(_: Exception, retry_count: int, __: str) -> bool:
            nonlocal retry_counter
            assert retry_count == floor(retry_counter / 2)
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
            retry_stepping_func=lambda _: 0.01,
        )
        async def async_function(hello: str) -> str:
            nonlocal retry_counter
            if hello == "world...":
                # "guarantee" that the first raise is the one for "World!"
                await asyncio.sleep(0.005)
            await asyncio.sleep(0.02)
            if retry_counter < 4:
                raise ValueError(retry_counter)
            return f"Hello {hello}"

        results = await asyncio.gather(async_function("World!"), async_function("world..."))
        assert set(results) == {"Hello World!", "Hello world..."}
        assert len(error_tracker) == 4
        error_tracker_args = [(error.args[0], num, hello) for (error, num, hello), _ in error_tracker]
        assert set(error_tracker_args[:2]) == {(0, 0, "World!"), (1, 0, "world...")}
        assert set(error_tracker_args[2:]) == {(2, 1, "World!"), (3, 1, "world...")}
        assert success_tracker == [(("Hello World!", 2, "World!"), {}), (("Hello world...", 2, "world..."), {})]
        assert finalize_tracker == [((2, "World!"), {}), ((2, "world..."), {})]
