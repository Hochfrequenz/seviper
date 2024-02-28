import asyncio
import logging

import aiostream

import error_handler


class TestComplexExample:
    async def test_complex_use_case(self):
        op = aiostream.stream.iterate(range(10))

        def log_error(error: Exception):
            """Only log error and reraise it"""
            logging.error(error)
            raise error

        @error_handler.decorator(on_error=log_error)
        async def double_only_odd_nums_except_5(num: int) -> int:
            if num % 2 == 0:
                raise ValueError(num)
            with error_handler.context_manager(on_success=lambda: logging.info(f"Success: {num}")):
                if num == 5:
                    raise RuntimeError("Another unexpected error. Number 5 will not be doubled.")
                num *= 2
            return num

        def catch_value_errors(error: Exception):
            if not isinstance(error, ValueError):
                raise error

        def log_success(num: int):
            logging.info(f"Success: {num}")

        op = op | error_handler.pipe.map(
            double_only_odd_nums_except_5,
            on_error=catch_value_errors,
            on_success=log_success,
            wrap_secured_function=True,
            suppress_recalling_on_error=False,
        )

        result = await aiostream.stream.list(op)

        assert result == [2, 6, 5, 14, 18]
