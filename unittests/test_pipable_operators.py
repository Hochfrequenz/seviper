import pytest
from aiostream import stream

import error_handler


class TestErrorHandlerPipableOperators:
    def test_aiostream_import_error(self, trigger_aiostream_import_error):
        with pytest.raises(ImportError) as error:
            _ = error_handler.stream.map

        assert "aiostream not found" in str(error.value)

        with pytest.raises(ImportError) as error:
            _ = error_handler.pipe.map

        assert "aiostream not found" in str(error.value)

    def test_aiostream_import_error_import_from_init(self, trigger_aiostream_import_error):
        # pylint: disable=import-outside-toplevel, reimported
        from error_handler import pipe, stream

        with pytest.raises(ImportError) as error:
            _ = stream.map

        assert "aiostream not found" in str(error.value)

        with pytest.raises(ImportError) as error:
            _ = pipe.map

        assert "aiostream not found" in str(error.value)

    def test_aiostream_import_error_import_from_submodule(self, trigger_aiostream_import_error):
        # pylint: disable=import-outside-toplevel, redefined-builtin, unused-import
        with pytest.raises(ImportError) as error:
            from error_handler.stream import map

        assert "aiostream not found" in str(error.value)

        with pytest.raises(ImportError) as error:
            from error_handler.pipe import map

        assert "aiostream not found" in str(error.value)

    async def test_secured_map_stream(self):
        errored_nums: set[int] = set()
        op = stream.iterate([1, 2, 3, 4, 5, 6])

        def raise_for_even(num: int) -> int:
            if num % 2 == 0:
                raise ValueError(f"{num}")
            return num

        def store(error: Exception):
            nonlocal errored_nums
            errored_nums.add(int(str(error)))

        op = error_handler.stream.map(op, raise_for_even, on_error=store)

        elements = await stream.list(op)
        assert set(elements) == {1, 3, 5}
        assert errored_nums == {2, 4, 6}

    async def test_secured_map_stream_double_secure_invalid_return_value(self):
        op = stream.iterate([1, 2, 3, 4, 5, 6])

        @error_handler.decorator(on_error_return_always=0)
        def raise_for_even(_: int) -> int:
            return 1

        with pytest.raises(ValueError) as error:
            _ = error_handler.stream.map(op, raise_for_even)

        assert "The given function is already secured but does not return ERRORED in error case" in str(error.value)

    async def test_secured_map_stream_double_secure_invalid_arguments(self):
        op = stream.iterate([1, 2, 3, 4, 5, 6])

        @error_handler.decorator()
        def raise_for_even(_: int) -> int:
            return 1

        with pytest.raises(ValueError) as error:
            _ = error_handler.stream.map(op, raise_for_even, on_error=lambda _: None)

        assert "Please do not set on_success, on_error, on_finalize as they would be ignored" in str(error.value)

    async def test_secured_map_stream_double_secure_no_wrap(self):
        errored_nums: set[int] = set()
        op = stream.iterate([1, 2, 3, 4, 5, 6])

        def store(error: Exception):
            nonlocal errored_nums
            errored_nums.add(int(str(error)))

        @error_handler.decorator(on_error=store)
        def raise_for_even(num: int) -> int:
            if num % 2 == 0:
                raise ValueError(f"{num}")
            return num

        op = error_handler.stream.map(op, raise_for_even)

        elements = await stream.list(op)
        assert set(elements) == {1, 3, 5}
        assert errored_nums == {2, 4, 6}

    async def test_secured_map_stream_double_secure_wrap(self):
        errored_nums_from_map: set[int] = set()
        errored_nums_from_decorator: set[int] = set()
        op = stream.iterate([1, 2, 3, 4, 5, 6])

        def store_from_map(error: Exception):
            nonlocal errored_nums_from_map
            errored_nums_from_map.add(int(str(error)))

        def store_from_decorator(error: Exception):
            nonlocal errored_nums_from_decorator
            errored_nums_from_decorator.add(int(str(error)))
            raise error

        @error_handler.decorator(on_error=store_from_decorator)
        def raise_for_even(num: int) -> int:
            if num % 2 == 0:
                raise ValueError(f"{num}")
            return num

        op = error_handler.stream.map(
            op, raise_for_even, on_error=store_from_map, wrap_secured_function=True, suppress_recalling_on_error=False
        )

        elements = await stream.list(op)
        assert set(elements) == {1, 3, 5}
        assert errored_nums_from_map == errored_nums_from_decorator == {2, 4, 6}

    async def test_secured_map_pipe(self):
        errored_nums: set[int] = set()
        op = stream.iterate([1, 2, 3, 4, 5, 6])

        def raise_for_even(num: int) -> int:
            if num % 2 == 0:
                raise ValueError(f"{num}")
            return num

        def store(error: Exception):
            nonlocal errored_nums
            errored_nums.add(int(str(error)))

        op = op | error_handler.pipe.map(raise_for_even, on_error=store)

        elements = await stream.list(op)
        assert set(elements) == {1, 3, 5}
        assert errored_nums == {2, 4, 6}
