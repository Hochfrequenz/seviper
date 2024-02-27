import pytest

import error_handler


class TestErrorHandlerContextManager:
    def test_context_manager_error_case(self):
        catched_error: Exception | None = None

        def store_error(error: Exception):
            nonlocal catched_error
            catched_error = error

        with error_handler.context_manager(
            on_error=store_error,
        ):
            raise ValueError("This is a test error world")

        assert isinstance(catched_error, ValueError)
        assert str(catched_error) == "This is a test error world"

    def test_context_manager_success_case(self):
        succeeded = False

        def succeeded_callback():
            nonlocal succeeded
            succeeded = True

        with error_handler.context_manager(on_success=succeeded_callback):
            pass

        assert succeeded

    def test_context_manager_reraise(self):
        catched_error: Exception | None = None

        def store_error(error: Exception):
            nonlocal catched_error
            catched_error = error
            raise error

        with pytest.raises(ValueError) as error:
            with error_handler.context_manager(
                on_error=store_error,
            ):
                raise ValueError("This is a test error world")

        assert isinstance(catched_error, ValueError)
        assert error.value is catched_error
        assert str(catched_error) == "This is a test error world"
