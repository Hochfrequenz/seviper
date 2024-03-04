import pytest

import error_handler

from .utils import assert_not_called, create_callback_tracker


class TestCallbackErrors:
    def test_decorator_callbacks_wrong_signature_call_all_callbacks(self):
        on_success_callback, success_tracker = create_callback_tracker()

        def on_finalize_wrong_signature():
            pass

        @error_handler.decorator(
            on_success=on_success_callback, on_error=assert_not_called, on_finalize=on_finalize_wrong_signature
        )
        def func(hello: str) -> str:
            return f"Hello {hello}"

        with pytest.raises(BaseExceptionGroup) as error:
            func("World!")

        assert len(error.value.exceptions) == 1
        assert isinstance(error.value.exceptions[0], TypeError)
        assert "Arguments do not match signature of callback" in str(error.value.exceptions[0])
        assert "on_finalize_wrong_signature()" in str(error.value.exceptions[0])
        assert "on_finalize_wrong_signature(hello: str) -> Any" in str(error.value.exceptions[0])
        assert success_tracker == [(("Hello World!", "World!"), {})]

    def test_decorator_callbacks_wrong_signature_and_unexpected_error(self):

        def on_finalize_wrong_signature():
            pass

        def on_error_callback(_: BaseException, __: str):
            raise ValueError("This is a test error")

        @error_handler.decorator(
            on_success=assert_not_called, on_error=on_error_callback, on_finalize=on_finalize_wrong_signature
        )
        def func(hello: str) -> str:
            raise ValueError(f"This is a test error {hello}")

        with pytest.raises(BaseExceptionGroup) as error:
            func("World!")

        assert len(error.value.exceptions) == 2
        if isinstance(error.value.exceptions[0], ValueError):
            value_error, wrong_signature_error = error.value.exceptions
        else:
            wrong_signature_error, value_error = error.value.exceptions

        assert isinstance(wrong_signature_error, TypeError)
        assert "Arguments do not match signature of callback" in str(wrong_signature_error)
        assert "on_finalize_wrong_signature()" in str(wrong_signature_error)
        assert "on_finalize_wrong_signature(hello: str) -> Any" in str(wrong_signature_error)
        assert isinstance(value_error, ValueError)
        assert str(value_error) == "This is a test error"
        assert "This is a test error World!" in str(error.value.__context__)
