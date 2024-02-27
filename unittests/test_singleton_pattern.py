import pytest

from error_handler.types import Singleton


class TestSingleton:
    def test_singleton(self):
        class MySingleton(metaclass=Singleton):
            def __init__(self):
                self.x = 7

            def get_me(self) -> int:
                return self.x

        my_singleton1 = MySingleton()
        my_singleton2 = MySingleton()
        assert my_singleton1 is my_singleton2
        assert my_singleton1.get_me() == 7

    def test_singleton_with_args(self):
        with pytest.raises(AttributeError) as error_info:

            class _(metaclass=Singleton):
                def __init__(self, x: int):
                    self.x = x

                def get_me(self) -> int:
                    return self.x

        assert (
            str(error_info.value)
            == "__init__ method of _ cannot receive arguments. This is contrary to the singleton pattern."
        )
