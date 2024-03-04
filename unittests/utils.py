from typing import Any, Callable


def assert_not_called(*_, **__):
    raise ValueError("This should not be called")


def create_callback_tracker(
    additional_callback: Callable = lambda *args, **kwargs: None,
) -> tuple[Callable, list[tuple[tuple[Any, ...], dict[str, Any]]]]:
    """
    Creates a callback function taking any arguments and a tracker which stores all calls to the callback in a list.
    The callback function will call the additional_callback with the same arguments and return its return value.
    It will also store the arguments as tuple in the call_args list.
    Returns the callback and the call_args list.
    """
    call_args = []

    def callback(*args, **kwargs):
        call_args.append((args, kwargs))
        return additional_callback(*args, **kwargs)

    return callback, call_args
