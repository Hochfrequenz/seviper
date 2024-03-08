# Seviper - Error handling framework to catch 'em all

![Unittests status badge](https://github.com/Hochfrequenz/seviper/workflows/Unittests/badge.svg)
![Coverage status badge](https://github.com/Hochfrequenz/seviper/workflows/Coverage/badge.svg)
![Linting status badge](https://github.com/Hochfrequenz/seviper/workflows/Linting/badge.svg)
![Black status badge](https://github.com/Hochfrequenz/seviper/workflows/Formatting/badge.svg)

## Features
This framework provides several error handlers to catch errors and call callback functions to handle these errors
(or successes). It comes fully equipped with:

- A decorator to handle errors in functions or coroutines
- A decorator to retry a function or coroutine if it fails (can be useful for network requests)
- A context manager to handle errors in a block of code

Additionally, if you use `aiostream` (e.g. using `pip install seviper[aiostream]`), you can use the following features:

- The `stream.map` (or `pipe.map`, analogous to the `aiostream` functions) function to run the function, catch all
    exceptions and call the error handler if an exception occurs. Additionally, filters out all failed results.

## Installation

```bash
pip install seviper
```

or optionally:

```bash
pip install seviper[aiostream]
```

## Usage
Here is a more or less complex example as showcase of the features of this library:

```python
import asyncio
import logging
import sys
import aiostream
import error_handler

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
logger = logging.root
op = aiostream.stream.iterate(range(10))


def log_error(error: Exception, num: int):
    """Only log error and reraise it"""
    logger.error("double_only_odd_nums_except_5 failed for input %d. ", num)
    raise error


@error_handler.decorator(on_error=log_error)
async def double_only_odd_nums_except_5(num: int) -> int:
    if num % 2 == 0:
        raise ValueError(num)
    with error_handler.context_manager(on_success=lambda: logging.info("Success: %s", num)):
        if num == 5:
            raise RuntimeError("Another unexpected error. Number 5 will not be doubled.")
        num *= 2
    return num


def catch_value_errors(error: Exception, _: int):
    if not isinstance(error, ValueError):
        raise error


def log_success(result_num: int, provided_num: int):
    logger.info("Success: %d -> %d", provided_num, result_num)


op = op | error_handler.pipe.map(
    double_only_odd_nums_except_5,
    on_error=catch_value_errors,
    on_success=log_success,
    wrap_secured_function=True,
    suppress_recalling_on_error=False,
)

result = asyncio.run(aiostream.stream.list(op))

assert result == [2, 6, 5, 14, 18]
```

This outputs:

```
ERROR:root:double_only_odd_nums_except_5 failed for input 0.
INFO:root:Success: 2
INFO:root:Success: 1 -> 2
ERROR:root:double_only_odd_nums_except_5 failed for input 2.
INFO:root:Success: 6
INFO:root:Success: 3 -> 6
ERROR:root:double_only_odd_nums_except_5 failed for input 4.
INFO:root:Success: 5 -> 5
ERROR:root:double_only_odd_nums_except_5 failed for input 6.
INFO:root:Success: 14
INFO:root:Success: 7 -> 14
ERROR:root:double_only_odd_nums_except_5 failed for input 8.
INFO:root:Success: 18
INFO:root:Success: 9 -> 18
```

## How to use this Repository on Your Machine

Please refer to the respective section in our [Python template repository](https://github.com/Hochfrequenz/python_template_repository?tab=readme-ov-file#how-to-use-this-repository-on-your-machine)
to learn how to use this repository on your machine.

## Contribute

You are very welcome to contribute to this template repository by opening a pull request against the main branch.
