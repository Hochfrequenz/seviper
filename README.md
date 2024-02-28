# Seviper - Error handling framework to catch 'em all

![Unittests status badge](https://github.com/Hochfrequenz/seviper/workflows/Unittests/badge.svg)
![Coverage status badge](https://github.com/Hochfrequenz/seviper/workflows/Coverage/badge.svg)
![Linting status badge](https://github.com/Hochfrequenz/seviper/workflows/Linting/badge.svg)
![Black status badge](https://github.com/Hochfrequenz/seviper/workflows/Formatting/badge.svg)

## Features
This framework provides several error handlers to catch errors and call callback functions to handle these errors
(or successes). It comes fully equipped with:

- A decorator to handle errors in functions or coroutines
- A decorator to retry a function or coroutine if it fails
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
Here is a complex example as showcase of the features of this library:

```python
import asyncio
import aiostream
import error_handler
import logging

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

result = asyncio.run(aiostream.stream.list(op))

assert result == [2, 6, 5, 14, 18]
```

## How to use this Repository on Your Machine

Please refer to the respective section in our [Python template repository](https://github.com/Hochfrequenz/python_template_repository?tab=readme-ov-file#how-to-use-this-repository-on-your-machine)
to learn how to use this repository on your machine.

## Contribute

You are very welcome to contribute to this template repository by opening a pull request against the main branch.
