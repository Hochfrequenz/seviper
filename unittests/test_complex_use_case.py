import asyncio
import logging

import aiostream

import error_handler


class TestComplexExample:
    def test_complex_use_case(self):
        op = aiostream.stream.iterate(range(10))
