"""
This module contains the pipe operator for the map function. It mirrors the behaviour of aiostream.pipe module.
"""

import sys

from ._extra import IS_AIOSTREAM_INSTALLED

if IS_AIOSTREAM_INSTALLED:
    from .stream import map as _map

    # pylint: disable=redefined-builtin
    map = _map.pipe
else:
    from ._extra import _NotInstalled

    sys.modules[__name__] = _NotInstalled()  # type: ignore[assignment]
