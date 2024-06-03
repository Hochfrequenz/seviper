"""
This module handles stuff related to the optional dependency aiostream.
The features that require aiostream are disabled if the package is not installed.
"""

try:
    # pylint: disable=unused-import
    import aiostream

    IS_AIOSTREAM_INSTALLED = True
except ImportError:
    IS_AIOSTREAM_INSTALLED = False


# pylint: disable=too-few-public-methods
class _NotInstalled:
    """
    This class is used to replace the aiostream module when it is not installed.
    """

    def __getattr__(self, item):
        if item in ("map", "action"):
            raise_import_error()


def raise_import_error():
    """
    Raises an ImportError if aiostream is not installed but the dev attempts to use features that require it.
    """
    raise ImportError(
        "aiostream not found. This feature needs aiostream installed. "
        "Consider using `pip install seviper[aiostream]`."
    )
