"""
BINDER package initialization.
"""

import logging
from pathlib import Path

# -----------------------------------------------------------------------------
# Package information
# -----------------------------------------------------------------------------

BINDERDIR = Path(__file__).parent.absolute()

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Version
# -----------------------------------------------------------------------------

try:
    from ._version import __version__
except ImportError:
    # Useful when running directly from an unbuilt source tree
    __version__ = "unknown"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

from .utils import *


# -----------------------------------------------------------------------------
# Native registration backend
#
# CMake decides which backend to build:
#
#   CUDA available  -> BINDER._core contains GPU implementation
#   CUDA unavailable -> BINDER._core contains CPU implementation
#   BINDER_FORCE_CPU=ON -> BINDER._core contains CPU implementation
#
# There is therefore only ONE extension to import.
# -----------------------------------------------------------------------------

try:
    from . import _core

    # Registration class exposed by pybind11
    Registration = _core

    # "cuda" or "cpu", exposed by bindings.cpp
    _backend = getattr(_core, "backend", "unknown")

except ImportError as exc:
    Registration = None
    _backend = None

    logger.warning(
        "Could not import the compiled BINDER extension '_core': %s",
        exc,
    )


# -----------------------------------------------------------------------------
# High-level Python interface
# -----------------------------------------------------------------------------

if Registration is not None:
    try:
        from .FastRegistration import FastRegistration
    except ImportError as exc:
        logger.warning(
            "Could not import FastRegistration: %s",
            exc,
        )
        FastRegistration = None
else:
    FastRegistration = None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

__all__ = [
    "FastRegistration",
    "Registration",
    "_backend",
    "__version__",
]
