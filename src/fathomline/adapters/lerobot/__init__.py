"""The LeRobot adapter: the one place LeRobot is imported (D3).

Importing this package is cheap and works on any platform. `installed()` says
whether the `lerobot` extra is present at all (anywhere but the container, it is
not); `load_fixture` imports LeRobot when it is called.

It imports nothing from the core at run time, so the image's `warm.py` layer
depends on this package's files alone and survives changes elsewhere.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fathomline.core.protocols import Env, Policy

__all__ = ["installed", "load_fixture"]


def installed() -> bool:
    """Whether the `lerobot` extra is installed here.

    The only condition under which `eval` may say "not here". A module missing
    *inside* an installed stack is a broken image, and must surface as one.
    """
    return importlib.util.find_spec("lerobot") is not None


def load_fixture() -> tuple[Env, Policy, int]:
    """The D1 fixture: (env, policy, episode step limit). See `adapter.load_fixture`."""
    from fathomline.adapters.lerobot.adapter import load_fixture as _load_fixture

    return _load_fixture()
