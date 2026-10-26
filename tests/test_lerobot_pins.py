"""The LeRobot fixture's pins and the digest that checks them.

`pins` is standard-library only, so this runs on any platform, although the
adapter it serves runs only in the container.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

from fathomline.adapters import lerobot
from fathomline.adapters.lerobot import pins

SHA1 = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")


def _tree(root: Path, files: dict[str, bytes]) -> Path:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


@pytest.mark.parametrize("pin", pins.PINS, ids=lambda pin: pin.repo_id)
def test_every_pin_names_a_full_revision_sha_and_a_content_digest(pin: pins.HubPin) -> None:
    """A branch name or a short SHA is not a pin; neither is a missing digest."""
    assert SHA1.fullmatch(pin.revision)
    assert SHA256.fullmatch(pin.digest)


def test_the_digest_covers_content_and_paths(tmp_path: Path) -> None:
    base = pins.tree_digest(_tree(tmp_path / "a", {"x.xml": b"1", "sub/y.obj": b"2"}))
    same = pins.tree_digest(_tree(tmp_path / "b", {"sub/y.obj": b"2", "x.xml": b"1"}))
    edited = pins.tree_digest(_tree(tmp_path / "c", {"x.xml": b"1", "sub/y.obj": b"3"}))
    moved = pins.tree_digest(_tree(tmp_path / "d", {"x.xml": b"1", "y.obj": b"2"}))
    assert base == same
    assert base not in {edited, moved}


def test_the_digest_ignores_the_hub_download_metadata(tmp_path: Path) -> None:
    clean = pins.tree_digest(_tree(tmp_path / "a", {"x.xml": b"1"}))
    with_metadata = pins.tree_digest(
        _tree(tmp_path / "b", {"x.xml": b"1", ".cache/huggingface/x.xml.metadata": b"etag"})
    )
    assert clean == with_metadata


def test_an_empty_directory_has_no_digest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pins.tree_digest(tmp_path)


def test_verify_rejects_a_snapshot_that_is_not_the_pinned_one(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"x.xml": b"1"})
    pin = pins.HubPin(repo_id="org/name", repo_type="model", revision="0" * 40, digest="f" * 64)
    with pytest.raises(pins.DigestMismatch, match="org/name"):
        pins.verify(pin, root)
    matching = pins.HubPin(
        repo_id="org/name", repo_type="model", revision="0" * 40, digest=pins.tree_digest(root)
    )
    assert pins.verify(matching, root) == matching.digest
    assert matching.dirname == "name"


def test_the_package_hands_out_whatever_the_adapter_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    """`load_fixture` delegates; the real adapter needs LeRobot, so a stand-in is used here."""
    fixture = (object(), object(), 280)
    stand_in = types.ModuleType("fathomline.adapters.lerobot.adapter")
    stand_in.load_fixture = lambda: fixture  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, stand_in.__name__, stand_in)
    assert lerobot.load_fixture() is fixture


def test_installed_reports_whether_lerobot_can_be_found(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find_spec = importlib.util.find_spec
    for present in (True, False):
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, present=present: (
                (object() if present else None) if name == "lerobot" else real_find_spec(name)
            ),
        )
        assert lerobot.installed() is present
