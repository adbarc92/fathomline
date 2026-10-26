"""The image build step, `warm.py`, without the network or the simulator.

The claim the image rests on is "the build fails on a digest mismatch". A build
that has never seen a mismatch does not prove it; these tests do, on every PR.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from fathomline.adapters.lerobot import pins, warm


@pytest.fixture
def libero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in `hf-libero` package: just enough for `find_spec("libero")`."""
    package = tmp_path / "site" / "libero"
    (package / "libero").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "libero-config"))
    return package / "libero"


@pytest.fixture
def snapshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[pins.HubPin, Path]:
    """Every pin's target directory, under tmp_path, and a `fetch` that writes one file."""
    targets = {pin: tmp_path / "artifacts" / pin.dirname for pin in pins.PINS}
    monkeypatch.setattr(pins, "artifact_dirs", lambda: targets)

    def fetch(pin: pins.HubPin, target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / "payload.bin").write_bytes(pin.repo_id.encode())

    monkeypatch.setattr(warm, "fetch", fetch)
    return targets


def test_the_build_fails_when_a_snapshot_does_not_hash_to_its_pin(
    snapshots: dict[pins.HubPin, Path], libero: Path
) -> None:
    """Nothing in `main` catches the mismatch, so the Dockerfile's RUN exits non-zero."""
    with pytest.raises(pins.DigestMismatch, match=pins.POLICY.repo_id):
        warm.main([])
    config = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    assert not config.exists(), "a failed build must not go on to configure LIBERO"


def test_the_build_passes_when_every_snapshot_matches(
    snapshots: dict[pins.HubPin, Path],
    libero: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    matching = tuple(
        pins.HubPin(
            repo_id=pin.repo_id,
            repo_type=pin.repo_type,
            revision=pin.revision,
            digest=pins.tree_digest(_fetched(pin, path)),
        )
        for pin, path in snapshots.items()
    )
    monkeypatch.setattr(
        pins, "artifact_dirs", lambda: dict(zip(matching, snapshots.values(), strict=True))
    )
    assert warm.main([]) == 0
    assert capsys.readouterr().out.count(" ok\n") == len(pins.PINS)


def test_print_digests_reports_without_checking(
    snapshots: dict[pins.HubPin, Path], libero: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The way a pin is recomputed; the Dockerfile never passes this flag."""
    assert warm.main(["--print-digests"]) == 0
    out = capsys.readouterr().out
    for pin in pins.PINS:
        assert f"{pin.repo_id}@{pin.revision}: " in out


def test_the_libero_config_points_at_the_package_so_import_never_prompts(
    libero: Path,
) -> None:
    config = warm.write_libero_config()
    lines = dict(line.split(": ", 1) for line in config.read_text().splitlines())
    assert set(lines) == {"benchmark_root", "bddl_files", "init_states", "datasets", "assets"}
    assert Path(lines["assets"]) == libero / "assets"
    assert Path(lines["datasets"]) == libero.parent / "datasets"


def test_fetch_asks_the_hub_for_the_pinned_revision_never_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda repo_id, **kwargs: calls.append({"repo_id": repo_id, **kwargs})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    warm.fetch(pins.VLM, tmp_path)
    warm.fetch(pins.LIBERO_ASSETS, tmp_path)

    vlm, assets = calls
    assert vlm["revision"] == pins.VLM.revision
    assert vlm["ignore_patterns"] == ["onnx/*"]
    assert assets["revision"] == pins.LIBERO_ASSETS.revision
    assert assets["repo_type"] == "dataset"
    assert assets["ignore_patterns"] is None


def test_artifact_dirs_put_the_assets_where_libero_looks_first(libero: Path) -> None:
    dirs = pins.artifact_dirs()
    assert dirs[pins.LIBERO_ASSETS] == libero / "assets"
    assert dirs[pins.POLICY] == pins.ARTIFACT_ROOT / "smolvla_libero"
    assert dirs[pins.VLM] == pins.ARTIFACT_ROOT / "SmolVLM2-500M-Video-Instruct"


def test_without_hf_libero_the_asset_location_is_an_error_not_a_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None if name == "libero" else real_find_spec(name)
    )
    with pytest.raises(ModuleNotFoundError, match="hf-libero"):
        pins.libero_package_dir()


def _fetched(pin: pins.HubPin, target: Path) -> Path:
    warm.fetch(pin, target)
    return target
