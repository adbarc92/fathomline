"""Every artifact the LeRobot fixture runs on, pinned by revision and by content.

Standard library only: `warm.py` imports this at image build time, the adapter
at run time, and the unit tests on any platform.

A revision SHA fixes what the Hub serves; the digest fixes what landed on disk.
The LIBERO assets need both, because upstream fetches them with no revision at
all (spike finding 9): `hf-libero` downloads whatever `main` is that day.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

#: D2. Results are bound to the LeRobot version (spike finding 4), so the adapter
#: refuses to run on any other.
LEROBOT_VERSION = "0.6.1"

#: D1. LIBERO `libero_object`, task 0.
SUITE = "libero_object"
TASK_ID = 0

#: SmolVLA's feature validation runs before its preprocessor, so the map in the
#: checkpoint's `policy_preprocessor.json` is not enough on its own (spike finding 3).
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}

#: Actions played from each predicted chunk before the policy looks again. The checkpoint
#: ships `n_action_steps: 50`, the framework default, so a whole chunk runs blind. That is
#: SmolVLA's worst setting on LIBERO (paper Table 13: Object 70% at 50, 94% at 10), and
#: huggingface/lerobot#4614 reports the same gap for this checkpoint. 10 is the setting the
#: LeRobot LIBERO docs use for pi0.5 and the one #4614 proposes.
N_ACTION_STEPS = 10

#: Where `warm.py` puts the pinned snapshots inside the image.
ARTIFACT_ROOT = Path(os.environ.get("FATHOMLINE_ARTIFACTS", "/opt/fathomline"))

#: `huggingface_hub` keeps download metadata here inside a `local_dir`. It is
#: bookkeeping, not content, and it differs between downloads.
_HUB_METADATA_DIR = ".cache"

_CHUNK = 1 << 20


@dataclass(frozen=True, slots=True)
class HubPin:
    """One Hub snapshot: where it comes from, and what it must hash to."""

    repo_id: str
    repo_type: str
    revision: str
    #: `tree_digest` of the downloaded snapshot.
    digest: str
    ignore_patterns: tuple[str, ...] = ()

    @property
    def dirname(self) -> str:
        return self.repo_id.split("/")[1]


#: D1's checkpoint. Upstream `main` moved on 2026-10-12; this revision did not.
POLICY = HubPin(
    repo_id="lerobot/smolvla_libero",
    repo_type="model",
    revision="31d453f7edd78c839a8bbc39744a292686daf0de",
    digest="1c59f05ac24f1121e2b02cdedc69174779795903f2cb3ab65535099af5e6523e",
)

#: SmolVLA's VLM backbone. The checkpoint names it by repo id only, so without a
#: pin the constructor fetches whatever `main` is (spike finding 10). Its weights
#: are overwritten by the checkpoint's; its config and tokenizer are not. The ONNX
#: exports (~5.9 GB) are never read.
VLM = HubPin(
    repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    repo_type="model",
    revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467",
    digest="6f49f584038c421d9bcefe03ae7af7986cc9bec8bfc11df41c5a92728d225241",
    ignore_patterns=("onnx/*",),
)

#: The meshes and textures LIBERO scenes load. Placed in the `hf-libero` package
#: directory, which `libero.libero.get_assets_path()` checks before it downloads.
LIBERO_ASSETS = HubPin(
    repo_id="lerobot/libero-assets",
    repo_type="dataset",
    revision="0b3ea86be5fe169d0fd036ae63d1070ec09e90f6",
    digest="51ba6b7a3a701fd8dfe870f9291ec138db7d5a646dccc1954137332d848168a2",
)

PINS = (POLICY, VLM, LIBERO_ASSETS)


def libero_package_dir() -> Path:
    """`hf-libero`'s `libero/libero` directory, found without importing it.

    Importing `libero.libero` with no config file prompts on stdin.
    """
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("hf-libero is not installed (it comes with the `lerobot` extra)")
    return Path(next(iter(spec.submodule_search_locations))) / "libero"


def artifact_dirs() -> dict[HubPin, Path]:
    """Where each pinned snapshot lives in the image."""
    return {
        POLICY: ARTIFACT_ROOT / POLICY.dirname,
        VLM: ARTIFACT_ROOT / VLM.dirname,
        LIBERO_ASSETS: libero_package_dir() / "assets",
    }


def tree_digest(root: Path) -> str:
    """SHA-256 over every file under `root`: its POSIX relative path and its SHA-256.

    Sorted by the POSIX path string, so the digest is the same on any host (a
    Windows `Path` sorts case-insensitively). The Hub's download metadata is left out.
    """
    files = sorted(
        (path.relative_to(root).as_posix(), path)
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).parts[0] != _HUB_METADATA_DIR
    )
    if not files:
        raise FileNotFoundError(f"no files under {root}")
    digest = hashlib.sha256()
    for relative, path in files:
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(_CHUNK):
                file_digest.update(chunk)
        digest.update(f"{relative}\0{file_digest.hexdigest()}\n".encode())
    return digest.hexdigest()


class DigestMismatch(RuntimeError):
    """A pinned artifact on disk is not the one the pin names."""


def verify(pin: HubPin, root: Path) -> str:
    """Return the digest of `root`, or raise if it is not `pin.digest`."""
    actual = tree_digest(root)
    if actual != pin.digest:
        raise DigestMismatch(
            f"{pin.repo_id}@{pin.revision[:12]} at {root}: sha256 {actual}, pinned {pin.digest}"
        )
    return actual
