"""Image build step: put every pinned artifact on disk and prove it is the pinned one.

    python -m fathomline.adapters.lerobot.warm

Run once, in the Dockerfile, before `HF_HUB_OFFLINE=1` is set. Afterwards an
eval reads nothing from the network. Three things happen:

1. each `pins.PINS` snapshot is downloaded at its revision SHA, never `main`;
2. each is hashed, and the build fails if a digest differs from its pin;
3. LIBERO's config file is written, so importing `libero.libero` does not stop
   to ask a question on stdin (the spike piped `echo N` into it).

`--print-digests` skips the check and prints what each snapshot hashes to: the
way a pin is (re)computed, and never run in the build.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from fathomline.adapters.lerobot import pins


def fetch(pin: pins.HubPin, target: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        pin.repo_id,
        repo_type=pin.repo_type,
        revision=pin.revision,
        local_dir=target,
        ignore_patterns=list(pin.ignore_patterns) or None,
    )


def write_libero_config() -> Path:
    """The paths `libero.libero.get_default_path_dict()` would write, without the prompt."""
    root = pins.libero_package_dir()
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    config_dir.mkdir(parents=True, exist_ok=True)
    entries = {
        "benchmark_root": root,
        "bddl_files": root / "bddl_files",
        "init_states": root / "init_files",
        "datasets": root.parent / "datasets",
        "assets": root / "assets",
    }
    config = config_dir / "config.yaml"
    config.write_text("".join(f"{key}: {path}\n" for key, path in entries.items()))
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="warm", description=__doc__.splitlines()[0])
    parser.add_argument("--print-digests", action="store_true")
    args = parser.parse_args(argv)

    for pin, target in pins.artifact_dirs().items():
        fetch(pin, target)
        if args.print_digests:
            print(f"{pin.repo_id}@{pin.revision}: {pins.tree_digest(target)}")
        else:
            print(f"{pin.repo_id}@{pin.revision}: sha256 {pins.verify(pin, target)} ok")
    print(f"wrote {write_libero_config()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
