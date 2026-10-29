"""Host facts: what a run ran *on*, recorded so a determinism claim can be scoped.

The action-trajectory hash has differed between CI runners running identical
images: it depends on torch's intra-op thread count and on the instruction set
its kernels dispatch to (D9a). D9b rules the claim: bit-exact reproduction on
any x86_64 host with AVX2, under a pin the image sets (`PINNED_TORCH`). Every
persisted run records what that claim is scoped by: the CPU model and its SIMD
flags, how many CPUs the process could use, and the numeric runtime's thread
and kernel-dispatch settings.

Nothing here changes a thread count or a kernel. `pin_mismatches` only reports
whether torch is running under the pin; `sweep` refuses to run when it is not.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: Environment variables that change how many threads the numeric stack uses or
#: which instruction set its kernels dispatch to. Recorded when set.
THREADING_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "ATEN_CPU_CAPABILITY",
    "MKL_ENABLE_INSTRUCTIONS",
    "MKL_CBWR",
    "ONEDNN_MAX_CPU_ISA",
    "DNNL_MAX_CPU_ISA",
)

#: `/proc/cpuinfo` flags kept: the SIMD extensions a kernel can dispatch on.
_SIMD_PREFIXES = ("sse", "ssse", "avx", "fma", "f16c", "amx")

CPUINFO = Path("/proc/cpuinfo")

#: The torch settings a bit-exact claim is scoped to (D9b). The image sets them:
#: `OMP_NUM_THREADS` fixes the intra-op thread count and `ATEN_CPU_CAPABILITY`
#: the kernels' instruction set. A CPU without AVX2 reports a lower capability,
#: whatever the variable asks for, so it fails the pin rather than passing it.
PINNED_TORCH: dict[str, str | int] = {"num_threads": 2, "cpu_capability": "AVX2"}


@dataclass(frozen=True, slots=True)
class Host:
    platform: str
    machine: str
    cpu_model: str
    #: The SIMD flags the CPU reports, sorted. Empty where `/proc/cpuinfo` is absent.
    cpu_flags: tuple[str, ...]
    #: CPUs on the machine (`os.cpu_count()`).
    logical_cpus: int | None
    #: CPUs this process may run on (its affinity mask; a container's `--cpuset-cpus`).
    usable_cpus: int | None
    python_version: str
    #: The `THREADING_ENV` variables that were set, and their values.
    threading_env: dict[str, str] = field(default_factory=dict)
    #: Torch's settings, if the fixture imported torch; None if it did not.
    torch: dict[str, str | int] | None = None


def capture_host(cpuinfo: Path = CPUINFO) -> Host:
    model, flags = _read_cpuinfo(cpuinfo)
    return Host(
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_model=model or platform.processor() or "unknown",
        cpu_flags=flags,
        logical_cpus=os.cpu_count(),
        usable_cpus=_usable_cpus(),
        python_version=platform.python_version(),
        threading_env={name: os.environ[name] for name in THREADING_ENV if name in os.environ},
        torch=_torch_facts(),
    )


def _read_cpuinfo(path: Path) -> tuple[str, tuple[str, ...]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", ()
    model = ""
    flags: tuple[str, ...] = ()
    # Every processor repeats the same block; the first one is enough.
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "model name" and not model:
            model = value.strip()
        elif key == "flags" and not flags:
            flags = tuple(sorted(f for f in value.split() if f.startswith(_SIMD_PREFIXES)))
    return model, flags


def _usable_cpus() -> int | None:
    # Linux only; elsewhere the affinity mask is not exposed by `os`.
    affinity = getattr(os, "sched_getaffinity", None)
    return len(affinity(0)) if affinity is not None else None


def _torch_facts() -> dict[str, str | int] | None:
    # Read only if the fixture already imported it: importing torch here would be
    # a side effect of recording, and would put a large import in every fake run.
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    return {
        "version": str(torch.__version__),
        "num_threads": int(torch.get_num_threads()),
        "num_interop_threads": int(torch.get_num_interop_threads()),
        "cpu_capability": str(torch.backends.cpu.get_cpu_capability()),
    }


def pin_mismatches(host: Host) -> tuple[str, ...]:
    """Each torch setting that is not the pinned one; empty when the run is under the pin.

    A run that never imported torch (the fakes) has nothing to pin.
    """
    if host.torch is None:
        return ()
    return tuple(
        f"torch {key} is {host.torch.get(key)!r}, pinned {pinned!r}"
        for key, pinned in PINNED_TORCH.items()
        if host.torch.get(key) != pinned
    )
