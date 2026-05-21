"""Lightweight memory instrumentation helpers.

This module is intentionally dependency-free (no psutil/memray).
It is designed for answering a simple question during long runs:
is RSS monotonically increasing (leak-like), or is it a transient peak?
"""

from __future__ import annotations

import gc
import logging
import os
import time

from dataclasses import dataclass
from pathlib import Path

console_logger = logging.getLogger(__name__)


def _read_proc_status_kb() -> dict[str, int]:
    """Read selected fields from /proc/self/status (Linux/WSL)."""
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return {}

    wanted = {
        "VmRSS": "rss_kb",
        "VmHWM": "hwm_kb",
        "VmSize": "vms_kb",
        "VmSwap": "swap_kb",
        "RssAnon": "rss_anon_kb",
        "RssFile": "rss_file_kb",
        "RssShmem": "rss_shmem_kb",
    }

    out: dict[str, int] = {}
    try:
        for line in status_path.read_text().splitlines():
            # Example: "VmRSS:\t  123456 kB"
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            key = key.strip()
            if key not in wanted:
                continue
            parts = rest.strip().split()
            if not parts:
                continue
            try:
                out[wanted[key]] = int(parts[0])
            except ValueError:
                continue
    except Exception:
        return {}

    return out


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def memory_debug_enabled() -> bool:
    return _env_flag("SCENESMITH_MEMLOG")


def tracemalloc_enabled() -> bool:
    return _env_flag("SCENESMITH_TRACEMALLOC")


def asset_memory_debug_enabled() -> bool:
    return _env_flag("SCENESMITH_MEMLOG_ASSETS")


def asset_tracemalloc_enabled() -> bool:
    return _env_flag("SCENESMITH_TRACEMALLOC_ASSETS")


@dataclass(frozen=True)
class MemorySample:
    t_unix: float
    tag: str
    rss_kb: int | None = None
    swap_kb: int | None = None
    hwm_kb: int | None = None
    vms_kb: int | None = None
    rss_anon_kb: int | None = None
    rss_file_kb: int | None = None
    rss_shmem_kb: int | None = None
    gc_counts: tuple[int, int, int] | None = None

    @staticmethod
    def collect(tag: str) -> "MemorySample":
        proc = _read_proc_status_kb()
        return MemorySample(
            t_unix=time.time(),
            tag=tag,
            rss_kb=proc.get("rss_kb"),
            swap_kb=proc.get("swap_kb"),
            hwm_kb=proc.get("hwm_kb"),
            vms_kb=proc.get("vms_kb"),
            rss_anon_kb=proc.get("rss_anon_kb"),
            rss_file_kb=proc.get("rss_file_kb"),
            rss_shmem_kb=proc.get("rss_shmem_kb"),
            gc_counts=gc.get_count(),
        )


def append_memory_csv(output_dir: Path, sample: MemorySample) -> None:
    """Append a memory sample to output_dir/memory.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "memory.csv"
    is_new = not path.exists()

    header = (
        "t_unix,tag,rss_kb,swap_kb,hwm_kb,vms_kb,"
        "rss_anon_kb,rss_file_kb,rss_shmem_kb,gc0,gc1,gc2\n"
    )
    line = (
        f"{sample.t_unix:.3f},{sample.tag},"
        f"{'' if sample.rss_kb is None else sample.rss_kb},"
        f"{'' if sample.swap_kb is None else sample.swap_kb},"
        f"{'' if sample.hwm_kb is None else sample.hwm_kb},"
        f"{'' if sample.vms_kb is None else sample.vms_kb},"
        f"{'' if sample.rss_anon_kb is None else sample.rss_anon_kb},"
        f"{'' if sample.rss_file_kb is None else sample.rss_file_kb},"
        f"{'' if sample.rss_shmem_kb is None else sample.rss_shmem_kb},"
        f"{'' if sample.gc_counts is None else sample.gc_counts[0]},"
        f"{'' if sample.gc_counts is None else sample.gc_counts[1]},"
        f"{'' if sample.gc_counts is None else sample.gc_counts[2]}\n"
    )

    try:
        with open(path, "a", encoding="utf-8") as f:
            if is_new:
                f.write(header)
            f.write(line)
    except Exception as e:
        console_logger.debug(f"Failed to write memory.csv to {path}: {e}")


def maybe_log_memory(output_dir: Path, tag: str, also_print: bool = False) -> None:
    """Log memory to memory.csv when SCENESMITH_MEMLOG is enabled."""
    if not memory_debug_enabled():
        return
    sample = MemorySample.collect(tag=tag)
    append_memory_csv(output_dir=output_dir, sample=sample)
    if also_print and sample.rss_kb is not None:
        console_logger.info(
            f"[mem] {tag}: rss={sample.rss_kb/1024:.1f}MiB "
            f"swap={0.0 if sample.swap_kb is None else sample.swap_kb/1024:.1f}MiB"
        )


def maybe_start_tracemalloc() -> None:
    """Start tracemalloc if SCENESMITH_TRACEMALLOC is enabled."""
    if not tracemalloc_enabled():
        return
    try:
        import tracemalloc

        if not tracemalloc.is_tracing():
            # 25 frames is usually enough to spot hot allocation sites.
            tracemalloc.start(25)
            console_logger.info("tracemalloc enabled (SCENESMITH_TRACEMALLOC=1)")
    except Exception as e:
        console_logger.warning(f"Failed to start tracemalloc: {e}")


def maybe_dump_tracemalloc(output_dir: Path, tag: str, limit: int = 50) -> None:
    """Dump a tracemalloc snapshot when enabled."""
    if not tracemalloc_enabled():
        return
    try:
        import tracemalloc

        if not tracemalloc.is_tracing():
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        snap = tracemalloc.take_snapshot()
        stats = snap.statistics("lineno")
        path = output_dir / f"tracemalloc_{tag}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"tag={tag}\n")
            f.write(f"time={time.time():.3f}\n\n")
            for stat in stats[:limit]:
                f.write(str(stat) + "\n")
    except Exception as e:
        console_logger.debug(f"Failed to dump tracemalloc snapshot: {e}")


def maybe_log_asset_memory(
    output_dir: Path, tag: str, also_print: bool = False
) -> None:
    """Log memory for per-asset steps when SCENESMITH_MEMLOG_ASSETS is enabled."""
    if not asset_memory_debug_enabled():
        return
    sample = MemorySample.collect(tag=tag)
    append_memory_csv(output_dir=output_dir, sample=sample)
    if also_print and sample.rss_kb is not None:
        console_logger.info(
            f"[mem] {tag}: rss={sample.rss_kb/1024:.1f}MiB "
            f"swap={0.0 if sample.swap_kb is None else sample.swap_kb/1024:.1f}MiB"
        )


def maybe_dump_asset_tracemalloc(output_dir: Path, tag: str, limit: int = 50) -> None:
    """Dump tracemalloc snapshot for per-asset steps when enabled."""
    if not asset_tracemalloc_enabled():
        return
    maybe_dump_tracemalloc(output_dir=output_dir, tag=tag, limit=limit)
