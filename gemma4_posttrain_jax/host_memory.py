"""Read Linux process RSS and return unused glibc pages to the operating system."""

from __future__ import annotations

import ctypes
import os
import time
from pathlib import Path
from typing import Any


def resident_memory_bytes() -> int:
    fields = Path("/proc/self/statm").read_text().split()
    if len(fields) < 2 or not fields[1].isdigit():
        raise RuntimeError("无法读取当前进程RSS")
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")


def trim_host_allocator() -> dict[str, Any]:
    """Request glibc page reclamation and record its return value, elapsed time, and RSS before and after."""
    before = resident_memory_bytes()
    started = time.perf_counter()
    libc = ctypes.CDLL(None)
    trim = libc.malloc_trim
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    returned = int(trim(0))
    return {
        "enabled": True,
        "returned": returned,
        "rss_before_bytes": before,
        "rss_after_bytes": resident_memory_bytes(),
        "wall_s": time.perf_counter() - started,
    }
