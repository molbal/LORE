"""Process-safe advisory file locks, on POSIX and on Windows.

LORE grew up on Linux and used `fcntl.flock` directly in two places: the
library lock (`lore.library_lock`, one file per library that keeps a book's
delete apart from the work writing the same stores) and the sample copy
(`lore.sample`). `fcntl` does not exist on Windows, so importing either
module failed there and `lore ui` could not start.

`lock(fd, exclusive=..., blocking=...)` is the one call both use:

  * POSIX: `fcntl.flock`, shared or exclusive, blocking or not, exactly as
    before (behaviour on Linux and macOS is unchanged).
  * Windows: `kernel32.LockFileEx` through ctypes. The CRT's `msvcrt.locking`
    only offers exclusive locks, which would make a second upload or a
    delete wait behind a processing run that POSIX lets it run beside:
    LockFileEx has the SHARED flag, so the same "many readers, one writer"
    rule holds. A hard kill releases the lock with the process, as on POSIX.
  * Anything else: a no-op, so a single-user install still works rather than
    failing to import.

`BlockingIOError` is what a non-blocking lock raises when it is taken, on
every platform, so the callers' refusal path (`LibraryBusy`) is unchanged.
"""
from __future__ import annotations

import errno
import os

try:  # POSIX
    import fcntl as _fcntl
except ImportError:  # Windows
    _fcntl = None

_LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
_LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
_ERROR_LOCK_VIOLATION = 33


def _windows_types():
    import ctypes
    from ctypes import wintypes

    class OVERLAPPED(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_void_p),
            ("InternalHigh", ctypes.c_void_p),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    return ctypes, wintypes, OVERLAPPED


def _handle(fd: int, wintypes):
    import msvcrt

    return wintypes.HANDLE(msvcrt.get_osfhandle(fd))


def _lock_windows(fd: int, *, exclusive: bool, blocking: bool) -> None:
    ctypes, wintypes, OVERLAPPED = _windows_types()
    kernel32 = ctypes.windll.kernel32
    flags = _LOCKFILE_EXCLUSIVE_LOCK if exclusive else 0
    if not blocking:
        flags |= _LOCKFILE_FAIL_IMMEDIATELY
    overlapped = OVERLAPPED()
    ok = kernel32.LockFileEx(_handle(fd, wintypes), flags, 0, 1, 0,
                             ctypes.byref(overlapped))
    if ok:
        return
    error = kernel32.GetLastError()
    if error == _ERROR_LOCK_VIOLATION:
        raise BlockingIOError(errno.EAGAIN, "the file is locked")
    raise OSError(error, f"LockFileEx failed ({error})")


def _unlock_windows(fd: int) -> None:
    ctypes, wintypes, OVERLAPPED = _windows_types()
    overlapped = OVERLAPPED()
    ctypes.windll.kernel32.UnlockFileEx(_handle(fd, wintypes), 0, 1, 0,
                                        ctypes.byref(overlapped))


def lock(fd: int, *, exclusive: bool = True, blocking: bool = True) -> None:
    """Lock `fd`'s file, raising BlockingIOError if a non-blocking attempt fails."""
    if _fcntl is not None:
        mode = _fcntl.LOCK_EX if exclusive else _fcntl.LOCK_SH
        if not blocking:
            mode |= _fcntl.LOCK_NB
        _fcntl.flock(fd, mode)
        return
    if os.name == "nt":
        _lock_windows(fd, exclusive=exclusive, blocking=blocking)


def unlock(fd: int) -> None:
    """Release a lock taken by `lock` (closing `fd` also releases it)."""
    if _fcntl is not None:
        _fcntl.flock(fd, _fcntl.LOCK_UN)
        return
    if os.name == "nt":
        _unlock_windows(fd)
