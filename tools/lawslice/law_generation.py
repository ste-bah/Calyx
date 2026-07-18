#!/usr/bin/env python3
"""Durable, fail-closed publication for lawslice artifact generations."""

from __future__ import annotations

from builtins import ExceptionGroup
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import sys
import tempfile

from structured_error import StructuredError


class GenerationError(StructuredError):
    """A generation could not be constructed or independently verified."""

    code = "law_generation_error"
    default_remediation = (
        "use a new absent destination or repair the named generation bytes, then republish from zero"
    )


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish without a check-then-rename overwrite race."""
    if sys.platform == "win32":
        # MoveFileEx without MOVEFILE_REPLACE_EXISTING is what os.rename uses
        # on Windows; an existing destination is an error.
        os.rename(source, destination)
        return
    if sys.platform.startswith("linux"):
        import ctypes
        import errno

        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise GenerationError(
                "Linux libc does not expose renameat2; cannot guarantee no-replace publication"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        at_fdcwd = -100
        rename_noreplace = 1
        result = renameat2(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(destination),
            rename_noreplace,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise GenerationError(
                "generation destination appeared during publication: %s" % destination
            )
        raise OSError(error_number, os.strerror(error_number), str(destination))
    raise GenerationError(
        "atomic no-replace directory publication is unsupported on %s" % sys.platform
    )


def _member_name(name: str) -> str:
    candidate = Path(name)
    if (
        not name
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.name in {".", ".."}
    ):
        raise GenerationError("generation member must be one plain file name: %r" % name)
    return candidate.name


def _read_publish_lock_owner(lock_path: Path) -> str:
    """Read only plain, bounded owner evidence for an existing lock."""
    try:
        lock_state = lock_path.lstat()
        if not stat.S_ISDIR(lock_state.st_mode) or lock_path.is_symlink():
            return "untrusted lock entry mode=%o" % stat.S_IFMT(lock_state.st_mode)
        owner_path = lock_path / "owner.json"
        owner_state = owner_path.lstat()
        if not stat.S_ISREG(owner_state.st_mode) or owner_path.is_symlink():
            return "owner.json is not a plain file"
        if owner_state.st_size > 64 * 1024:
            return "owner.json exceeds 65536 bytes"
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(owner_path, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (
                owner_state.st_dev,
                owner_state.st_ino,
            ):
                return "owner.json changed while opening"
            payload = os.read(descriptor, 64 * 1024 + 1)
        finally:
            os.close(descriptor)
        if len(payload) > 64 * 1024:
            return "owner.json exceeds 65536 bytes"
        value = json.loads(payload.decode("utf-8"))
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except BaseException as error:
        return "unreadable owner evidence: %s: %s" % (type(error).__name__, error)


class GenerationPublisher:
    """Build one sibling staging directory and expose it with one rename.

    The final directory is the source of truth. It is never created until all
    members and the manifest have been fsynced and independently re-read.
    Existing destinations are rejected, including broken symlinks.
    """

    def __init__(self, destination: str | os.PathLike[str], manifest_name: str):
        self.destination = Path(destination).absolute()
        self.parent = self.destination.parent
        self.manifest_name = _member_name(manifest_name)
        self.lock_path = self.parent / (".%s.publisher-lock" % self.destination.name)
        self.staging: Path | None = None
        self._handles: dict[str, object] = {}
        self._published = False
        self._finalized = False
        self._lock_held = False
        self._owner_durable = False
        if not self.parent.is_dir():
            raise GenerationError("generation parent does not exist: %s" % self.parent)
        self._acquire_publish_lock()
        try:
            orphans = self._orphan_staging_paths()
            if orphans:
                raise GenerationError(
                    "abandoned generation staging exists; inspect and explicitly relocate it "
                    "before retrying destination %s: %s"
                    % (self.destination, ", ".join(str(path) for path in orphans))
                )
            if os.path.lexists(self.destination):
                raise GenerationError(
                    "generation destination already exists; refusing overwrite: %s"
                    % self.destination
                )
            self.staging = Path(
                tempfile.mkdtemp(
                    prefix=".%s.staging." % self.destination.name,
                    dir=self.parent,
                )
            )
        except BaseException as error:
            try:
                self._release_publish_lock()
            except BaseException as release_error:
                raise ExceptionGroup(
                    "generation publisher initialization and lock cleanup both failed",
                    [error, release_error],
                ) from None
            raise

    def _acquire_publish_lock(self) -> None:
        try:
            os.mkdir(self.lock_path, 0o700)
        except FileExistsError:
            raise GenerationError(
                "generation publisher lock already exists; a publisher is active or a prior "
                "publisher crashed. Inspect and explicitly recover %s; owner=%s"
                % (self.lock_path, _read_publish_lock_owner(self.lock_path))
            ) from None
        except OSError as error:
            raise GenerationError(
                "cannot create generation publisher lock %s: %s"
                % (self.lock_path, error)
            ) from error
        self._lock_held = True
        _fsync_directory(self.parent)
        owner = {
            "schema_version": 1,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
            "destination": str(self.destination),
        }
        owner_path = self.lock_path / "owner.json"
        try:
            with open(owner_path, "x", encoding="utf-8", newline="\n") as output:
                json.dump(owner, output, ensure_ascii=False, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            _fsync_directory(self.lock_path)
            self._owner_durable = True
        except BaseException as error:
            # Leave the atomic lock directory in place. Without durable owner
            # evidence, silently reopening publication would be unsafe.
            raise GenerationError(
                "generation publisher lock owner evidence is not durable at %s: %s"
                % (owner_path, error)
            ) from error

    def _orphan_staging_paths(self) -> list[Path]:
        prefix = ".%s.staging." % self.destination.name
        return sorted(
            (entry for entry in self.parent.iterdir() if entry.name.startswith(prefix)),
            key=lambda entry: entry.name,
        )

    def _release_publish_lock(self) -> None:
        if not self._lock_held:
            return
        if not self._owner_durable:
            raise GenerationError(
                "refusing to remove publisher lock without durable owner evidence: %s"
                % self.lock_path
            )
        entries = sorted(entry.name for entry in self.lock_path.iterdir())
        if entries != ["owner.json"]:
            raise GenerationError(
                "publisher lock contains unexpected state; refusing cleanup: lock=%s entries=%r"
                % (self.lock_path, entries)
            )
        (self.lock_path / "owner.json").unlink()
        _fsync_directory(self.lock_path)
        self.lock_path.rmdir()
        _fsync_directory(self.parent)
        self._lock_held = False

    def path(self, name: str) -> Path:
        if self.staging is None:
            raise GenerationError("generation publisher has no staging directory")
        return self.staging / _member_name(name)

    def open_text(self, name: str, *, newline: str = "\n"):
        member = _member_name(name)
        if member == self.manifest_name:
            raise GenerationError("manifest is reserved for publish(): %s" % member)
        if member in self._handles or self.path(member).exists():
            raise GenerationError("generation member opened twice: %s" % member)
        handle = open(self.path(member), "x", encoding="utf-8", newline=newline)
        self._handles[member] = handle
        return handle

    def open_binary(self, name: str):
        member = _member_name(name)
        if member == self.manifest_name:
            raise GenerationError("manifest is reserved for publish(): %s" % member)
        if member in self._handles or self.path(member).exists():
            raise GenerationError("generation member opened twice: %s" % member)
        handle = open(self.path(member), "xb")
        self._handles[member] = handle
        return handle

    def write_bytes(self, name: str, value: bytes) -> None:
        if not isinstance(value, bytes):
            raise GenerationError("binary generation member must be bytes: %s" % name)
        self.open_binary(name).write(value)

    def write_json(self, name: str, value: object) -> None:
        handle = self.open_text(name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    def _close_members(self) -> None:
        errors: list[BaseException] = []
        for name, handle in self._handles.items():
            try:
                if not handle.closed:
                    handle.flush()
                    os.fsync(handle.fileno())
                    handle.close()
            except BaseException as error:
                errors.append(GenerationError("failed to close %s: %s" % (name, error)))
        if errors:
            raise ExceptionGroup("generation member close failures", errors)

    def publish(self, manifest: dict) -> Path:
        if self._finalized or self._published:
            raise GenerationError("generation publisher was already finalized")
        if self.staging is None:
            raise GenerationError("generation publisher has no staging directory")
        self._close_members()
        members = {}
        for path in sorted(self.staging.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.is_symlink():
                raise GenerationError("unexpected staged generation entry: %s" % path)
            members[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        committed = dict(manifest)
        committed["files"] = members
        manifest_path = self.path(self.manifest_name)
        with open(manifest_path, "x", encoding="utf-8", newline="\n") as output:
            json.dump(committed, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        _fsync_directory(self.staging)

        # Verify from fresh handles before the generation is made visible.
        verify_generation(self.staging, self.manifest_name, expected=committed)
        _rename_directory_noreplace(self.staging, self.destination)
        self._published = True
        _fsync_directory(self.parent)
        self._release_publish_lock()
        self._finalized = True
        return self.destination

    def abort(self) -> None:
        if self._published or self._finalized:
            return
        close_errors: list[BaseException] = []
        for name, handle in self._handles.items():
            try:
                if not handle.closed:
                    handle.close()
            except BaseException as error:
                close_errors.append(
                    GenerationError("failed to abort member %s: %s" % (name, error))
                )
        try:
            if self.staging is None:
                raise GenerationError("generation publisher has no staging directory to abort")
            shutil.rmtree(self.staging)
            _fsync_directory(self.parent)
        except FileNotFoundError as error:
            close_errors.append(
                GenerationError(
                    "generation staging disappeared before abort cleanup: %s" % self.staging
                )
            )
        except BaseException as error:
            close_errors.append(
                GenerationError("failed to remove staging %s: %s" % (self.staging, error))
            )
        if not close_errors:
            try:
                self._release_publish_lock()
            except BaseException as error:
                close_errors.append(error)
        self._finalized = True
        if close_errors:
            raise ExceptionGroup("generation abort failures", close_errors)

    def __enter__(self) -> "GenerationPublisher":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None:
            try:
                self.abort()
            except BaseException as abort_error:
                raise ExceptionGroup(
                    "generation failed and cleanup also failed", [exc, abort_error]
                ) from None
        elif not self._published:
            self.abort()
            raise GenerationError("generation context exited without publish()")
        return False


def verify_generation(
    directory: str | os.PathLike[str],
    manifest_name: str,
    *,
    expected: dict | None = None,
) -> dict:
    root = Path(directory).absolute()
    if not root.is_dir() or root.is_symlink():
        raise GenerationError("generation is not a plain directory: %s" % root)
    manifest_path = root / _member_name(manifest_name)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise GenerationError("generation manifest is missing: %s" % manifest_path)
    try:
        with open(manifest_path, "r", encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise GenerationError("cannot read generation manifest %s: %s" % (manifest_path, error))
    if expected is not None and manifest != expected:
        raise GenerationError("manifest readback differs from the value written")
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise GenerationError("manifest files must be a non-empty object")
    actual_names = {
        item.name
        for item in root.iterdir()
        if item.name != manifest_path.name
    }
    if actual_names != set(declared):
        raise GenerationError(
            "generation members differ from manifest: actual=%r declared=%r"
            % (sorted(actual_names), sorted(declared))
        )
    for name, facts in declared.items():
        path = root / _member_name(name)
        if not path.is_file() or path.is_symlink():
            raise GenerationError("declared member is not a plain file: %s" % path)
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if facts != {"bytes": actual_size, "sha256": actual_hash}:
            raise GenerationError(
                "member readback mismatch for %s: declared=%r actual=%r"
                % (name, facts, {"bytes": actual_size, "sha256": actual_hash})
            )
    return manifest


def generation_member(
    directory: str | os.PathLike[str], manifest_name: str, member: str
) -> Path:
    manifest = verify_generation(directory, manifest_name)
    name = _member_name(member)
    if name not in manifest["files"]:
        raise GenerationError("member %s is not declared by %s" % (name, manifest_name))
    return Path(directory).absolute() / name
