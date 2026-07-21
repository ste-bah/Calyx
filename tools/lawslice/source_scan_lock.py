#!/usr/bin/env python3
"""Fail-closed, cross-revision serialization for lawslice source scans.

Every extractor acquires two real Linux ``flock`` locks in one fixed order:

1. the established physical-source v1 lock (device/inode identity), and
2. the semantic archive-hash/policy lock used by newer extractors.

Holding both makes the transition interoperable in either direction.  An
older physical-only process and a newer semantic-only process each contend
with this implementation, while all current processes use the same ordering.
"""

from __future__ import annotations

from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys

from structured_error import StructuredError


PHYSICAL_FORMAT = "calyx-lawslice-source-scan-lock-v1"
SEMANTIC_FORMAT = "calyx-lawslice-source-scan-lock-v2"
ERROR_CODE = "CALYX_LAWSLICE_SOURCE_LOCK_FAILURE"
CONFLICT_CODE = "CALYX_LAWSLICE_SOURCE_SCAN_BUSY"
ERROR_REMEDIATION = (
    "correct the reported source, runtime, repository, or lock security condition; "
    "the scan did not start and no generation was published"
)
CONFLICT_REMEDIATION = (
    "inspect the recorded owner PID/cgroup/repo/output; stop only a confirmed "
    "obsolete owned unit, then rerun at a new destination"
)
_ROLES = ("clusters", "dockets", "opinions")
_PHYSICAL_ROLES = ("clusters", "dockets", "manifest", "opinions")
_SHA256_CHARS = frozenset("0123456789abcdef")


class SourceScanLockError(StructuredError):
    """A source scan could not acquire a trustworthy kernel lock."""

    def __init__(self, message: str, **context):
        self.code = ERROR_CODE
        super().__init__(message, remediation=ERROR_REMEDIATION, **context)


class SourceScanConflict(SourceScanLockError):
    """Another cooperating extractor owns the same source authority."""

    def __init__(self, message: str, **context):
        super().__init__(message, **context)
        self.code = CONFLICT_CODE
        self.remediation = CONFLICT_REMEDIATION


def make_source_identity(
    sources: dict[str, dict[str, str]],
    *,
    selector_version: str,
    text_policy_version: str,
) -> dict:
    """Return the canonical semantic identity for an exact archive contract."""
    if set(sources) != set(_ROLES):
        raise SourceScanLockError(
            "source identity requires the exact canonical archive roles",
            required_roles=list(_ROLES),
            actual_roles=sorted(sources),
        )
    canonical_sources = {}
    for role in _ROLES:
        value = sources[role]
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise SourceScanLockError(
                "source identity role requires only path and sha256", role=role
            )
        path = value["path"]
        digest = value["sha256"]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise SourceScanLockError(
                "source identity path must be absolute", role=role, path=path
            )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in _SHA256_CHARS for character in digest)
        ):
            raise SourceScanLockError(
                "source identity SHA-256 is malformed", role=role, sha256=digest
            )
        canonical_sources[role] = {"path": path, "sha256": digest}
    policies = {
        "selector_version": selector_version,
        "text_policy_version": text_policy_version,
    }
    for name, value in policies.items():
        if not isinstance(value, str) or not value:
            raise SourceScanLockError(
                "source identity policy must be a non-empty string",
                policy=name,
                value=value,
            )
    return {
        "schema_version": 1,
        "sources": canonical_sources,
        "policies": policies,
    }


def source_identity_key(identity: dict) -> str:
    try:
        encoded = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SourceScanLockError(
            "source identity is not canonical JSON", error=str(error)
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _canonical_physical_sources(sources: dict[str, str]) -> dict[str, dict]:
    if set(sources) != set(_PHYSICAL_ROLES):
        raise SourceScanLockError(
            "physical source lock requires the exact canonical source roles",
            required_roles=list(_PHYSICAL_ROLES),
            actual_roles=sorted(sources),
        )
    canonical = {}
    for role in _PHYSICAL_ROLES:
        supplied = Path(sources[role]).expanduser().absolute()
        try:
            metadata = supplied.lstat()
        except OSError as error:
            raise SourceScanLockError(
                "cannot inspect physical source before lock acquisition",
                role=role,
                path=str(supplied),
                errno=error.errno,
                error=str(error),
            ) from error
        if not stat.S_ISREG(metadata.st_mode) or supplied.is_symlink():
            raise SourceScanLockError(
                "physical source input is not a plain file",
                role=role,
                path=str(supplied),
                is_regular=stat.S_ISREG(metadata.st_mode),
                is_symlink=supplied.is_symlink(),
            )
        canonical[role] = {
            "path": str(supplied.resolve(strict=True)),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
        }
    return canonical


def _physical_identity_key(sources: dict[str, dict]) -> str:
    identity = {
        role: {
            "device": sources[role]["device"],
            "inode": sources[role]["inode"],
        }
        for role in sorted(sources)
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repo_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SourceScanLockError(
            "cannot attest extractor repository revision",
            repo_root=str(repo_root),
            error_type=type(error).__name__,
            error=str(error),
        ) from error
    value = result.stdout.strip().lower()
    if len(value) != 40 or any(character not in _SHA256_CHARS for character in value):
        raise SourceScanLockError(
            "extractor repository revision is not a full Git object ID",
            repo_root=str(repo_root),
            stdout=result.stdout,
            stderr=result.stderr,
        )
    return value


def _process_identity() -> dict:
    try:
        cgroup = Path("/proc/self/cgroup").read_text(encoding="utf-8")
        process_start_ticks = Path("/proc/self/stat").read_text(
            encoding="utf-8"
        ).split()[21]
    except (OSError, IndexError) as error:
        raise SourceScanLockError(
            "cannot read Linux process identity for source scan ownership",
            pid=os.getpid(),
            error_type=type(error).__name__,
            error=str(error),
        ) from error
    return {
        "pid": os.getpid(),
        "uid": os.getuid(),
        "process_start_ticks": process_start_ticks,
        "process_cgroup": cgroup.strip().splitlines(),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
    }


def _kernel_flock_owner(metadata: os.stat_result) -> int | None:
    """Return the PID recorded by Linux for an exclusive flock inode."""
    expected_major = os.major(metadata.st_dev)
    expected_minor = os.minor(metadata.st_dev)
    try:
        lines = Path("/proc/locks").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SourceScanLockError(
            "cannot read Linux kernel lock state",
            path="/proc/locks",
            errno=error.errno,
            error=str(error),
        ) from error
    for line in lines:
        fields = line.split()
        if len(fields) < 8 or fields[1:4] != ["FLOCK", "ADVISORY", "WRITE"]:
            continue
        device_inode = fields[5].split(":")
        if len(device_inode) != 3:
            continue
        try:
            major = int(device_inode[0], 16)
            minor = int(device_inode[1], 16)
            inode = int(device_inode[2])
            pid = int(fields[4])
        except ValueError:
            continue
        if (
            major == expected_major
            and minor == expected_minor
            and inode == metadata.st_ino
        ):
            return pid
    return None


def _read_owner(descriptor: int) -> dict:
    try:
        payload = os.pread(descriptor, 64 * 1024 + 1, 0)
        if len(payload) > 64 * 1024:
            return {"unparseable_owner_record": "owner evidence exceeds 65536 bytes"}
        value = json.loads(payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("owner record is not an object")
        return value
    except BaseException as error:
        return {
            "unparseable_owner_record": "%s: %s"
            % (type(error).__name__, error)
        }


def _validate_owned_directory(path: Path, *, create: bool) -> os.stat_result:
    if create:
        try:
            path.mkdir(mode=0o700, exist_ok=True)
        except OSError as error:
            raise SourceScanLockError(
                "cannot create source scan lock directory",
                path=str(path),
                errno=error.errno,
                error=str(error),
            ) from error
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SourceScanLockError(
            "cannot inspect source scan lock directory",
            path=str(path),
            errno=error.errno,
            error=str(error),
        ) from error
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or path.is_symlink()
        or metadata.st_uid != os.geteuid()
        or mode != 0o700
    ):
        raise SourceScanLockError(
            "source scan lock directory is not a private owned directory",
            path=str(path),
            expected_uid=os.geteuid(),
            actual_uid=metadata.st_uid,
            expected_mode="0700",
            actual_mode="%04o" % mode,
            is_directory=stat.S_ISDIR(metadata.st_mode),
            is_symlink=path.is_symlink(),
        )
    return metadata


class _KernelLockFile:
    def __init__(self, root: Path, name: str, authority: str):
        self.root = root
        self.name = name
        self.authority = authority
        self.path = root / name
        self.root_descriptor: int | None = None
        self.descriptor: int | None = None
        self.held = False

    def acquire(self, record: dict) -> None:
        import fcntl

        _validate_owned_directory(self.root, create=False)
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        self.root_descriptor = os.open(self.root, directory_flags)
        required_flags = ("O_CLOEXEC", "O_NOFOLLOW")
        missing = [name for name in required_flags if not hasattr(os, name)]
        if missing:
            raise SourceScanLockError(
                "secure lock-file open flags are unavailable",
                missing_flags=missing,
                platform=sys.platform,
            )
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            descriptor = os.open(
                self.name, flags, 0o600, dir_fd=self.root_descriptor
            )
        except OSError as error:
            raise SourceScanLockError(
                "cannot securely open source scan lock file",
                authority=self.authority,
                lock_path=str(self.path),
                errno=error.errno,
                error=str(error),
            ) from error
        self.descriptor = descriptor
        metadata = os.fstat(descriptor)
        mode = stat.S_IMODE(metadata.st_mode)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or mode != 0o600
        ):
            raise SourceScanLockError(
                "source scan lock file is not a private owned regular file",
                authority=self.authority,
                lock_path=str(self.path),
                expected_uid=os.geteuid(),
                actual_uid=metadata.st_uid,
                expected_mode="0600",
                actual_mode="%04o" % mode,
                is_regular=stat.S_ISREG(metadata.st_mode),
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in {errno.EACCES, errno.EAGAIN}:
                raise SourceScanLockError(
                    "kernel source scan lock acquisition failed",
                    authority=self.authority,
                    lock_path=str(self.path),
                    errno=error.errno,
                    error=str(error),
                ) from error
            owner = _read_owner(descriptor)
            kernel_owner_pid = _kernel_flock_owner(metadata)
            raise SourceScanConflict(
                "canonical source archives are already owned by another extractor",
                authority=self.authority,
                lock_path=str(self.path),
                requested_output=record["output"],
                kernel_owner_pid=kernel_owner_pid,
                owner_record_matches_kernel=(
                    kernel_owner_pid is not None
                    and owner.get("pid") == kernel_owner_pid
                ),
                owner=owner,
            ) from None
        self.held = True
        encoded = (
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        os.ftruncate(descriptor, 0)
        written = 0
        while written < len(encoded):
            count = os.write(descriptor, encoded[written:])
            if count <= 0:
                raise SourceScanLockError(
                    "source scan lock owner write made no progress",
                    authority=self.authority,
                    lock_path=str(self.path),
                )
            written += count
        os.fsync(descriptor)
        os.fsync(self.root_descriptor)

    def release(self) -> None:
        descriptor = self.descriptor
        self.descriptor = None
        if descriptor is not None:
            try:
                if self.held:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                self.held = False
                os.close(descriptor)
        if self.root_descriptor is not None:
            os.close(self.root_descriptor)
            self.root_descriptor = None


class SourceScanLock:
    """Hold physical-v1 then semantic-v2 locks for one source scan."""

    def __init__(
        self,
        identity: dict,
        *,
        physical_sources: dict[str, str],
        destination: str | os.PathLike[str],
        repo_root: str | os.PathLike[str],
        runtime_root: str | os.PathLike[str] | None = None,
    ):
        if not sys.platform.startswith("linux"):
            raise SourceScanLockError(
                "kernel-backed source scan locking requires Linux flock semantics",
                platform=sys.platform,
            )
        self.identity = identity
        self.key = source_identity_key(identity)
        self.destination = str(Path(destination).expanduser().absolute())
        self.repo_root = Path(repo_root).absolute()
        self.physical_sources = _canonical_physical_sources(physical_sources)
        for role in _ROLES:
            semantic_path = identity["sources"][role]["path"]
            physical_path = self.physical_sources[role]["path"]
            if semantic_path != physical_path:
                raise SourceScanLockError(
                    "semantic and physical source paths disagree",
                    role=role,
                    semantic_path=semantic_path,
                    physical_path=physical_path,
                )
        self.physical_key = _physical_identity_key(self.physical_sources)
        if runtime_root is None:
            self.runtime_root = Path("/run/user") / str(os.getuid())
        else:
            self.runtime_root = Path(runtime_root).absolute()
        _validate_owned_directory(self.runtime_root, create=False)
        self.physical_lock_root = (
            self.runtime_root / "calyx-lawslice-source-locks"
        )
        _validate_owned_directory(self.physical_lock_root, create=True)
        self.physical_lock_path = self.physical_lock_root / (
            self.physical_key + ".lock"
        )
        self.semantic_lock_path = self.runtime_root / (
            "calyx-lawslice-source-%s.lock" % self.key
        )
        self.lock_path = self.semantic_lock_path
        self._physical_lock = _KernelLockFile(
            self.physical_lock_root,
            self.physical_key + ".lock",
            "physical-v1",
        )
        self._semantic_lock = _KernelLockFile(
            self.runtime_root,
            "calyx-lawslice-source-%s.lock" % self.key,
            "semantic-v2",
        )
        self._acquired = False
        self.record: dict | None = None

    def _base_owner(self) -> dict:
        owner = _process_identity()
        owner.update(
            {
                "acquired_at": datetime.now(timezone.utc).isoformat(),
                "host": socket.gethostname(),
                "repo_root": str(self.repo_root),
                "repo_sha": _repo_sha(self.repo_root),
                "output": self.destination,
                "physical_identity_sha256": self.physical_key,
                "semantic_identity_sha256": self.key,
                "physical_lock_path": str(self.physical_lock_path),
                "semantic_lock_path": str(self.semantic_lock_path),
            }
        )
        return owner

    def acquire(self) -> "SourceScanLock":
        if self._acquired:
            raise SourceScanLockError(
                "source scan lock instance was acquired twice",
                physical_lock_path=str(self.physical_lock_path),
                semantic_lock_path=str(self.semantic_lock_path),
            )
        base = self._base_owner()
        physical_record = dict(base)
        physical_record.update(
            {
                "format": PHYSICAL_FORMAT,
                "identity_sha256": self.physical_key,
                "sources": self.physical_sources,
            }
        )
        semantic_record = dict(base)
        semantic_record.update(
            {
                "format": SEMANTIC_FORMAT,
                "identity_sha256": self.key,
                "source_identity": self.identity,
            }
        )
        try:
            self._physical_lock.acquire(physical_record)
            self._semantic_lock.acquire(semantic_record)
        except BaseException:
            self.release()
            raise
        self._acquired = True
        self.record = semantic_record
        return self

    def assert_sources_unchanged(self) -> None:
        """Independently re-read path identity while both locks are held."""
        if not self._acquired:
            raise SourceScanLockError("cannot verify sources without both locks held")
        for role, expected in self.physical_sources.items():
            path = Path(expected["path"])
            try:
                actual = path.lstat()
            except OSError as error:
                raise SourceScanLockError(
                    "locked physical source disappeared",
                    role=role,
                    path=str(path),
                    errno=error.errno,
                    error=str(error),
                ) from error
            if (
                not stat.S_ISREG(actual.st_mode)
                or path.is_symlink()
                or actual.st_dev != expected["device"]
                or actual.st_ino != expected["inode"]
            ):
                raise SourceScanLockError(
                    "locked physical source identity changed",
                    role=role,
                    path=str(path),
                    expected_device=expected["device"],
                    expected_inode=expected["inode"],
                    actual_device=actual.st_dev,
                    actual_inode=actual.st_ino,
                    is_regular=stat.S_ISREG(actual.st_mode),
                    is_symlink=path.is_symlink(),
                )

    def release(self) -> None:
        errors = []
        for lock in (self._semantic_lock, self._physical_lock):
            try:
                lock.release()
            except BaseException as error:
                errors.append(error)
        self._acquired = False
        if errors:
            raise ExceptionGroup("source scan lock release failed", errors)

    def __enter__(self) -> "SourceScanLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.release()
        return False
