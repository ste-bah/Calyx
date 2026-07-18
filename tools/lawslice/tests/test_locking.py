from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest


LAW_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = LAW_DIR.parents[1]
if str(LAW_DIR) not in sys.path:
    sys.path.insert(0, str(LAW_DIR))

from extract_cuyahoga import prepare_bulk_sources  # noqa: E402
from law_generation import (  # noqa: E402
    GenerationError,
    GenerationPublisher,
    verify_generation,
)
from source_scan_lock import (  # noqa: E402
    SourceScanLock,
    SourceScanConflict,
    SourceScanLockError,
    make_source_identity,
)


PUBLISHER_HOLDER = r"""
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, sys.argv[1])
from law_generation import GenerationPublisher
destination = Path(sys.argv[2])
ready = Path(sys.argv[3])
release = Path(sys.argv[4])
parent_pid = int(sys.argv[5])
with GenerationPublisher(destination, "manifest.json") as publisher:
    publisher.write_json("payload.json", {"real": "subprocess"})
    ready.write_text("ready\n", encoding="utf-8")
    while not release.exists():
        if os.getppid() != parent_pid:
            raise SystemExit("publisher test parent exited before release")
        time.sleep(0.01)
    publisher.publish({"format": "publisher-concurrency-v1"})
print(json.dumps({"status": "published", "destination": str(destination)}))
"""


PUBLISHER_COMPETITOR = r"""
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from law_generation import GenerationError, GenerationPublisher
try:
    GenerationPublisher(Path(sys.argv[2]), "manifest.json")
except GenerationError as error:
    print(str(error), file=sys.stderr)
    raise SystemExit(23)
raise SystemExit("competitor unexpectedly acquired publisher")
"""


SOURCE_LOCK_HOLDER = r"""
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, sys.argv[1])
from source_scan_lock import SourceScanLock
identity = json.loads(sys.argv[2])
physical_sources = json.loads(sys.argv[3])
ready = Path(sys.argv[7])
release = Path(sys.argv[8])
parent_pid = int(sys.argv[9])
with SourceScanLock(
    identity,
    physical_sources=physical_sources,
    destination=sys.argv[4],
    repo_root=sys.argv[5],
    runtime_root=sys.argv[6],
) as lock:
    ready.write_text(json.dumps({
        "key": lock.key,
        "path": str(lock.lock_path),
        "physical_key": lock.physical_key,
        "physical_path": str(lock.physical_lock_path),
    }) + "\n", encoding="utf-8")
    while not release.exists():
        if os.getppid() != parent_pid:
            raise SystemExit("source-lock test parent exited before release")
        time.sleep(0.01)
"""


SOURCE_LOCK_ORPHAN_DRIVER = r"""
import os
from pathlib import Path
import subprocess
import sys
import time
holder_code = sys.argv[1]
holder_args = sys.argv[2:-1]
child_pid_path = Path(sys.argv[-1])
ready = Path(holder_args[-2])
child = subprocess.Popen([
    sys.executable,
    "-c",
    holder_code,
    *holder_args,
    str(os.getpid()),
])
deadline = time.monotonic() + 10
while time.monotonic() < deadline:
    if ready.is_file():
        child_pid_path.write_text(str(child.pid) + "\n", encoding="utf-8")
        os._exit(0)
    if child.poll() is not None:
        raise SystemExit("source-lock child exited before readiness")
    time.sleep(0.01)
child.kill()
child.wait(timeout=10)
raise SystemExit("source-lock child readiness timed out")
"""


PHYSICAL_V1_HOLDER = r"""
import fcntl
import json
import os
from pathlib import Path
import sys
import time
path = Path(sys.argv[1])
ready = Path(sys.argv[2])
release = Path(sys.argv[3])
parent_pid = int(sys.argv[4])
descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
record = {"format": "calyx-lawslice-source-scan-lock-v1", "pid": os.getpid(), "output": "legacy-holder"}
os.ftruncate(descriptor, 0)
os.write(descriptor, (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
os.fsync(descriptor)
ready.write_text(json.dumps(record) + "\n", encoding="utf-8")
try:
    while not release.exists():
        if os.getppid() != parent_pid:
            raise SystemExit("physical-v1 test parent exited before release")
        time.sleep(0.01)
finally:
    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)
"""


PHYSICAL_V1_CONTENDER = r"""
import errno
import fcntl
import os
from pathlib import Path
import sys
path = Path(sys.argv[1])
descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
try:
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError as error:
    if error.errno in (errno.EACCES, errno.EAGAIN):
        raise SystemExit(73)
    raise
raise SystemExit("physical-v1 contender unexpectedly acquired the lock")
"""


def _wait_for(path: Path, process: subprocess.Popen, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                "subprocess exited before readiness: rc=%s stdout=%r stderr=%r"
                % (process.returncode, stdout, stderr)
            )
        time.sleep(0.01)
    process.kill()
    stdout, stderr = process.communicate()
    raise AssertionError(
        "subprocess readiness timed out: stdout=%r stderr=%r" % (stdout, stderr)
    )


def _stop_holder(process: subprocess.Popen, release: Path) -> None:
    if process.poll() is not None:
        return
    if release.parent.is_dir():
        release.write_text("test cleanup\n", encoding="utf-8")
    else:
        process.terminate()
    try:
        process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=10)
        raise AssertionError("test holder ignored release and SIGTERM")


def _subprocess_environment(runtime: Path | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if runtime is not None:
        environment["XDG_RUNTIME_DIR"] = str(runtime)
    return environment


@contextmanager
def _runtime_environment(runtime: Path):
    previous = os.environ.get("XDG_RUNTIME_DIR")
    os.environ["XDG_RUNTIME_DIR"] = str(runtime)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("XDG_RUNTIME_DIR", None)
        else:
            os.environ["XDG_RUNTIME_DIR"] = previous


def _physical_sources(root: Path, suffix: str = "a") -> dict[str, str]:
    sources = {}
    for role in ("clusters", "dockets", "opinions"):
        path = root / (role + "-" + suffix + ".csv.bz2")
        if not path.exists():
            path.write_bytes((role + "-" + suffix + "-physical\n").encode("utf-8"))
        sources[role] = str(path.absolute())
    manifest = root / ("manifest-" + suffix + ".sha256")
    if not manifest.exists():
        manifest.write_text("physical manifest " + suffix + "\n", encoding="utf-8")
    sources["manifest"] = str(manifest.absolute())
    return sources


def _identity(root: Path, suffix: str = "a") -> dict:
    physical_sources = _physical_sources(root, suffix)
    sources = {
        role: {
            "path": physical_sources[role],
            "sha256": hashlib.sha256((role + suffix).encode("utf-8")).hexdigest(),
        }
        for role in ("clusters", "dockets", "opinions")
    }
    return make_source_identity(
        sources,
        selector_version="selector-v1",
        text_policy_version="text-v1",
    )


class PublisherLockTests(unittest.TestCase):
    def test_orphan_enumeration_is_exact_and_never_reuses_or_deletes_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "generation"
            first = root / ".generation.staging.first"
            second = root / ".generation.staging.second"
            unrelated = root / ".generation.staging-related"
            first.mkdir()
            unrelated.mkdir()

            with self.assertRaisesRegex(GenerationError, "staging.first"):
                GenerationPublisher(destination, "manifest.json")
            self.assertTrue(first.is_dir())
            self.assertTrue(unrelated.is_dir())
            self.assertFalse((root / ".generation.publisher-lock").exists())

            second.mkdir()
            with self.assertRaises(GenerationError) as raised:
                GenerationPublisher(destination, "manifest.json")
            self.assertIn("staging.first", str(raised.exception))
            self.assertIn("staging.second", str(raised.exception))
            self.assertNotIn("staging-related", str(raised.exception))
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

            # Explicit operator recovery is outside the publisher. Once both
            # exact orphans are relocated, the unrelated prefix is ignored.
            recovered = root / "recovered"
            recovered.mkdir()
            shutil.move(first, recovered / first.name)
            shutil.move(second, recovered / second.name)
            with GenerationPublisher(destination, "manifest.json") as publisher:
                publisher.write_json("payload.json", {"state": "physical"})
                publisher.publish({"format": "orphan-prefix-v1"})
            self.assertTrue(unrelated.is_dir())
            self.assertEqual(verify_generation(destination, "manifest.json")["format"], "orphan-prefix-v1")

    def test_real_subprocess_competition_has_exactly_one_publisher(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "generation"
            ready = root / "ready"
            release = root / "release"
            winner = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    PUBLISHER_HOLDER,
                    str(LAW_DIR),
                    str(destination),
                    str(ready),
                    str(release),
                    str(os.getpid()),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_subprocess_environment(),
            )
            self.addCleanup(_stop_holder, winner, release)
            _wait_for(ready, winner)
            lock = root / ".generation.publisher-lock"
            owner = json.loads((lock / "owner.json").read_text(encoding="utf-8"))
            self.assertEqual(owner["pid"], winner.pid)
            self.assertEqual(owner["destination"], str(destination.absolute()))
            self.assertEqual(len(list(root.glob(".generation.staging.*"))), 1)

            loser = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    PUBLISHER_COMPETITOR,
                    str(LAW_DIR),
                    str(destination),
                ],
                text=True,
                capture_output=True,
                env=_subprocess_environment(),
                timeout=10,
                check=False,
            )
            self.assertEqual(loser.returncode, 23, loser.stderr)
            self.assertIn("publisher lock already exists", loser.stderr)
            self.assertFalse(destination.exists())
            self.assertEqual(len(list(root.glob(".generation.staging.*"))), 1)

            release.write_text("publish\n", encoding="utf-8")
            stdout, stderr = winner.communicate(timeout=10)
            self.assertEqual(winner.returncode, 0, stderr)
            self.assertIn('"status": "published"', stdout)
            manifest = verify_generation(destination, "manifest.json")
            self.assertEqual(manifest["format"], "publisher-concurrency-v1")
            self.assertFalse(lock.exists())
            self.assertEqual(list(root.glob(".generation.staging.*")), [])

    def test_killed_publisher_leaves_owner_and_stage_for_explicit_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "generation"
            ready = root / "ready"
            release = root / "release"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    PUBLISHER_HOLDER,
                    str(LAW_DIR),
                    str(destination),
                    str(ready),
                    str(release),
                    str(os.getpid()),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_subprocess_environment(),
            )
            self.addCleanup(_stop_holder, process, release)
            _wait_for(ready, process)
            lock = root / ".generation.publisher-lock"
            stage = list(root.glob(".generation.staging.*"))
            self.assertEqual(len(stage), 1)
            owner_before = (lock / "owner.json").read_bytes()
            process.kill()
            process.communicate(timeout=10)

            with self.assertRaisesRegex(GenerationError, "prior publisher crashed"):
                GenerationPublisher(destination, "manifest.json")
            self.assertEqual((lock / "owner.json").read_bytes(), owner_before)
            self.assertTrue(stage[0].is_dir())
            self.assertFalse(destination.exists())


@unittest.skipUnless(os.name == "posix", "requires real POSIX flock")
class SourceScanLockTests(unittest.TestCase):
    def _runtime(self, root: Path) -> Path:
        runtime = root / "runtime"
        runtime.mkdir(mode=0o700)
        runtime.chmod(0o700)
        return runtime

    def _holder(
        self,
        root: Path,
        runtime: Path,
        identity: dict,
        physical_sources: dict[str, str],
        name: str = "owner",
    ) -> tuple[subprocess.Popen, Path, Path]:
        ready = root / (name + ".ready")
        release = root / (name + ".release")
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                SOURCE_LOCK_HOLDER,
                str(LAW_DIR),
                json.dumps(identity, sort_keys=True),
                json.dumps(physical_sources, sort_keys=True),
                str(root / (name + ".generation")),
                str(REPO_ROOT),
                str(runtime),
                str(ready),
                str(release),
                str(os.getpid()),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_subprocess_environment(runtime),
        )
        self.addCleanup(_stop_holder, process, release)
        _wait_for(ready, process)
        return process, ready, release

    def test_competing_extractor_fails_before_opening_source_archives(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = Path("/run/user") / str(os.getuid())
            archive_paths = {}
            manifest_lines = []
            for role in ("dockets", "clusters", "opinions"):
                path = root / (role + ".csv.bz2")
                payload = (role + "-real-source\n").encode("utf-8")
                path.write_bytes(payload)
                archive_paths[role] = path
                digest = hashlib.sha256(payload).hexdigest()
                manifest_lines.append(digest + "  " + path.name)
            manifest = root / "MANIFEST.sha256"
            manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
            args = SimpleNamespace(
                dockets=str(archive_paths["dockets"]),
                clusters=str(archive_paths["clusters"]),
                opinions=str(archive_paths["opinions"]),
                bulk_manifest=str(manifest),
                snapshot_date="2026-06-30",
                acquired_at="2026-07-13T19:55:00Z",
            )
            prepared = prepare_bulk_sources(args)
            owner, _, release = self._holder(
                root,
                runtime,
                prepared["identity"],
                prepared["physical_sources"],
            )
            command = [
                sys.executable,
                str(LAW_DIR / "extract_cuyahoga.py"),
                "build",
                "--dockets",
                str(archive_paths["dockets"]),
                "--clusters",
                str(archive_paths["clusters"]),
                "--opinions",
                str(archive_paths["opinions"]),
                "--bulk-manifest",
                str(manifest),
                "--authoritative-documents-generation",
                str(root / "must-not-open-authority"),
                "--snapshot-date",
                "2026-06-30",
                "--acquired-at",
                "2026-07-13T19:55:00Z",
                "--corrections",
                str(root / "must-not-open-corrections.json"),
                "--out",
                str(root / "competitor.generation"),
            ]
            try:
                competitor = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    env=_subprocess_environment(runtime),
                    timeout=5,
                    check=False,
                )
                self.assertEqual(competitor.returncode, 1, competitor.stderr)
                self.assertIn(
                    "canonical source archives are already owned", competitor.stderr
                )
                self.assertIn(
                    '"code": "CALYX_LAWSLICE_SOURCE_SCAN_BUSY"',
                    competitor.stderr,
                )
                self.assertIn('"authority": "physical-v1"', competitor.stderr)
                self.assertNotIn("source-hash: verifying", competitor.stderr)
                self.assertFalse((root / "competitor.generation").exists())
            finally:
                release.write_text("release\n", encoding="utf-8")
                owner.communicate(timeout=10)
            self.assertEqual(owner.returncode, 0)

    def test_established_physical_v1_lock_interoperates_in_both_directions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root)
            identity = _identity(root)
            physical_sources = _physical_sources(root)
            probe = SourceScanLock(
                identity,
                physical_sources=physical_sources,
                destination=root / "probe",
                repo_root=REPO_ROOT,
                runtime_root=runtime,
            )

            legacy_ready = root / "legacy.ready"
            legacy_release = root / "legacy.release"
            legacy_holder = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    PHYSICAL_V1_HOLDER,
                    str(probe.physical_lock_path),
                    str(legacy_ready),
                    str(legacy_release),
                    str(os.getpid()),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_subprocess_environment(runtime),
            )
            self.addCleanup(_stop_holder, legacy_holder, legacy_release)
            try:
                _wait_for(legacy_ready, legacy_holder)
                with self.assertRaises(SourceScanConflict) as raised:
                    probe.acquire()
                self.assertEqual(raised.exception.code, "CALYX_LAWSLICE_SOURCE_SCAN_BUSY")
                self.assertEqual(raised.exception.context["authority"], "physical-v1")
                self.assertEqual(
                    raised.exception.context["kernel_owner_pid"], legacy_holder.pid
                )
                self.assertTrue(
                    raised.exception.context["owner_record_matches_kernel"]
                )
                self.assertIn(
                    "confirmed obsolete owned unit", raised.exception.remediation
                )
                self.assertFalse(probe.semantic_lock_path.exists())
            finally:
                legacy_release.write_text("release\n", encoding="utf-8")
                legacy_holder.communicate(timeout=10)
            self.assertEqual(legacy_holder.returncode, 0)

            current_holder, ready, release = self._holder(
                root, runtime, identity, physical_sources, "current"
            )
            try:
                state = json.loads(ready.read_text(encoding="utf-8"))
                legacy_contender = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        PHYSICAL_V1_CONTENDER,
                        state["physical_path"],
                    ],
                    text=True,
                    capture_output=True,
                    env=_subprocess_environment(runtime),
                    timeout=10,
                    check=False,
                )
                self.assertEqual(
                    legacy_contender.returncode, 73, legacy_contender.stderr
                )
            finally:
                release.write_text("release\n", encoding="utf-8")
                current_holder.communicate(timeout=10)
            self.assertEqual(current_holder.returncode, 0)

    def test_different_identity_can_hold_a_distinct_kernel_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root)
            first, _, first_release = self._holder(
                root,
                runtime,
                _identity(root, "a"),
                _physical_sources(root, "a"),
                "first",
            )
            second, _, second_release = self._holder(
                root,
                runtime,
                _identity(root, "b"),
                _physical_sources(root, "b"),
                "second",
            )
            lock_files = sorted(runtime.glob("calyx-lawslice-source-*.lock"))
            self.assertEqual(len(lock_files), 2)
            owners = [json.loads(path.read_text(encoding="utf-8")) for path in lock_files]
            self.assertEqual({owner["pid"] for owner in owners}, {first.pid, second.pid})
            first_release.write_text("release\n", encoding="utf-8")
            second_release.write_text("release\n", encoding="utf-8")
            first.communicate(timeout=10)
            second.communicate(timeout=10)
            self.assertEqual((first.returncode, second.returncode), (0, 0))

    def test_holder_exits_and_releases_kernel_lock_when_parent_disappears(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root)
            identity = _identity(root)
            physical_sources = _physical_sources(root)
            ready = root / "orphan.ready"
            release = root / "orphan.release"
            child_pid_path = root / "orphan-child.pid"
            holder_args = [
                str(LAW_DIR),
                json.dumps(identity, sort_keys=True),
                json.dumps(physical_sources, sort_keys=True),
                str(root / "orphan.generation"),
                str(REPO_ROOT),
                str(runtime),
                str(ready),
                str(release),
            ]
            driver = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    SOURCE_LOCK_ORPHAN_DRIVER,
                    SOURCE_LOCK_HOLDER,
                    *holder_args,
                    str(child_pid_path),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_subprocess_environment(runtime),
            )
            _wait_for(child_pid_path, driver)
            child_pid = int(child_pid_path.read_text(encoding="utf-8").strip())
            stdout, stderr = driver.communicate(timeout=10)
            self.assertEqual(driver.returncode, 0, (stdout, stderr))

            deadline = time.monotonic() + 10
            child_state = None
            while time.monotonic() < deadline:
                stat_path = Path("/proc") / str(child_pid) / "stat"
                try:
                    stat_fields = stat_path.read_text(encoding="utf-8").split()
                except FileNotFoundError:
                    # Process exit removes /proc/<pid> atomically with respect to
                    # neither exists() nor a later open().  An absent stat file is
                    # the physical success state this loop is waiting to observe.
                    child_state = "absent"
                    break
                self.assertGreaterEqual(len(stat_fields), 3, stat_fields)
                child_state = stat_fields[2]
                time.sleep(0.01)
            self.assertEqual(child_state, "absent", "orphan helper did not exit")
            self.assertFalse(release.exists())

            with _runtime_environment(runtime):
                with SourceScanLock(
                    identity,
                    physical_sources=physical_sources,
                    destination=root / "recovered-after-parent-exit",
                    repo_root=REPO_ROOT,
                    runtime_root=runtime,
                ) as recovered:
                    owner = json.loads(
                        recovered.semantic_lock_path.read_text(encoding="utf-8")
                    )
                    self.assertEqual(owner["pid"], os.getpid())

    def test_killed_owner_releases_kernel_lock_and_next_owner_overwrites_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root)
            identity = _identity(root)
            physical_sources = _physical_sources(root)
            process, ready, _ = self._holder(
                root, runtime, identity, physical_sources
            )
            lock_path = Path(json.loads(ready.read_text(encoding="utf-8"))["path"])
            owner_before = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual(owner_before["pid"], process.pid)
            process.kill()
            process.communicate(timeout=10)

            with _runtime_environment(runtime):
                with SourceScanLock(
                    identity,
                    physical_sources=physical_sources,
                    destination=root / "recovered",
                    repo_root=REPO_ROOT,
                    runtime_root=runtime,
                ) as recovered:
                    owner_after = json.loads(lock_path.read_text(encoding="utf-8"))
                    self.assertEqual(owner_after["pid"], os.getpid())
                    self.assertEqual(
                        owner_after["semantic_identity_sha256"], recovered.key
                    )
                    self.assertNotEqual(owner_after["pid"], owner_before["pid"])

    def test_insecure_runtime_and_invalid_physical_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = _identity(root)
            physical_sources = _physical_sources(root)

            open_runtime = root / "open-runtime"
            open_runtime.mkdir(mode=0o755)
            open_runtime.chmod(0o755)
            with self.assertRaisesRegex(SourceScanLockError, "not a private"):
                SourceScanLock(
                    identity,
                    physical_sources=physical_sources,
                    destination=root / "open",
                    repo_root=REPO_ROOT,
                    runtime_root=open_runtime,
                )

            secure = root / "secure"
            secure.mkdir(mode=0o700)
            secure.chmod(0o700)
            link = root / "runtime-link"
            link.symlink_to(secure, target_is_directory=True)
            with self.assertRaisesRegex(SourceScanLockError, "not a private"):
                SourceScanLock(
                    identity,
                    physical_sources=physical_sources,
                    destination=root / "link",
                    repo_root=REPO_ROOT,
                    runtime_root=link,
                )

            Path(physical_sources["opinions"]).unlink()
            with self.assertRaisesRegex(SourceScanLockError, "cannot inspect") as raised:
                SourceScanLock(
                    identity,
                    physical_sources=physical_sources,
                    destination=root / "missing",
                    repo_root=REPO_ROOT,
                    runtime_root=secure,
                )
            self.assertIn("correct the reported source", raised.exception.remediation)
            self.assertIn("scan did not start", raised.exception.remediation)
            self.assertNotIn("stop only", raised.exception.remediation)
            self.assertEqual(list(secure.rglob("*.lock")), [])

    def test_symlink_lock_file_is_rejected_without_touching_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root)
            identity = _identity(root)
            physical_sources = _physical_sources(root)
            lock = SourceScanLock(
                identity,
                physical_sources=physical_sources,
                destination=root / "generation",
                repo_root=REPO_ROOT,
                runtime_root=runtime,
            )
            target = root / "protected-target"
            target.write_text("must-remain-unchanged\n", encoding="utf-8")
            lock.physical_lock_path.symlink_to(target)

            with self.assertRaises(SourceScanLockError):
                lock.acquire()

            self.assertEqual(
                target.read_text(encoding="utf-8"), "must-remain-unchanged\n"
            )
            self.assertTrue(stat.S_ISLNK(lock.physical_lock_path.lstat().st_mode))
            self.assertFalse(lock.semantic_lock_path.exists())


if __name__ == "__main__":
    unittest.main()
