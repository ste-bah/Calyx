"""Audited termination signals for long-running law generation commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import threading

from structured_error import error_envelope


SCHEMA = "calyx-lawslice-external-signal-v2"
ERROR_CODE = "CALYX_LAWSLICE_EXTERNAL_SIGNAL"
_WATCHED_NAMES = ("SIGINT", "SIGTERM")
_RELAY_NAME = "SIGUSR1"
_ACTIVE_AUDIT: SignalAudit | None = None


class ExternalSignal(BaseException):
    """An audited external termination request delivered to the main thread."""

    def __init__(self, record: dict):
        self.record = record
        super().__init__(
            "%s from pid=%s uid=%s during %s at rows_checkpoint=%s"
            % (
                record["signal_name"],
                record["sender_pid"],
                record["sender_uid"],
                record["phase"],
                record["rows_checkpoint"],
            )
        )


class SignalAudit:
    """Route SIGINT/SIGTERM through sigwaitinfo so sender identity is retained."""

    def __init__(self):
        self.process_id = os.getpid()
        self.main_thread_id = threading.get_ident()
        self.phase = "command_start"
        self.rows_checkpoint = 0
        self.record: dict | None = None
        self._watched = {
            getattr(signal, name) for name in _WATCHED_NAMES if hasattr(signal, name)
        }

    def update_progress(self, phase: str, rows_checkpoint: int) -> None:
        self.phase = phase
        self.rows_checkpoint = rows_checkpoint

    def install(self) -> None:
        if not self._watched:
            raise RuntimeError("this platform exposes no SIGINT/SIGTERM signals")
        if _supports_siginfo_wait():
            self._install_posix_siginfo_wait()
        else:
            for signum in self._watched:
                signal.signal(signum, self._basic_handler)

    def _install_posix_siginfo_wait(self) -> None:
        relay = getattr(signal, _RELAY_NAME)
        existing = signal.getsignal(relay)
        if existing not in (signal.SIG_DFL, signal.SIG_IGN):
            raise RuntimeError("%s already has an application handler" % _RELAY_NAME)
        signal.signal(relay, self._relay_handler)
        signal.pthread_sigmask(signal.SIG_BLOCK, self._watched)
        threading.Thread(
            target=self._wait_for_signal,
            name="lawslice-signal-audit",
            daemon=True,
        ).start()

    def _wait_for_signal(self) -> None:
        info = signal.sigwaitinfo(self._watched)
        sender_pid = _optional_int(info, "si_pid")
        sender_process = _snapshot_sender_process(sender_pid)
        self.record = self._record(
            info.si_signo,
            sender_pid=sender_pid,
            sender_uid=_optional_int(info, "si_uid"),
            signal_code=_optional_int(info, "si_code"),
            sender_identity_available=True,
            sender_process=sender_process,
        )
        self._emit(self.record)
        signal.pthread_kill(self.main_thread_id, getattr(signal, _RELAY_NAME))

    def _basic_handler(self, signum, _frame) -> None:
        self.record = self._record(
            signum,
            sender_pid=None,
            sender_uid=None,
            signal_code=None,
            sender_identity_available=False,
            sender_process={
                "available": False,
                "reason": "platform signal handler exposes no sender PID",
            },
        )
        self._emit(self.record)
        raise ExternalSignal(self.record)

    def _relay_handler(self, _signum, _frame) -> None:
        if self.record is None:
            raise RuntimeError("signal relay ran without captured siginfo")
        raise ExternalSignal(self.record)

    def _record(
        self,
        signum: int,
        *,
        sender_pid: int | None,
        sender_uid: int | None,
        signal_code: int | None,
        sender_identity_available: bool,
        sender_process: dict,
    ) -> dict:
        return {
            "schema": SCHEMA,
            "code": ERROR_CODE,
            "process_id": self.process_id,
            "signal_number": int(signum),
            "signal_name": signal.Signals(signum).name,
            "signal_code": signal_code,
            "sender_pid": sender_pid,
            "sender_uid": sender_uid,
            "sender_identity_available": sender_identity_available,
            "sender_process": sender_process,
            "phase": self.phase,
            "rows_checkpoint": self.rows_checkpoint,
        }

    @staticmethod
    def _emit(record: dict) -> None:
        refusal = error_envelope(
            ExternalSignal(record),
            code=ERROR_CODE,
            remediation=(
                "identify and correct the recorded sender PID and UID before rerunning "
                "the complete bound generation at a new destination"
            ),
            context=record,
        )
        sys.stderr.write(json.dumps(refusal, ensure_ascii=False, sort_keys=True) + "\n")
        sys.stderr.flush()


def install_signal_audit() -> SignalAudit:
    global _ACTIVE_AUDIT
    if _ACTIVE_AUDIT is not None:
        raise RuntimeError("law-slice signal audit is already installed")
    audit = SignalAudit()
    audit.install()
    _ACTIVE_AUDIT = audit
    return audit


def update_signal_progress(phase: str, rows_checkpoint: int) -> None:
    if _ACTIVE_AUDIT is not None:
        _ACTIVE_AUDIT.update_progress(phase, rows_checkpoint)


def _supports_siginfo_wait() -> bool:
    return (
        os.name == "posix"
        and hasattr(signal, "pthread_sigmask")
        and hasattr(signal, "sigwaitinfo")
        and hasattr(signal, _RELAY_NAME)
    )


def _optional_int(value, name: str) -> int | None:
    actual = getattr(value, name, None)
    return int(actual) if actual is not None else None


def _process_start_ticks(path: Path) -> str:
    value = path.read_text(encoding="utf-8")
    command_end = value.rfind(")")
    if command_end < 0:
        raise ValueError("process stat record has no command terminator")
    fields_from_state = value[command_end + 1 :].split()
    if len(fields_from_state) <= 19:
        raise ValueError("process stat record has no start-time field")
    return fields_from_state[19]


def _snapshot_sender_process(sender_pid: int | None) -> dict:
    """Capture ephemeral sender identity while the signaling process exists."""
    if sender_pid is None or sender_pid <= 0:
        return {
            "available": False,
            "reason": "signal sender has no positive userspace PID",
        }
    root = Path("/proc") / str(sender_pid)
    try:
        start_before = _process_start_ticks(root / "stat")
        command_bytes = (root / "cmdline").read_bytes()
        argv = [
            part.decode("utf-8", errors="backslashreplace")
            for part in command_bytes.split(b"\0")
            if part
        ]
        executable = os.readlink(root / "exe")
        cgroup = (root / "cgroup").read_text(encoding="utf-8").splitlines()
        start_after = _process_start_ticks(root / "stat")
        if start_before != start_after:
            return {
                "available": False,
                "reason": "sender PID was reused during process snapshot",
                "pid": sender_pid,
                "start_ticks_before": start_before,
                "start_ticks_after": start_after,
            }
        return {
            "available": True,
            "pid": sender_pid,
            "start_ticks": start_before,
            "executable": executable,
            "argv": argv,
            "cgroup": cgroup,
        }
    except (OSError, ValueError) as error:
        return {
            "available": False,
            "reason": "sender exited or proc identity could not be read at receipt",
            "pid": sender_pid,
            "error_type": type(error).__name__,
            "error": str(error),
        }
