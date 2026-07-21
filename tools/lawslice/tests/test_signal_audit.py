import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest


LAW_DIR = Path(__file__).resolve().parents[1]


def process_start_ticks(pid: int) -> str:
    value = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    return value[value.rfind(")") + 1 :].split()[19]


def process_argv(pid: int) -> list[str]:
    value = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    return [part.decode("utf-8") for part in value.split(b"\0") if part]


@unittest.skipUnless(
    os.name == "posix"
    and hasattr(signal, "sigwaitinfo")
    and hasattr(signal, "pthread_sigmask"),
    "sender siginfo is a POSIX contract",
)
class SignalAuditTests(unittest.TestCase):
    def test_real_sigint_records_sender_and_aborts_unpublished_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "generation-v1"
            child = root / "signal_child.py"
            child.write_text(
                textwrap.dedent(
                    """
                    import json
                    import os
                    import signal
                    import sys

                    from law_generation import GenerationPublisher
                    from signal_audit import ExternalSignal, install_signal_audit

                    destination = sys.argv[1]
                    audit = install_signal_audit()
                    try:
                        with GenerationPublisher(destination, "manifest.json") as generation:
                            generation.write_json(
                                "real-state.json",
                                {"process_id": os.getpid(), "uid": os.getuid()},
                            )
                            audit.update_progress("real-os-signal", 500000)
                            print("READY", flush=True)
                            signal.pause()
                    except ExternalSignal as error:
                        print(json.dumps(error.record, sort_keys=True), flush=True)
                        raise SystemExit(42)
                    """
                ),
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(LAW_DIR)
            process = subprocess.Popen(
                [sys.executable, str(child), str(destination)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
            )
            self.assertEqual(process.stdout.readline().strip(), "READY")

            os.kill(process.pid, signal.SIGINT)
            try:
                stdout, stderr = process.communicate(timeout=10)
            except BaseException:
                process.kill()
                process.wait(timeout=10)
                raise

            self.assertEqual(process.returncode, 42)
            record = json.loads(stdout.strip())
            self.assertEqual(record["process_id"], process.pid)
            self.assertEqual(record["sender_pid"], os.getpid())
            self.assertEqual(record["sender_uid"], os.getuid())
            self.assertEqual(record["signal_name"], "SIGINT")
            self.assertEqual(record["phase"], "real-os-signal")
            self.assertEqual(record["rows_checkpoint"], 500000)
            self.assertTrue(record["sender_identity_available"])
            sender_process = record["sender_process"]
            self.assertTrue(sender_process["available"])
            self.assertEqual(sender_process["pid"], os.getpid())
            self.assertEqual(
                sender_process["start_ticks"], process_start_ticks(os.getpid())
            )
            self.assertEqual(sender_process["argv"], process_argv(os.getpid()))
            self.assertEqual(
                sender_process["executable"],
                os.readlink(Path("/proc") / str(os.getpid()) / "exe"),
            )
            self.assertEqual(
                sender_process["cgroup"],
                (Path("/proc") / str(os.getpid()) / "cgroup")
                .read_text(encoding="utf-8")
                .splitlines(),
            )
            self.assertIn("CALYX_LAWSLICE_EXTERNAL_SIGNAL", stderr)
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".generation-v1.staging.*")), [])


if __name__ == "__main__":
    unittest.main()
