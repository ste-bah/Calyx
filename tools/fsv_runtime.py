#!/usr/bin/env python3
"""Publish and independently verify an exact FSV runtime generation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import traceback


LAWSLICE = Path(__file__).resolve().parent / "lawslice"
sys.path.insert(0, str(LAWSLICE))
from law_generation import (  # noqa: E402
    GenerationPublisher,
    sha256_file,
    verify_generation,
)


FORMAT = "calyx-fsv-runtime-generation-v1"
MANIFEST_FILE = "runtime_manifest.json"
BINARY_FILE = "calyx"
BUILD_INFO_FILE = "build-info.json"
DEPENDENCIES_FILE = "dependencies.json"
LDD_FILE = "ldd.log"
TOOLCHAIN_FILE = "toolchain.json"
BUILD_STDOUT_FILE = "build.action.stdout"
BUILD_STDERR_FILE = "build.action.stderr"
BUILD_RC_FILE = "build.action.rc"
MODE = 0o555
REQUIRED_MEMBERS = {
    BINARY_FILE,
    BUILD_INFO_FILE,
    DEPENDENCIES_FILE,
    LDD_FILE,
    TOOLCHAIN_FILE,
    BUILD_STDOUT_FILE,
    BUILD_STDERR_FILE,
    BUILD_RC_FILE,
}


class RuntimeGenerationError(RuntimeError):
    pass


def fail(message: str, **context) -> None:
    if context:
        message = "%s | %s" % (message, json.dumps(context, sort_keys=True))
    raise RuntimeGenerationError(message)


def run(command: list[str], label: str) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        fail(
            "%s failed" % label,
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    return result.stdout


def read_json_bytes(path: Path, label: str):
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        fail("cannot read %s" % label, path=str(path), error=str(error))
    return value


def semantic_build_info(binary: Path) -> dict:
    try:
        value = json.loads(run([str(binary), "build-info"], "runtime build-info"))
    except json.JSONDecodeError as error:
        fail("runtime build-info is not JSON", binary=str(binary), error=str(error))
    if not isinstance(value, dict):
        fail("runtime build-info must be an object", binary=str(binary))
    required = {
        "binary",
        "package",
        "package_version",
        "git_sha",
        "git_dirty",
        "git_commit_unix_secs",
        "features",
        "capabilities",
        "executable",
    }
    if set(value) != required:
        fail(
            "runtime build-info field contract mismatch",
            expected=sorted(required),
            actual=sorted(value),
        )
    if value["git_dirty"] is not False:
        fail("runtime was compiled from a dirty checkout", git_dirty=value["git_dirty"])
    executable = Path(value.pop("executable")).resolve()
    if executable != binary.resolve():
        fail(
            "runtime build-info executable differs from invoked binary",
            invoked=str(binary.resolve()),
            reported=str(executable),
        )
    return value


def ldd_state(binary: Path) -> tuple[str, list[dict[str, str]]]:
    if not sys.platform.startswith("linux"):
        fail("runtime generation requires Linux ldd", platform=sys.platform)
    output = run(["ldd", str(binary)], "runtime dynamic dependency inspection")
    dependencies = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        before_address = line.split(" (0x", 1)[0].strip()
        if "=>" in before_address:
            name, target = (part.strip() for part in before_address.split("=>", 1))
            if target == "not found":
                fail("runtime dynamic dependency is missing", dependency=name)
            dependencies.append({"name": name, "path": target})
        else:
            dependencies.append(
                {"name": Path(before_address).name, "path": before_address}
            )
    if not dependencies:
        fail("ldd returned no dynamic dependencies", binary=str(binary))
    dependencies.sort(key=lambda row: (row["name"], row["path"]))
    return output, dependencies


def git_state(repo: Path) -> dict:
    if not repo.is_dir():
        fail("source repository is not a directory", repo=str(repo))
    sha = run(["git", "-C", str(repo), "rev-parse", "HEAD"], "source git SHA").strip()
    dirty = run(
        ["git", "-C", str(repo), "status", "--porcelain"], "source git status"
    )
    if dirty:
        fail("source repository is dirty", repo=str(repo), status=dirty.splitlines())
    commit_unix = run(
        ["git", "-C", str(repo), "show", "-s", "--format=%ct", "HEAD"],
        "source commit timestamp",
    ).strip()
    if not commit_unix.isdigit():
        fail("source commit timestamp is not an integer", value=commit_unix)
    return {"sha": sha, "dirty": False, "commit_unix_secs": int(commit_unix)}


def toolchain_state(build_info: dict) -> dict:
    value = {
        "rustc": run(["rustc", "-Vv"], "rustc identity").strip(),
        "cargo": run(["cargo", "-Vv"], "cargo identity").strip(),
    }
    if "cuda" in build_info["features"]:
        value["nvcc"] = run(["nvcc", "--version"], "CUDA compiler identity").strip()
    return value


def copy_member(generation: GenerationPublisher, name: str, source: Path) -> None:
    if not source.is_file() or source.is_symlink():
        fail("evidence member is not a plain file", member=name, source=str(source))
    output = generation.open_binary(name)
    with open(source, "rb") as incoming:
        shutil.copyfileobj(incoming, output, length=1 << 20)
    output.flush()
    os.fsync(output.fileno())
    output.close()


def build(args) -> dict:
    binary = Path(args.binary).absolute()
    repo = Path(args.repo).absolute()
    stdout = Path(args.build_stdout).absolute()
    stderr = Path(args.build_stderr).absolute()
    rc_path = Path(args.build_rc).absolute()
    if not binary.is_file() or binary.is_symlink():
        fail("runtime binary is not a plain file", binary=str(binary))
    rc = rc_path.read_text(encoding="utf-8").strip()
    if rc != "0":
        fail("runtime build did not succeed", build_rc=rc, path=str(rc_path))
    source = git_state(repo)

    with GenerationPublisher(args.out, MANIFEST_FILE) as generation:
        copy_member(generation, BINARY_FILE, binary)
        staged_binary = generation.path(BINARY_FILE)
        os.chmod(staged_binary, MODE)
        build_info = semantic_build_info(staged_binary)
        if build_info["git_sha"] != source["sha"]:
            fail(
                "runtime commit differs from clean source checkout",
                runtime=build_info["git_sha"],
                source=source["sha"],
            )
        if build_info["git_commit_unix_secs"] != source["commit_unix_secs"]:
            fail(
                "runtime commit timestamp differs from source checkout",
                runtime=build_info["git_commit_unix_secs"],
                source=source["commit_unix_secs"],
            )
        ldd_output, dependencies = ldd_state(staged_binary)
        toolchain = toolchain_state(build_info)
        generation.write_json(BUILD_INFO_FILE, build_info)
        generation.write_json(DEPENDENCIES_FILE, dependencies)
        generation.open_text(LDD_FILE).write(ldd_output)
        generation.write_json(TOOLCHAIN_FILE, toolchain)
        copy_member(generation, BUILD_STDOUT_FILE, stdout)
        copy_member(generation, BUILD_STDERR_FILE, stderr)
        copy_member(generation, BUILD_RC_FILE, rc_path)
        generation.publish(
            {
                "format": FORMAT,
                "source_of_truth": MANIFEST_FILE,
                "binary_member": BINARY_FILE,
                "binary_mode": format(MODE, "04o"),
                "build_info": build_info,
                "dependencies": dependencies,
                "source": source,
                "source_repository": str(repo),
                "input_binary": str(binary),
                "build_rc": 0,
            }
        )
    return verify_runtime_generation(args.out)


def verify_runtime_generation(path: str) -> dict:
    root = Path(path).absolute()
    manifest = verify_generation(root, MANIFEST_FILE)
    if manifest.get("format") != FORMAT:
        fail("runtime generation format mismatch", actual=manifest.get("format"))
    if set(manifest["files"]) != REQUIRED_MEMBERS:
        fail(
            "runtime generation member contract mismatch",
            expected=sorted(REQUIRED_MEMBERS),
            actual=sorted(manifest["files"]),
        )
    binary = root / BINARY_FILE
    mode = stat.S_IMODE(binary.stat().st_mode)
    if mode != MODE:
        fail(
            "sealed runtime mode mismatch",
            expected=format(MODE, "04o"),
            actual=format(mode, "04o"),
        )
    build_info = semantic_build_info(binary)
    if build_info != manifest.get("build_info"):
        fail("sealed runtime build-info differs from manifest")
    if build_info != read_json_bytes(root / BUILD_INFO_FILE, "sealed build-info"):
        fail("sealed runtime build-info differs from member readback")
    _, dependencies = ldd_state(binary)
    if dependencies != manifest.get("dependencies"):
        fail("sealed runtime dependency closure differs from manifest")
    if dependencies != read_json_bytes(root / DEPENDENCIES_FILE, "dependencies"):
        fail("sealed runtime dependency closure differs from member readback")
    if (root / BUILD_RC_FILE).read_text(encoding="utf-8").strip() != "0":
        fail("sealed runtime build RC is not zero")
    source = manifest.get("source")
    if not isinstance(source, dict) or build_info["git_sha"] != source.get("sha"):
        fail("sealed runtime source binding mismatch")
    return {
        "generation": str(root),
        "status": "verified",
        "git_sha": build_info["git_sha"],
        "binary_bytes": binary.stat().st_size,
        "binary_sha256": sha256_file(binary),
        "manifest_sha256": sha256_file(root / MANIFEST_FILE),
        "features": build_info["features"],
        "capabilities": build_info["capabilities"],
        "dependency_count": len(dependencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_command = commands.add_parser("build")
    build_command.add_argument("--binary", required=True)
    build_command.add_argument("--repo", required=True)
    build_command.add_argument("--build-stdout", required=True)
    build_command.add_argument("--build-stderr", required=True)
    build_command.add_argument("--build-rc", required=True)
    build_command.add_argument("--out", required=True)
    verify_command = commands.add_parser("verify")
    verify_command.add_argument("--generation", required=True)
    args = parser.parse_args()
    try:
        report = (
            build(args)
            if args.command == "build"
            else verify_runtime_generation(args.generation)
        )
        print(json.dumps(report, sort_keys=True))
    except BaseException as error:
        sys.stderr.write(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            )
            + "\n"
        )
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
