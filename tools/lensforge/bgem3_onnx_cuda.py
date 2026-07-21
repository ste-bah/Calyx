#!/usr/bin/env python3
"""Build and verify Calyx's pinned joint BGE-M3 CUDA ONNX artifact.

The official BAAI ONNX backbone exports the normalized dense vector and token
embeddings.  This builder attaches the official sparse and ColBERT projection
weights to that same graph, converts internal tensors to FP16, and publishes
one immutable external-data model plus three LensForge manifests.  The three
manifests therefore resolve to one artifact-set hash and one runtime cache key.

There is deliberately no alternate model, floating revision, CPU export, or
partial publication path.  A failed download, graph validation, weight check,
or fsync leaves no published generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Iterable


MODEL_ID = "BAAI/bge-m3"
PINNED_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MAX_TOKENS = 512
HIDDEN_DIM = 1024
VOCAB_DIM = 250_002
EXTERNAL_DATA_NAME = "model.onnx_data"
EXTERNAL_DATA_THRESHOLD = 1024
SOURCE_FILES = {
    "model.onnx": "onnx/model.onnx",
    "model.onnx_data": "onnx/model.onnx_data",
    "tokenizer.json": "onnx/tokenizer.json",
    "config.json": "onnx/config.json",
    "tokenizer_config.json": "onnx/tokenizer_config.json",
    "special_tokens_map.json": "onnx/special_tokens_map.json",
    "sparse_linear.pt": "sparse_linear.pt",
    "colbert_linear.pt": "colbert_linear.pt",
}
ARTIFACT_ROLES = (
    ("model", "model.onnx"),
    ("tokenizer", "tokenizer.json"),
    ("config", "config.json"),
    ("tokenizer_config", "tokenizer_config.json"),
    ("special_tokens_map", "special_tokens_map.json"),
    ("external_data", EXTERNAL_DATA_NAME),
)
OUTPUTS = {
    "dense": ("onnx-bgem3-dense", HIDDEN_DIM, {"kind": "dense", "dim": HIDDEN_DIM}, "l2"),
    "sparse": (
        "onnx-bgem3-sparse",
        VOCAB_DIM,
        {"kind": "sparse", "dim": VOCAB_DIM},
        "finite",
    ),
    "colbert": (
        "onnx-bgem3-colbert",
        HIDDEN_DIM,
        {"kind": "multi", "token_dim": HIDDEN_DIM},
        "finite",
    ),
}


class BuildError(RuntimeError):
    pass


def fail(message: str) -> "NoReturn":
    raise BuildError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def length_delimited_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_dir(path.parent)


def download(cache: Path, revision: str, local_name: str, remote_name: str) -> Path:
    destination = cache / revision / local_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    if destination.exists():
        fail(f"cached source is not a non-empty file: {destination}")
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{revision}/{remote_name}?download=true"
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    request = urllib.request.Request(url, headers={"User-Agent": "calyx-lensforge-bgem3/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("xb") as out:
            expected = response.headers.get("Content-Length")
            copied = 0
            while chunk := response.read(8 * 1024 * 1024):
                out.write(chunk)
                copied += len(chunk)
            out.flush()
            os.fsync(out.fileno())
        if copied == 0:
            fail(f"download returned zero bytes: {url}")
        if expected is not None and copied != int(expected):
            fail(f"download length mismatch for {url}: copied={copied} header={expected}")
        os.replace(temporary, destination)
        fsync_dir(destination.parent)
        return destination
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def require_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import onnx
        import torch
        from onnxconverter_common import float16
    except Exception as error:
        fail(
            "BGE-M3 exporter dependencies are unavailable; install "
            f"tools/lensforge/requirements-bgem3-onnx.txt exactly: {error!r}"
        )
    return np, onnx, torch, float16


def state_tensor(torch: Any, path: Path, key: str, expected: tuple[int, ...]) -> Any:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        fail(f"load pinned projection weights {path} failed: {error!r}")
    if not isinstance(state, dict) or key not in state:
        fail(f"projection file {path} does not contain tensor {key!r}")
    tensor = state[key].detach().cpu().to(dtype=torch.float32)
    if tuple(tensor.shape) != expected:
        fail(f"projection tensor {path}:{key} shape={tuple(tensor.shape)} expected={expected}")
    if not bool(torch.isfinite(tensor).all()):
        fail(f"projection tensor {path}:{key} contains NaN or Inf")
    return tensor.numpy()


def graph_value_names(graph: Any) -> set[str]:
    names = {item.name for item in graph.input}
    names.update(item.name for item in graph.output)
    names.update(item.name for item in graph.value_info)
    names.update(item.name for item in graph.initializer)
    for node in graph.node:
        names.update(node.output)
    return names


def require_value(graph: Any, name: str) -> None:
    if name not in graph_value_names(graph):
        fail(f"official ONNX backbone is missing required value {name!r}")


def convert_source_float_casts(model: Any, onnx: Any) -> int:
    """Apply microsoft/onnxconverter-common@1e4cdf1 before FP16 conversion.

    Converter 1.16 changes Cast<float32> output metadata to FP16 without
    changing the source Cast operator.  Upstream fixed this after the 1.16
    release by changing source Cast<float32> nodes before blocked-op casts are
    inserted.  Keep this exact ordering so converter-generated FP32 boundary
    casts are not touched.
    """
    pending = [model.graph]
    changed = 0
    while pending:
        graph = pending.pop()
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    pending.append(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    pending.extend(attribute.graphs)
            if node.op_type != "Cast":
                continue
            for attribute in node.attribute:
                if attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT:
                    attribute.i = onnx.TensorProto.FLOAT16
                    changed += 1
    return changed


def count_fp16_tensor_declarations(model: Any, onnx: Any) -> int:
    pending = [model.graph]
    count = 0
    while pending:
        graph = pending.pop()
        values = list(graph.input) + list(graph.output) + list(graph.value_info)
        count += sum(
            1
            for value in values
            if value.type.HasField("tensor_type")
            and value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
        )
        count += sum(
            1 for tensor in graph.initializer if tensor.data_type == onnx.TensorProto.FLOAT16
        )
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    pending.append(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    pending.extend(attribute.graphs)
                elif (
                    attribute.type == onnx.AttributeProto.TENSOR
                    and attribute.t.data_type == onnx.TensorProto.FLOAT16
                ):
                    count += 1
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    count += sum(
                        1
                        for tensor in attribute.tensors
                        if tensor.data_type == onnx.TensorProto.FLOAT16
                    )
    return count


def add_joint_heads(source_model: Path, sparse_weights: Path, colbert_weights: Path, output: Path) -> None:
    np, onnx, torch, float16 = require_dependencies()
    from onnx import TensorProto, helper, numpy_helper

    # Passing a loaded >2 GiB external-data ModelProto to the checker forces
    # protobuf serialization and fails at protobuf's hard size ceiling.  ONNX
    # explicitly requires path-based checking for such models.
    try:
        onnx.checker.check_model(str(source_model), full_check=True)
    except Exception as error:
        fail(f"path-check official external-data ONNX model failed: {error!r}")
    try:
        model = onnx.load(str(source_model), load_external_data=True)
    except Exception as error:
        fail(f"load official external-data ONNX model failed: {error!r}")
    graph = model.graph
    require_value(graph, "token_embeddings")
    require_value(graph, "sentence_embedding")
    existing = graph_value_names(graph)
    reserved = {
        "dense_vecs",
        "sparse_vecs",
        "colbert_vecs",
        "calyx_sparse_weight",
        "calyx_sparse_bias",
        "calyx_colbert_weight",
        "calyx_colbert_bias",
    }
    collision = sorted(existing.intersection(reserved))
    if collision:
        fail(f"official graph collides with Calyx BGE-M3 values: {collision}")

    sparse_weight = state_tensor(torch, sparse_weights, "weight", (1, HIDDEN_DIM)).T.copy()
    sparse_bias = state_tensor(torch, sparse_weights, "bias", (1,)).copy()
    colbert_weight = state_tensor(
        torch, colbert_weights, "weight", (HIDDEN_DIM, HIDDEN_DIM)
    ).T.copy()
    colbert_bias = state_tensor(torch, colbert_weights, "bias", (HIDDEN_DIM,)).copy()
    graph.initializer.extend(
        [
            numpy_helper.from_array(sparse_weight, "calyx_sparse_weight"),
            numpy_helper.from_array(sparse_bias, "calyx_sparse_bias"),
            numpy_helper.from_array(colbert_weight, "calyx_colbert_weight"),
            numpy_helper.from_array(colbert_bias, "calyx_colbert_bias"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "calyx_slice_starts"),
            numpy_helper.from_array(np.asarray([sys.maxsize], dtype=np.int64), "calyx_slice_ends"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "calyx_slice_axes"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "calyx_slice_steps"),
        ]
    )
    graph.node.extend(
        [
            helper.make_node("Identity", ["sentence_embedding"], ["dense_vecs"], name="CalyxDense"),
            helper.make_node(
                "MatMul",
                ["token_embeddings", "calyx_sparse_weight"],
                ["calyx_sparse_linear"],
                name="CalyxSparseMatMul",
            ),
            helper.make_node(
                "Add",
                ["calyx_sparse_linear", "calyx_sparse_bias"],
                ["calyx_sparse_biased"],
                name="CalyxSparseBias",
            ),
            helper.make_node("Relu", ["calyx_sparse_biased"], ["sparse_vecs"], name="CalyxSparseRelu"),
            helper.make_node(
                "Slice",
                [
                    "token_embeddings",
                    "calyx_slice_starts",
                    "calyx_slice_ends",
                    "calyx_slice_axes",
                    "calyx_slice_steps",
                ],
                ["calyx_colbert_tokens"],
                name="CalyxColbertSkipCls",
            ),
            helper.make_node(
                "MatMul",
                ["calyx_colbert_tokens", "calyx_colbert_weight"],
                ["calyx_colbert_linear"],
                name="CalyxColbertMatMul",
            ),
            helper.make_node(
                "Add",
                ["calyx_colbert_linear", "calyx_colbert_bias"],
                ["colbert_vecs"],
                name="CalyxColbertBias",
            ),
        ]
    )
    del graph.output[:]
    graph.output.extend(
        [
            helper.make_tensor_value_info("dense_vecs", TensorProto.FLOAT, ["batch", HIDDEN_DIM]),
            helper.make_tensor_value_info(
                "sparse_vecs", TensorProto.FLOAT, ["batch", "sequence", 1]
            ),
            helper.make_tensor_value_info(
                "colbert_vecs", TensorProto.FLOAT, ["batch", "colbert_sequence", HIDDEN_DIM]
            ),
        ]
    )
    fp32_model = output.with_name(".joint-fp32.onnx")
    fp32_data = output.with_name(".joint-fp32.onnx_data")
    inferred_model = output.with_name(".joint-fp32-inferred.onnx")
    try:
        onnx.save_model(
            model,
            str(fp32_model),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=fp32_data.name,
            size_threshold=EXTERNAL_DATA_THRESHOLD,
            convert_attribute=False,
        )
        onnx.shape_inference.infer_shapes_path(str(fp32_model), str(inferred_model))
        onnx.checker.check_model(str(inferred_model), full_check=True)
        inferred = onnx.load(str(inferred_model), load_external_data=True)
        source_fp16 = count_fp16_tensor_declarations(inferred, onnx)
        if source_fp16 != 0:
            fail(
                "pinned BGE-M3 FP32 source unexpectedly contains FP16 tensor declarations: "
                f"count={source_fp16}"
            )
        changed_casts = convert_source_float_casts(inferred, onnx)
        if changed_casts != 1:
            fail(
                "pinned BGE-M3 graph changed its source Cast<float32> contract: "
                f"converted={changed_casts} expected=1"
            )
        converted = float16.convert_float_to_float16(
            inferred,
            keep_io_types=True,
            disable_shape_infer=True,
            check_fp16_ready=False,
        )
    except Exception as error:
        fail(f"path-infer and convert joint BGE-M3 graph to FP16 failed: {error!r}")
    try:
        onnx.save_model(
            converted,
            str(output),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=EXTERNAL_DATA_NAME,
            size_threshold=EXTERNAL_DATA_THRESHOLD,
            convert_attribute=False,
        )
    except Exception as error:
        fail(f"save joint FP16 BGE-M3 external-data graph failed: {error!r}")
    try:
        onnx.checker.check_model(str(output), full_check=True)
    except Exception as error:
        fail(f"path-check saved joint FP16 BGE-M3 graph failed: {error!r}")
    finally:
        for temporary in (fp32_model, fp32_data, inferred_model):
            if temporary.exists():
                temporary.unlink()


def artifact_entries(root: Path) -> list[dict[str, Any]]:
    entries = []
    for role, name in ARTIFACT_ROLES:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"published artifact is missing or empty: {path}")
        entries.append(
            {"role": role, "path": name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
    return entries


def build_manifest(
    root: Path,
    head: str,
    name: str,
    source_sha256: dict[str, str],
    dependency_versions: dict[str, str],
) -> dict[str, Any]:
    runtime, dim, shape, norm = OUTPUTS[head]
    files = artifact_entries(root)
    paths = [root / entry["path"] for entry in files]
    return {
        "name": name,
        "modality": "text",
        "runtime": runtime,
        "dim": dim,
        "shape": shape,
        "dtype": "f16",
        "weights_sha256": files[0]["sha256"],
        "artifact_set_sha256": length_delimited_sha256(paths),
        "files": files,
        "pooling": "joint_bgem3_device_postprocess_v1",
        "norm": norm,
        "source_hf_id": MODEL_ID,
        "license": "mit",
        "non_commercial": False,
        "max_tokens": MAX_TOKENS,
        "provenance": {
            "schema": "calyx.lensforge.bgem3-onnx-cuda.v1",
            "source_revision": PINNED_REVISION,
            "source_sha256": source_sha256,
            "dependencies": dependency_versions,
            "onnxconverter_common_cast_fix":
                "microsoft/onnxconverter-common@1e4cdf197a7303b71cb7317eed483319374c2ad7",
            "builder": "tools/lensforge/bgem3_onnx_cuda.py",
            "single_backbone_forward": True,
            "device_postprocess": "forge-cuda-v1",
        },
    }


def tensor_external_locations(model: Any) -> set[str]:
    locations = set()
    for tensor in model.graph.initializer:
        for item in tensor.external_data:
            if item.key == "location":
                locations.add(item.value)
    return locations


def verify(root: Path) -> dict[str, Any]:
    _, onnx, _, _ = require_dependencies()
    required = {name for _, name in ARTIFACT_ROLES}
    required.update({f"manifest-{head}.json" for head in OUTPUTS})
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        fail(f"BGE-M3 generation is incomplete; missing={missing}")
    try:
        model = onnx.load(str(root / "model.onnx"), load_external_data=False)
        onnx.checker.check_model(str(root / "model.onnx"), full_check=False)
    except Exception as error:
        fail(f"readback ONNX validation failed: {error!r}")
    output_names = [item.name for item in model.graph.output]
    if output_names != ["dense_vecs", "sparse_vecs", "colbert_vecs"]:
        fail(f"joint graph outputs {output_names!r} do not match the frozen contract")
    locations = tensor_external_locations(model)
    if locations != {EXTERNAL_DATA_NAME}:
        fail(f"joint graph external-data locations={sorted(locations)} expected={[EXTERNAL_DATA_NAME]}")
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    for name in (
        "calyx_slice_starts",
        "calyx_slice_ends",
        "calyx_slice_axes",
        "calyx_slice_steps",
    ):
        tensor = initializers.get(name)
        if tensor is None:
            fail(f"joint graph is missing inline Slice initializer {name}")
        if tensor.external_data or not tensor.raw_data:
            fail(
                f"joint graph Slice initializer {name} must be inline for ORT session shape inference"
            )
    entries = artifact_entries(root)
    artifact_hash = length_delimited_sha256(root / entry["path"] for entry in entries)
    manifests: dict[str, Any] = {}
    for head, (runtime, dim, shape, norm) in OUTPUTS.items():
        path = root / f"manifest-{head}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "runtime": runtime,
            "dim": dim,
            "shape": shape,
            "norm": norm,
            "artifact_set_sha256": artifact_hash,
            "source_hf_id": MODEL_ID,
            "max_tokens": MAX_TOKENS,
        }
        drift = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
        if drift:
            fail(f"manifest {path} contract drift: {drift}")
        if manifest.get("files") != entries:
            fail(f"manifest {path} artifact inventory does not match physical readback")
        provenance = manifest.get("provenance", {})
        if provenance.get("source_revision") != PINNED_REVISION:
            fail(f"manifest {path} is not pinned to revision {PINNED_REVISION}")
        manifests[head] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "name": manifest.get("name"),
            "runtime": runtime,
        }
    return {
        "schema": "calyx.lensforge.bgem3-onnx-cuda.verify.v1",
        "source_of_truth": str(root.resolve()),
        "model_sha256": sha256_file(root / "model.onnx"),
        "external_data_sha256": sha256_file(root / EXTERNAL_DATA_NAME),
        "artifact_set_sha256": artifact_hash,
        "artifact_bytes": sum(entry["bytes"] for entry in entries),
        "graph_outputs": output_names,
        "external_locations": sorted(locations),
        "manifests": manifests,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.revision != PINNED_REVISION:
        fail(
            f"floating or alternate BGE-M3 revision refused: {args.revision}; expected {PINNED_REVISION}"
        )
    output = args.output.resolve()
    if output.exists():
        fail(f"immutable output already exists: {output}; choose a new generation path")
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = {
        local: download(args.cache.resolve(), args.revision, local, remote)
        for local, remote in SOURCE_FILES.items()
    }
    source_sha256 = {name: sha256_file(path) for name, path in sources.items()}
    np, onnx, torch, _ = require_dependencies()
    dependency_versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "onnx": onnx.__version__,
        "torch": torch.__version__,
    }
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        add_joint_heads(
            sources["model.onnx"],
            sources["sparse_linear.pt"],
            sources["colbert_linear.pt"],
            stage / "model.onnx",
        )
        for name in (
            "tokenizer.json",
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ):
            shutil.copyfile(sources[name], stage / name)
            fsync_file(stage / name)
        for head, name in {
            "dense": args.dense_name,
            "sparse": args.sparse_name,
            "colbert": args.colbert_name,
        }.items():
            write_json(
                stage / f"manifest-{head}.json",
                build_manifest(stage, head, name, source_sha256, dependency_versions),
            )
        prepublish = verify(stage)
        for path in stage.iterdir():
            if path.is_file():
                fsync_file(path)
        fsync_dir(stage)
        os.rename(stage, output)
        fsync_dir(output.parent)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    readback = verify(output)
    if readback["artifact_set_sha256"] != prepublish["artifact_set_sha256"]:
        fail("published BGE-M3 artifact hash changed across atomic rename")
    write_json(output / "verification.json", readback)
    persisted = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    if persisted != readback:
        fail("verification.json readback differs from the value written")
    return readback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "calyx-bgem3")
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--dense-name", default="semantic-bge-m3-dense")
    parser.add_argument("--sparse-name", default="lexical-bge-m3-sparse")
    parser.add_argument("--colbert-name", default="late-interaction-bge-m3-colbert")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = verify(args.output.resolve()) if args.verify_only else build(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "code": "CALYX_BGE_M3_ONNX_BUILD_FAILED",
                    "error": str(error),
                    "output": str(args.output),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps({"status": "ok", **report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
