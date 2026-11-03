"""Compile the fixed D32 L1-flat control and treatment with cached CUDA12.8.1."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess


IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
FLAGS = [
    "-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
    "--expt-relaxed-constexpr", "-DOKC_QDIRECT=0", "-DOKC_PK2=0",
    "-DOKC_W0ENABLE=1", "-DOKC_SM90=1", "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__", "-Xptxas=-v,--warn-on-spills",
]
INSTANTIATION = "template __global__ void attn_kernel<1, true, 0>(const __grid_constant__ Params);\n"
PROBE = '''
extern "C" {
__device__ __constant__ unsigned int d32_attn_layout[] = {
  sizeof(Smem<1, true>), Cfg<1, true>::NTX, Cfg<1, true>::STAGES,
  Cfg<1, true>::PREGS, Cfg<1, true>::CREGS, Cfg<1, true>::MINB
};
}
'''


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def device_source(raw):
    marker = "template <int L>\nvoid launch("
    if raw.count(marker) != 1:
        raise ValueError("missing or ambiguous device/host boundary")
    prefix = raw.split(marker)[0]
    body = re.sub(r"^#include <(?:torch|ATen|c10)/[^>]+>\n", "", prefix, flags=re.M)
    close = "}  // namespace"
    if body.count(close) != 1:
        raise ValueError("missing or ambiguous selected-kernel namespace boundary")
    before, after = body.split(close)
    adapted = "#include <cuda_runtime.h>\n" + before + INSTANTIATION + close + after + PROBE
    return adapted, prefix


def resources(log):
    entries = {}
    chunks = re.split(r"Compiling entry function '([^']+)' for 'sm_90a'", log)
    for offset in range(1, len(chunks), 2):
        name, body = chunks[offset:offset + 2]
        if name in entries:
            raise ValueError("ambiguous repeated entry resources: " + name)
        stack = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", body)
        registers = re.search(r"Used (\d+) registers", body)
        barriers = re.search(r"(?:used )?(\d+) barriers", body, re.I)
        if stack is None or registers is None:
            raise ValueError("missing actual entry resources: " + name)
        entries[name] = dict(
            stack_bytes=int(stack[1]), spill_store_bytes=int(stack[2]),
            spill_load_bytes=int(stack[3]), registers=int(registers[1]),
            barriers=int(barriers[1]) if barriers else None,
        )
    if len(entries) != 1 or "attn_kernelILi1ELb1ELi0E" not in next(iter(entries)):
        raise ValueError("expected exactly the selected attn_kernel<1,true,0> entry")
    return entries


def layout_witness(ptx):
    declarations = re.findall(r"\.(b8|u32)\s+d32_attn_layout\[(\d+)\]\s*=\s*\{([^}]*)\}", ptx)
    if len(declarations) != 1:
        raise ValueError("missing or ambiguous emitted sizeof/Cfg witness")
    kind, declared, initializer = declarations[0]
    declared = int(declared)
    element_bytes = {"b8": 1, "u32": 4}[kind]
    if declared * element_bytes != 24:
        raise ValueError("unexpected declared sizeof/Cfg witness extent")
    values = [int(value.strip(), 0) for value in initializer.split(",") if value.strip()]
    initialized = len(values)
    if initialized > declared or any(value < 0 or value >= 1 << (8 * element_bytes) for value in values):
        raise ValueError("sizeof/Cfg witness initializer overflows declared extent or type")
    values.extend([0] * (declared - initialized))
    if kind == "b8":
        values = list(struct.unpack("<6I", bytes(values)))
    return dict(
        declared_type=kind, declared_elements=declared, element_bytes=element_bytes,
        initialized_elements=initialized,
        names=["sizeof_Smem_L1_flat", "threads", "stages", "producer_registers", "consumer_registers", "minimum_blocks"],
        values=values,
    )


def self_test():
    omitted = ".const .align 4 .b8 d32_attn_layout[24] = {0,240,0,0,0,1,0,0,3,0,0,0,56,0,0,0,200,0,0,0,2};"
    parsed = layout_witness(omitted)
    assert parsed["values"] == [61440, 256, 3, 56, 200, 2]
    assert parsed["declared_type"] == "b8" and parsed["declared_elements"] == 24
    assert parsed["initialized_elements"] == 21
    assert layout_witness(".u32 d32_attn_layout[6] = {61440,256,3,56,200};")["values"] == [61440, 256, 3, 56, 200, 0]
    for source in (
        ".b8 d32_attn_layout[24] = {" + ",".join(["0"] * 25) + "};",
        ".b8 d32_attn_layout[24] = {256};",
        ".u32 d32_attn_layout[6] = {4294967296};",
        ".u32 d32_attn_layout[7] = {};",
    ):
        try:
            layout_witness(source)
        except ValueError:
            continue
        raise AssertionError("overflow or wrong declared extent accepted")
    print("PASS sizeof/Cfg witness omitted-zero regression and overflow negatives")


def command(out, image_id, program, args):
    return [
        "docker", "run", "--rm", "--pull=never", "--network=none",
        "--platform=linux/arm64", "--env=NVIDIA_VISIBLE_DEVICES=void",
        "--volume", str(out.resolve()) + ":/w", "--entrypoint", program,
        image_id, *args,
    ]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def compile_arm(out, arm, image):
    source = out / (arm + ".cu")
    raw = source.read_bytes()
    adapted, prefix = device_source(raw.decode())
    folder = out / (arm + "-compile")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "device.cu").write_text(adapted)
    manifest = dict(
        arm=arm, source=str(source.resolve()), source_sha256=digest(raw),
        original_device_prefix_sha256=digest(prefix.encode()),
        adapted_source_sha256=digest(adapted.encode()), image=image,
        target="sm_90a", selected_instance="attn_kernel<1,true,0>", flags=FLAGS,
        adapters=["Torch/ATen/c10 includes removed from exact device prefix",
                  "cuda_runtime.h included for declarations supplied by Torch",
                  "explicit selected-instance template instantiation",
                  "constant-only sizeof/Cfg witness outside selected kernel"],
        commands=[],
        scope="CUDA12.8.1 compile only; host extension, runtime admission, CUDA numerics and performance not run",
    )
    path = folder / "manifest.json"
    try:
        calls = [
            ("cubin", "/usr/local/cuda/bin/nvcc", [*FLAGS, "-cubin", "/w/device.cu", "-o", "/w/device.cubin"]),
            ("ptx", "/usr/local/cuda/bin/nvcc", [*FLAGS, "-ptx", "/w/device.cu", "-o", "/w/device.ptx"]),
            ("sass", "/usr/local/cuda/bin/cuobjdump", ["--dump-sass", "/w/device.cubin"]),
        ]
        for label, program, args in calls:
            argv = command(folder, image["Id"], program, args)
            result = subprocess.run(argv, capture_output=True, text=True)
            log = result.stdout + result.stderr
            (folder / (label + ".log")).write_text(log)
            manifest["commands"].append(dict(stage=label, argv=argv, status=result.returncode))
            if label == "sass":
                (folder / "device.sass").write_text(result.stdout)
            write_json(path, manifest)
            result.check_returncode()
        log = (folder / "cubin.log").read_text()
        manifest["resources"] = resources(log)
        ptx = (folder / "device.ptx").read_text()
        manifest["layout_witness"] = layout_witness(ptx)
        manifest["ptx_instruction_counts"] = {
            token: ptx.count(token) for token in
            ("bar.sync", "barrier.cta", "mbarrier.init", "fence.proxy.async", "wgmma.wait_group")
        }
        manifest["files"] = {
            name: dict(bytes=(folder / name).stat().st_size, sha256=digest((folder / name).read_bytes()))
            for name in ("device.cu", "device.ptx", "device.cubin", "device.sass", "cubin.log", "ptx.log", "sass.log")
        }
        manifest["status"] = "COMPILED_NOT_TARGET_QUALIFIED"
    except Exception as error:
        manifest["status"] = "COMPILE_SCREEN_FAILED"
        manifest["error"] = repr(error)
        raise
    finally:
        write_json(path, manifest)
    return manifest


def run(out):
    for arm in ("control", "treatment"):
        if not (out / (arm + ".cu")).is_file():
            raise ValueError("generate both native sources before compiling")
    result = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, text=True, check=True)
    image = json.loads(result.stdout)[0]
    image = {key: image[key] for key in ("Id", "Architecture", "Os", "RepoTags", "RepoDigests")}
    if image["Architecture"] != "arm64" or image["Os"] != "linux":
        raise ValueError("cached compiler image is not linux/arm64")
    summary = dict(started_utc=datetime.now(timezone.utc).isoformat(), image=image, records=[])
    try:
        for arm in ("control", "treatment"):
            manifest = compile_arm(out, arm, image)
            summary["records"].append(dict(arm=arm, resources=manifest["resources"], manifest=arm + "-compile/manifest.json"))
        summary["status"] = "COMPILED_NOT_TARGET_QUALIFIED"
    except Exception as error:
        summary.update(status="COMPILE_SCREEN_FAILED", error=repr(error))
        raise
    finally:
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "compile-summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--out", type=Path)
    choice.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test() if args.self_test else run(args.out)
