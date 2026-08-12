"""Stage B gate: run this FIRST on the target compute machine, before
downloading any model. Records exact GPU count/model/VRAM/driver/CUDA
version, and reports them against what the LLM runtime code in
`src/classification/llm/local_client.py` assumes.

Written for (and manually verified against) an Azure Standard_NV24s_v3 VM:
2x Tesla M60, ~7.93 GiB VRAM each, driver 535.230.02, compute capability
5.2, CUDA 11.8 (torch 2.5.1+cu118). See EXPERIMENT_LOG.md, "Stage B".

**The only hard (fatal) requirement is that a CUDA GPU is visible to
torch** -- i.e. the plain FP16 `transformers` pipeline this project uses
(`float16`, `attn_implementation="eager"`, `device_map="auto"`) can run.
BF16, FlashAttention 2, and modern vLLM support are checked and reported
purely as *informational* capability notes -- their absence (expected and
confirmed on the M60) never fails this script. Do not reintroduce a hard
requirement on any of those three; the project intentionally avoids all of
them on this hardware (see local_client.py's UNSUPPORTED_ON_M60).

Usage:
    python scripts/verify_gpu.py
Writes a JSON report next to itself and prints a plain-language verdict.
Exits non-zero ONLY if no CUDA GPU is visible to torch.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPORT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "gpu_verification_report.json"

# Compute capability below which each feature is known to be unavailable.
MIN_COMPUTE_CAPABILITY = {
    "bfloat16": (8, 0),   # Ampere+
    "flash_attention_2": (8, 0),  # Ampere+
    "vllm_modern": (7, 0),  # Volta+
}


def run_nvidia_smi() -> dict:
    try:
        raw = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi command not found on this machine"}
    if raw.returncode != 0:
        return {"available": False, "error": raw.stderr.strip() or f"nvidia-smi exited {raw.returncode}"}

    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=count,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    gpus = []
    driver_version = None
    for line in query.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            _, name, mem_total, driver_version = parts
            gpus.append({"name": name, "vram_total": mem_total})

    count_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    gpu_count = len([l for l in count_result.stdout.strip().splitlines() if l.strip()])

    return {
        "available": True,
        "raw_nvidia_smi": raw.stdout,
        "gpu_count": gpu_count,
        "gpus": gpus,
        "driver_version": driver_version,
    }


def run_torch_checks() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch_installed": False}

    result = {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        result["cuda_version_torch_built_with"] = torch.version.cuda
        result["device_count"] = torch.cuda.device_count()
        devices = []
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            devices.append(
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "compute_capability": f"{major}.{minor}",
                    "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2),
                }
            )
        result["devices"] = devices
    return result


def build_verdict(torch_info: dict) -> tuple[list[str], bool]:
    """Returns (verdict lines, fp16_transformers_pipeline_ok). The second
    value is the ONLY thing that determines pass/fail -- BF16/FlashAttention2/
    vLLM lines below are informational notes, never a failure condition."""
    verdict = []
    if not torch_info.get("cuda_available"):
        verdict.append("REQUIRED [FAIL]: torch does not see a CUDA GPU. Do not proceed with model loading.")
        return verdict, False

    verdict.append(
        f"REQUIRED [OK]: {torch_info.get('device_count', 0)} CUDA GPU(s) visible to torch "
        "-- the plain FP16 transformers pipeline (float16, attn_implementation='eager', "
        "device_map='auto') this project uses can run."
    )

    for device in torch_info.get("devices", []):
        cc = tuple(int(x) for x in device["compute_capability"].split("."))
        for feature, min_cc in MIN_COMPUTE_CAPABILITY.items():
            supported = cc >= min_cc
            verdict.append(
                f"INFORMATIONAL: GPU {device['index']} ({device['name']}, compute capability "
                f"{device['compute_capability']}) {'supports' if supported else 'does NOT support'} "
                f"{feature} (needs >= {min_cc[0]}.{min_cc[1]}). Not required -- this project does not "
                "use it on this hardware regardless (see local_client.py's UNSUPPORTED_ON_M60)."
            )
    return verdict, True


def main() -> None:
    smi_info = run_nvidia_smi()
    torch_info = run_torch_checks()
    if smi_info.get("available"):
        verdict, gpu_ok = build_verdict(torch_info)
    else:
        verdict = [
            "REQUIRED [FAIL]: nvidia-smi failed or is not present on this machine. "
            "Do not proceed with model loading -- see 'error' in the report."
        ]
        gpu_ok = False

    report = {
        "nvidia_smi": smi_info,
        "torch": torch_info,
        "verdict": verdict,
        "fp16_transformers_pipeline_ok": gpu_ok,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== nvidia-smi ===")
    print(smi_info.get("raw_nvidia_smi", smi_info.get("error", "n/a")))
    print("=== torch/CUDA ===")
    print(json.dumps(torch_info, indent=2))
    print("=== Verdict ===")
    for line in verdict:
        print(f"- {line}")
    print(f"\nfp16_transformers_pipeline_ok: {gpu_ok}")
    print(f"Full report written to {REPORT_PATH}")

    if not gpu_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
