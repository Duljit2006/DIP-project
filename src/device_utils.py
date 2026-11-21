"""
src/device_utils.py - Intelligent Hardware and Accelerator Selection
--------------------------------------------------------------------
Automatically selects the optimal hardware device according to priority:
  1. Dedicated GPU (NVIDIA CUDA / AMD ROCm / Intel Arc dGPU)
  2. Integrated GPU (Apple Silicon MPS / Intel Iris / AMD Radeon iGPU)
  3. CPU (Fallback if no GPU acceleration is available)
"""

import os
import sys
import platform
import subprocess
import json
from typing import Tuple, Dict, Any, List
import torch


def detect_system_hardware() -> Dict[str, Any]:
    """
    Detects physical GPUs and CPU cores installed on the operating system.
    Supports Windows (Win32_VideoController), Linux (lspci/sysfs), and macOS.
    """
    system_os = platform.system()
    dedicated_gpus = []
    integrated_gpus = []
    cpu_info = f"{platform.processor() or platform.machine()} ({os.cpu_count() or 1} threads)"

    # --- Windows Detection ---
    if system_os == "Windows":
        try:
            cmd = [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                if isinstance(data, dict):
                    data = [data]
                for gpu in data:
                    name = gpu.get("Name", "")
                    ram = gpu.get("AdapterRAM", 0) or 0
                    ram_gb = round(ram / (1024 ** 3), 1)
                    lower_name = name.lower()

                    is_igpu = any(k in lower_name for k in [
                        "radeon(tm) graphics", "intel(r) hd", "intel(r) uhd",
                        "iris", "integrated", "vega"
                    ]) or (ram_gb < 1.0 and "geforce" not in lower_name and "rtx" not in lower_name)

                    gpu_desc = f"{name} ({ram_gb} GB VRAM)" if ram_gb > 0 else name
                    if is_igpu:
                        integrated_gpus.append(gpu_desc)
                    else:
                        dedicated_gpus.append(gpu_desc)
        except Exception:
            pass

    # --- macOS Detection ---
    elif system_os == "Darwin":
        try:
            cmd = ["system_profiler", "SPDisplaysDataType", "-json"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                for item in data.get("SPDisplaysDataType", []):
                    model = item.get("sppci_model", "")
                    if "Apple" in model:
                        integrated_gpus.append(f"{model} (Apple Unified Memory)")
                    else:
                        dedicated_gpus.append(model)
        except Exception:
            if "ARM" in platform.processor() or platform.machine() == "arm64":
                integrated_gpus.append("Apple Silicon M-Series SoC GPU")

    # --- Linux Detection ---
    elif system_os == "Linux":
        try:
            res = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if "VGA" in line or "3D" in line:
                        lower_line = line.lower()
                        if any(k in lower_line for k in ["intel", "integrated", "renoir", "cezanne", "picasso"]):
                            integrated_gpus.append(line.split(":")[-1].strip())
                        else:
                            dedicated_gpus.append(line.split(":")[-1].strip())
        except Exception:
            pass

    return {
        "dedicated_gpus": dedicated_gpus,
        "integrated_gpus": integrated_gpus,
        "cpu_info": cpu_info,
        "os": system_os,
    }


def classify_cuda_device(device_idx: int) -> Tuple[str, str, float]:
    """
    Classifies a CUDA device as 'Dedicated GPU' or 'Integrated GPU'.
    Returns: (category, name, vram_gb)
    """
    name = torch.cuda.get_device_name(device_idx)
    props = torch.cuda.get_device_properties(device_idx)
    vram_gb = round(props.total_memory / (1024 ** 3), 2)
    lower_name = name.lower()

    # Known integrated graphics identifiers
    is_igpu = any(k in lower_name for k in [
        "integrated", "radeon graphics", "intel hd", "uhd", "iris", "tegra"
    ]) or (vram_gb < 1.5 and "rtx" not in lower_name and "gtx" not in lower_name)

    category = "Integrated GPU" if is_igpu else "Dedicated GPU"
    return category, name, vram_gb


def get_optimal_device(preference: str = "auto", verbose: bool = True) -> Tuple[torch.device, Dict[str, Any]]:
    """
    Selects the optimal hardware device based on the user's priority rule:
      1. Dedicated GPU (CUDA dGPU, Arc dGPU, DirectML dGPU)
      2. Integrated GPU (Apple Silicon MPS, AMD/Intel iGPU via DirectML/XPU/CUDA)
      3. CPU (Fallback if no GPU is active or supported)

    Args:
        preference (str): 'auto', 'cuda', 'mps', 'xpu', 'directml', or 'cpu'.
        verbose (bool): Whether to print the detection and selection banner.

    Returns:
        (torch.device, dict): Selected PyTorch device and metadata dict.
    """
    hardware = detect_system_hardware()
    chosen_device = None
    device_tier = "CPU"
    device_name = hardware["cpu_info"]
    device_details = ""
    cuda_needs_install = False

    # Force CPU if requested
    if preference.lower() == "cpu":
        chosen_device = torch.device("cpu")
        device_tier = "CPU"
        device_name = hardware["cpu_info"]

    else:
        # -----------------------------------------------------------------
        # Tier 1: Check Dedicated GPU
        # -----------------------------------------------------------------
        # 1A. Check PyTorch CUDA devices
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            best_dgpu_idx = None
            max_vram = -1.0
            best_igpu_idx = None

            for idx in range(torch.cuda.device_count()):
                cat, dname, vram = classify_cuda_device(idx)
                if cat == "Dedicated GPU" and vram > max_vram:
                    max_vram = vram
                    best_dgpu_idx = idx
                elif cat == "Integrated GPU" and best_igpu_idx is None:
                    best_igpu_idx = idx

            if best_dgpu_idx is not None:
                chosen_device = torch.device(f"cuda:{best_dgpu_idx}")
                device_tier = "Dedicated GPU"
                d_name = torch.cuda.get_device_name(best_dgpu_idx)
                device_name = d_name
                device_details = f"VRAM: {max_vram:.2f} GB (CUDA)"

        # 1B. Check Intel XPU (dedicated Arc GPUs)
        if chosen_device is None and hasattr(torch, "xpu") and torch.xpu.is_available():
            try:
                xpu_name = torch.xpu.get_device_name(0)
                if "arc" in xpu_name.lower():
                    chosen_device = torch.device("xpu:0")
                    device_tier = "Dedicated GPU"
                    device_name = xpu_name
                    device_details = "Intel XPU"
            except Exception:
                pass

        # 1C. Check DirectML (Windows dedicated GPU via torch_directml)
        if chosen_device is None:
            try:
                import torch_directml
                for d_idx in range(torch_directml.device_count()):
                    d_name = torch_directml.device_name(d_idx)
                    lower = d_name.lower()
                    if any(k in lower for k in ["geforce", "rtx", "gtx", "radeon rx", "arc"]):
                        chosen_device = torch_directml.device(d_idx)
                        device_tier = "Dedicated GPU"
                        device_name = d_name
                        device_details = "DirectML"
                        break
            except ImportError:
                pass

        # -----------------------------------------------------------------
        # Tier 2: Check Integrated GPU
        # -----------------------------------------------------------------
        if chosen_device is None:
            # 2A. Apple Silicon MPS (Unified SoC Integrated GPU)
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                chosen_device = torch.device("mps")
                device_tier = "Integrated GPU"
                device_name = hardware["integrated_gpus"][0] if hardware["integrated_gpus"] else "Apple Silicon GPU"
                device_details = "Apple Metal Performance Shaders (MPS)"

            # 2B. CUDA Integrated GPU (e.g. Jetson, AMD APU with ROCm)
            elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
                for idx in range(torch.cuda.device_count()):
                    cat, dname, vram = classify_cuda_device(idx)
                    if cat == "Integrated GPU":
                        chosen_device = torch.device(f"cuda:{idx}")
                        device_tier = "Integrated GPU"
                        device_name = dname
                        device_details = f"VRAM: {vram:.2f} GB (CUDA)"
                        break

            # 2C. Intel XPU Integrated GPU (Iris Xe / UHD)
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                try:
                    chosen_device = torch.device("xpu:0")
                    device_tier = "Integrated GPU"
                    device_name = torch.xpu.get_device_name(0)
                    device_details = "Intel XPU"
                except Exception:
                    pass

            # 2D. DirectML Integrated GPU (Windows AMD Radeon Graphics / Intel Iris Xe)
            if chosen_device is None:
                try:
                    import torch_directml
                    if torch_directml.device_count() > 0:
                        chosen_device = torch_directml.device(0)
                        device_tier = "Integrated GPU"
                        device_name = torch_directml.device_name(0)
                        device_details = "DirectML"
                except ImportError:
                    pass

        # -----------------------------------------------------------------
        # Tier 3: CPU Fallback
        # -----------------------------------------------------------------
        if chosen_device is None:
            chosen_device = torch.device("cpu")
            device_tier = "CPU"
            device_name = hardware["cpu_info"]
            device_details = "Multi-threaded CPU"
            if hardware["dedicated_gpus"] or hardware["integrated_gpus"]:
                cuda_needs_install = True

    # Build metadata summary
    summary = {
        "device": chosen_device,
        "tier": device_tier,
        "device_name": device_name,
        "details": device_details,
        "hardware": hardware,
        "cuda_needs_install": cuda_needs_install,
    }

    if verbose:
        print("=" * 72)
        print("  AUTOMATIC HARDWARE & ACCELERATION CONFIGURATION")
        print("=" * 72)
        if hardware["dedicated_gpus"]:
            print(f"  * Physical Dedicated GPU : {', '.join(hardware['dedicated_gpus'])}")
        if hardware["integrated_gpus"]:
            print(f"  * Physical Integrated GPU: {', '.join(hardware['integrated_gpus'])}")
        print(f"  * Physical CPU Processor : {hardware['cpu_info']}")
        print("-" * 72)
        det_str = f" ({device_details})" if device_details else ""
        print(f"  -> ACTIVE RUNTIME TARGET : [{device_tier.upper()}] {device_name}{det_str}")
        print(f"  -> PyTorch Device String : {chosen_device}")

        if cuda_needs_install and hardware["dedicated_gpus"]:
            print("-" * 72)
            print("  [NOTE] A physical Dedicated GPU was detected on this system,")
            print("         but current PyTorch is CPU-only.")
            print("         To enable Dedicated GPU acceleration, run:")
            print("         uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        print("=" * 72 + "\n")

    return chosen_device, summary


if __name__ == "__main__":
    dev, info = get_optimal_device(verbose=True)
    print(f"Selected device object: {dev}")
