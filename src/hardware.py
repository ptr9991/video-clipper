"""Detect local hardware (GPU, VRAM, RAM, CPU) for visual AI readiness."""

from __future__ import annotations

import logging
import platform
import re
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional

log = logging.getLogger("video_clipper.hardware")


@dataclass
class HardwareInfo:
    os_name: str
    cpu: str
    ram_gb: float
    gpu_name: str
    vram_gb: float
    cuda_available: bool
    nvidia: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _run(cmd: list[str], timeout: float = 8.0) -> str:
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,  # type: ignore[attr-defined]
        )
        return (r.stdout or "") + (r.stderr or "")
    except Exception as exc:
        log.debug("cmd failed %s: %s", cmd, exc)
        return ""


def _detect_ram_gb() -> float:
    try:
        if platform.system() == "Windows":
            out = _run(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"])
            nums = re.findall(r"\d+", out)
            if nums:
                return round(int(nums[-1]) / (1024**3), 1)
        elif platform.system() == "Linux":
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024**2), 1)
        elif platform.system() == "Darwin":
            out = _run(["sysctl", "-n", "hw.memsize"])
            if out.strip().isdigit():
                return round(int(out.strip()) / (1024**3), 1)
    except Exception:
        pass
    return 0.0


def _detect_cpu() -> str:
    try:
        if platform.system() == "Windows":
            out = _run(["wmic", "cpu", "get", "Name"])
            lines = [l.strip() for l in out.splitlines() if l.strip() and l.strip().lower() != "name"]
            if lines:
                return lines[0]
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            if out.strip():
                return out.strip()
    except Exception:
        pass
    return platform.processor() or "Unknown"


def _detect_nvidia() -> tuple[str, float, bool]:
    """Return (gpu_name, vram_gb, cuda_ok) via nvidia-smi."""
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out.strip():
        return "Nenhuma GPU NVIDIA detectada", 0.0, False
    # First GPU only
    line = out.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    name = parts[0] if parts else "NVIDIA GPU"
    vram = 0.0
    if len(parts) > 1:
        try:
            vram = round(float(parts[1]) / 1024.0, 1)  # MiB -> GiB approx
        except ValueError:
            try:
                vram = round(float(parts[1]), 1)
            except ValueError:
                vram = 0.0
    return name, vram, True


def detect_hardware() -> HardwareInfo:
    gpu, vram, cuda = _detect_nvidia()
    return HardwareInfo(
        os_name=f"{platform.system()} {platform.release()}",
        cpu=_detect_cpu(),
        ram_gb=_detect_ram_gb(),
        gpu_name=gpu,
        vram_gb=vram,
        cuda_available=cuda,
        nvidia=cuda,
    )
