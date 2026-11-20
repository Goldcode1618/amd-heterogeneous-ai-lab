# Environment Snapshot — GOLDENRATIO1618

- **Captured (UTC):** `2026-11-20T16:35:12Z`
- **Git Commit:** `e859d81`
- **JSON Artifact:** `snapshot_20261120_163512.json`
- **JSON SHA-256:** `afcda0efcc8d443c172e25d78bcfecc49582132912d14ad3b92eaad042303970`

## Host Platform
- **OS:** Microsoft Windows 11 Pro (Version 10.0.26300, Build 26300)
- **CPU:** AMD Ryzen 5 8600G w/ Radeon 760M Graphics (6C / 12T)
- **RAM:** 27.38 GB
- **BIOS:** 1.E3
- **WSL2 Ubuntu Kernel:** `6.18.33.2-microsoft-standard-WSL2`

## Display & Compute Adapters
- **NVIDIA GeForce RTX 3050** — Driver `32.0.16.2124`
- **AMD Radeon 760M Graphics** — Driver `32.0.32029.2010`

## NVIDIA dGPU Isolation Check
```text
NVIDIA GeForce RTX 3050, 621.24, 6144 MiB
```

## XRT NPU Diagnostics (`xrt-smi examine`)
```text
System Configuration
  OS Name              : Windows NT
  Release              : 26300.9974
  Machine              : x86_64
  CPU Cores            : 12
  Memory               : 28041 MB
  Distribution         : Windows 11 Pro
  Model                : MS-7E24
  BIOS Vendor          : American Megatrends International, LLC.
  BIOS Version         : 1.E3
  Processor            : AMD Ryzen 5 8600G w/ Radeon 760M Graphics      

XRT
  Version              : 2.21.0
  Branch               : HEAD
  Hash                 : 15e6319be8de1e76a6150111a3861729e988fdb5
  Hash Date            : 2026-10-03 16:38:51
  NPU Driver Version   : 32.0.20102.4480
  NPU Firmware Version : 1.5.5.391

Device(s) Present
|BDF             |Name         |
|----------------|-------------|
|[0015:00:01.1]  |NPU Phoenix  |


```

## Python Environment Inventory
- `ryzen-ai-1.9.0`: 145 packages captured
- `04_CLOUD_AND_API_LABS/token_lab/.venv`: 15 packages captured
