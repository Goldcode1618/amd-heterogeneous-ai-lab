# Linux NPU XRT Package Archive (`RAI_1.9_Linux_NPU_XRT.zip`)

## Package Inventory
Extracted from `RAI_1.9_Linux_NPU_XRT.zip`:
- `xrt_202620.2.26.89_24.04-amd64-base.deb`
- `xrt_202620.2.26.89_24.04-amd64-base-dev.deb`
- `xrt_202620.2.26.89_24.04-amd64-npu.deb`
- `xrt_plugin.2.26.260102.59.release_24.04-amd64-amdxdna.deb`

## Compatibility Audit: WSL2 vs. Native Ubuntu 24.04

- **Audit Date:** 2026-11-20
- **Host Node:** `GOLDENRATIO1618` (Windows 11 Pro Build `26300.9974`, Ryzen 5 8600G / Hawk Point NPU)
- **WSL2 Distro Checked:** `Ubuntu` (WSL Version 2)
- **WSL2 Kernel:** `6.18.33.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Thu Jun 18 21:54:43 UTC 2026 x86_64 GNU/Linux`
- **`/dev/accel` Status in WSL2:** **NOT PRESENT** (`ls: cannot access '/dev/accel': No such file or directory`)

## Why These `.deb` Packages Were Archived Only (Not Installed in WSL2)
1. Windows WSL2 runs inside a virtualized Hyper-V utility VM and currently does **not** expose the host PCIe AMD NPU (`PCI\VEN_1022&DEV_1502`) as `/dev/accel/accel0` inside the Linux guest.
2. `xrt_plugin.2.26.260102.59.release_24.04-amd64-amdxdna.deb` targets **native bare-metal Ubuntu 24.04 LTS** with direct PCIe access to build/load the `amdxdna` kernel module (DKMS).
3. Attempting to force-install `amdxdna` kernel modules inside WSL2 will fail or risk destabilizing the Microsoft WSL2 kernel environment.

## How to Use on Native Ubuntu 24.04 (If Dual-Booting / Bare Metal)
If deploying onto a native Ubuntu 24.04 host with an AMD Ryzen AI NPU:
```bash
sudo apt-get update
sudo apt-get install -y ./xrt_202620.2.26.89_24.04-amd64-base.deb \
                        ./xrt_202620.2.26.89_24.04-amd64-base-dev.deb \
                        ./xrt_202620.2.26.89_24.04-amd64-npu.deb \
                        ./xrt_plugin.2.26.260102.59.release_24.04-amd64-amdxdna.deb
```
Verify on native Linux with:
```bash
ls -la /dev/accel/accel0
/opt/xilinx/xrt/bin/xrt-smi examine
```
