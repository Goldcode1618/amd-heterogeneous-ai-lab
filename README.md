# AMD AI Developer Program — 72-Hour Sprint Workspace

**Host Node:** `GOLDENRATIO1618` (Windows 11 Pro 64-bit, AMD Ryzen 5 8600G Hawk Point NPU XDNA 1, Radeon 760M iGPU, NVIDIA RTX 3050 6 GB dGPU, WSL2 `Ubuntu`)  
**Workspace Root:** `C:\Antigravity\AMD AI Developer Program`

## Directory Architecture

| Path | Purpose |
|---|---|
| `01_token_lab/` | OpenAI-compatible Python client for AMD Token Lab (`https://api.tokenlab.amd.com/v1`, e.g., `Gemma-4-31B-IT`) with `.env.example` secret hygiene. |
| `02_ryzen_ai_npu_local/` | Elevated MSI installer script for Ryzen AI 1.9.0 and Hawk Point (`HPT` / `'target': 'X1'`) NPU diagnostic verification (`verify_npu_ryzenai.py`). |
| `03_rocm_cert_labs/` | ROCm Certified Associate (Modules 1–5) lab readiness script (Docker + WSL2 + NVIDIA isolation check) and structured module lab notes. |
| `04_dev_cloud_mi300x/` | AMD Developer Cloud MI300X provisioning (`setup_env.sh`) and ROCm/PyTorch smoke test + BF16/FP16 TFLOPS benchmark (`smoke_test_rocm.py`) optimized for the \$100 credit window (~50.25 hrs @ \$1.99/hr). |
| `packages/nuget/` | Local NuGet feed containing `RyzenAI_Deployment.1.9.0.nupkg` and `nuget.config`. |
| `packages/linux_xrt/` | Archived `RAI_1.9_Linux_NPU_XRT.zip` + extracted Ubuntu 24.04 `.deb` packages and WSL2 compatibility notes. |

## Isolation & Safety Rules
1. **NVIDIA RTX 3050 & Multi-Monitor Preservation:** Ryzen AI 1.9.0 runs in its own isolated environment (`C:\Program Files\RyzenAI\ryzen-ai-1.9.0\`), separate from global Python and CUDA/PyTorch.
2. **WSL2 Kernel Safety:** Linux XRT `.deb` packages (`amdxdna`) target bare-metal Ubuntu 24.04 and are archived only because WSL2 does not expose `/dev/accel`.
3. **Zero Hardcoded Secrets:** Copy `01_token_lab\.env.example` to `01_token_lab\.env` and store your key locally (excluded by `.gitignore`).

## Quick-Start Verification Commands

- **Token Lab (`01_token_lab`):**
  ```powershell
  & ".\01_token_lab\.venv\Scripts\python.exe" ".\01_token_lab\test_tokenlab.py"
  ```
- **Local Hawk Point NPU (`02_ryzen_ai_npu_local`):**
  ```powershell
  cmd /c "\"C:\Program Files\RyzenAI\ryzen-ai-1.9.0\Scripts\activate.bat\" && python .\02_ryzen_ai_npu_local\verify_npu_ryzenai.py"
  ```
- **ROCm Cert Labs Readiness (`03_rocm_cert_labs`):**
  ```powershell
  & ".\03_rocm_cert_labs\check_rocm_lab_readiness.ps1"
  ```
- **AMD Developer Cloud MI300X (`04_dev_cloud_mi300x`):**
  - Register your public key from `~/.ssh/id_ed25519_amd_mi300x.pub` in the AMD Developer Cloud portal and update `HostName` in `~/.ssh/config` (`Host amd-mi300x`).
  - Upload `setup_env.sh` and `smoke_test_rocm.py`, then run `bash setup_env.sh` remotely.
