# AMD AI Developer Program — Publication-Grade Research Workspace

**Host Node:** `GOLDENRATIO1618` (Windows 11 Pro 64-bit Build `26300.9974`, AMD Ryzen 5 8600G Hawk Point XDNA 1 NPU, Radeon 760M RDNA 3 iGPU, NVIDIA RTX 3050 6 GB dGPU, WSL2 `Ubuntu`)  
**Workspace Root (SSOT):** `C:\Antigravity\AMD AI Developer Program`  
**Junction Alias:** `C:\Antigravity\AMD_AI_Developer_Program` *(pointing directly to the same physical workspace)*

---

## Centralized Architecture

| Directory | Purpose |
|---|---|
| `00_RESEARCH_LIBRARY/` | **Single Source of Truth** untuk riset & publikasi: `RESEARCH_CHARTER.md`, `MASTER_EXPERIMENT_LOG.jsonl` (append-only SHA-256 audit log), `MASTER_BENCHMARK.csv`, `ENVIRONMENT_SNAPSHOTS/`, dan `REFERENCES_BIBTEX.bib`. |
| `01_LOCAL_RYZEN_AI_1_9/` | Benchmark lokal **Ryzen AI 1.9.0** (`VitisAIExecutionProvider` `'target': 'X1'`), benchmark komparatif **DirectML** Radeon 760M vs NPU vs CPU (`directml_benchmarks/`), serta arsip installer terverifikasi (`installers_archive/`). |
| `02_ACADEMY_CURRICULUM/` | Pemetaan **26 kursus AMD AI Academy** (`01_beginner_basics/`, `02_intermediate_systems/`, `03_advanced_rdna_hip/`) dan `COURSE_PROGRESS_LEDGER.md`. |
| `03_ROCM_CERTIFIED_ASSOCIATE/` | Modul sertifikasi **ROCm Certified Associate** (`module_01_orientation_setup` s.d. `module_05_profiling_optimization`), audit kesiapan (`check_rocm_lab_readiness.ps1`), dan bukti log lab (`docker_lab_logs/`). |
| `04_CLOUD_AND_API_LABS/` | Klien terukur **AMD Token Lab** (`token_lab/`, `.venv` terisolasi, `.env` ter-ignore) dan **AMD Developer Cloud MI300X** (`mi300x_dev_cloud/`: `setup_env.sh`, `smoke_test_rocm.py`, `auto_shutdown_guard.sh`). |
| `05_PUBLICATION_DRAFTS/` | Generator grafik vektor/PNG (`figures/generate_figures.py`), tabel LaTeX `booktabs` otomatis (`tables/benchmark_summary.tex`), dan draf naskah (`manuscripts/PAPER_OUTLINE_IEEE_ROCM.md`). |
| `scripts/` | Mesin pencatat eksperimen (`log_experiment.py`) dan pengambil snapshot sistem (`snapshot_env.ps1`). |

---

## Quick-Start Verification & Research Commands

- **Capture Hardware & Environment Snapshot:**
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\snapshot_env.ps1"
  ```
- **Local Edge Head-to-Head Benchmark (XDNA 1 NPU vs Radeon 760M DirectML vs Zen 4 CPU):**
  ```powershell
  cmd /c "`"C:\Program Files\RyzenAI\ryzen-ai-1.9.0\Scripts\activate.bat`" && python .\01_LOCAL_RYZEN_AI_1_9\directml_benchmarks\benchmark_directml_vs_npu.py"
  ```
- **Local NPU Diagnostic & Latency Verification (`Ryzen AI 1.9.0`):**
  ```powershell
  cmd /c "`"C:\Program Files\RyzenAI\ryzen-ai-1.9.0\Scripts\activate.bat`" && python .\01_LOCAL_RYZEN_AI_1_9\npu_benchmarks\verify_npu_ryzenai.py"
  ```
- **AMD Token Lab Cloud API Benchmark:**
  ```powershell
  & ".\04_CLOUD_AND_API_LABS\token_lab\.venv\Scripts\python.exe" ".\04_CLOUD_AND_API_LABS\token_lab\test_tokenlab.py"
  ```
- **ROCm Certified Associate Readiness Audit:**
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File ".\03_ROCM_CERTIFIED_ASSOCIATE\check_rocm_lab_readiness.ps1"
  ```
- **Inspect Centralized Benchmark Summary & Export Publication Tables/Figures:**
  ```powershell
  python .\scripts\log_experiment.py summary
  python .\scripts\log_experiment.py export-latex
  python .\05_PUBLICATION_DRAFTS\figures\generate_figures.py
  ```

---

## Isolation & Safety Rules
1. **NVIDIA RTX 3050 & Multi-Monitor Preservation:** Ryzen AI `1.9.0` berjalan di environment terisolasi (`C:\Program Files\RyzenAI\ryzen-ai-1.9.0\`), terpisah dari `.venv` Token Lab dan driver CUDA.
2. **WSL2 Kernel Safety:** Paket `.deb` Linux XRT (`amdxdna`) diarsipkan di `01_LOCAL_RYZEN_AI_1_9\installers_archive\linux_xrt\` untuk Ubuntu 24.04 bare-metal saja karena WSL2 tidak mengekspos `/dev/accel`.
3. **Zero Hardcoded Secrets & Credit Protection:** `.env` diabaikan Git; instance MI300X (`$1.99/hr`, pagu `$100`) wajib menjalankan `auto_shutdown_guard.sh`.
