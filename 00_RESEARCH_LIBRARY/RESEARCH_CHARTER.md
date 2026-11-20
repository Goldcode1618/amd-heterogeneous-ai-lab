# Piagam Riset Mandiri (Research Charter) — AMD AI Developer Program

**Node Utama:** `GOLDENRATIO1618` (AMD Ryzen 5 8600G, XDNA 1 Hawk Point NPU, Radeon 760M RDNA 3 iGPU, 32 GB DDR5)  
**Target Publikasi:** Konferensi/Jurnal Sistem AI (IEEE/ACM) & Seri Artikel Teknis *AMD ROCm Blogs*  
**Single Source of Truth (SSOT):** `C:\Antigravity\AMD AI Developer Program\00_RESEARCH_LIBRARY\` (alias junction: `C:\Antigravity\AMD_AI_Developer_Program\00_RESEARCH_LIBRARY\`)

---

## 1. Tujuan & Posisi Riset

Riset mandiri ini mengkaji karakteristik performa, efisiensi komputasi, dan portabilitas perangkat lunak pada seluruh spektrum akselerator AI AMD secara terintegrasi dari *edge* hingga *datacenter*:

1. **Local Edge NPU (`XDNA 1`):** AMD Ryzen 5 8600G (`PCI\VEN_1022&DEV_1502`, Hawk Point `HPT`) menggunakan **Ryzen AI Software 1.9.0**, `ONNX Runtime 1.30.0`, `XRT 2.21.0`, dan `VitisAIExecutionProvider` (`'target': 'X1'`).
2. **Local Edge iGPU (`RDNA 3`):** AMD Radeon 760M Graphics menggunakan `DmlExecutionProvider` (ONNX Runtime DirectML) dan evaluasi HIP RDNA.
3. **Instant Cloud Inference API (`AMD Token Lab`):** Evaluasi latensi *end-to-end*, *Time-To-First-Token (TTFT)*, dan *generation throughput* (`tokens/s`) terhadap model terbuka (`Gemma-4-31B-IT`, `DeepSeek-V4.1-Flash`, `Qwen3.8-27B`) pada endpoint `https://api.tokenlab.amd.com/v1`.
4. **Datacenter Cloud Compute (`AMD Instinct MI300X`):** Eksperimen skala penuh pada node CDNA 3 ($192\text{ GB}$ HBM3, bandwidth $5.3\text{ TB/s}$) via AMD Developer Cloud (pagu kredit $\$100.00 \approx 50.25\text{ jam}$ @ $\$1.99/\text{jam}$).

---

## 2. Pertanyaan Riset Utama (Research Questions)

- **RQ1 (Edge NPU vs. RDNA 3 iGPU vs. Zen 4 CPU Efficiency):** Bagaimana perbedaan distribusi latensi ($p_{50}$, $p_{95}$) dan throughput inferensi model kuantisasi ketika dieksekusi pada NPU XDNA 1 (`VitisAIExecutionProvider`, `'target': 'X1'`), iGPU Radeon 760M (`DmlExecutionProvider`), dan CPU Zen 4 (`CPUExecutionProvider`) pada *host* yang sama?
- **RQ2 (Edge-to-Cloud Latency & Throughput Trade-off):** Di mana batas *latency-to-capability frontier* antara inferensi edge lokal dan API cloud AMD Token Lab (`Gemma-4-31B-IT`, `DeepSeek-V4.1-Flash`, `Qwen3.8-27B`) untuk skenario *agentic AI* dan *RAG* hibrida?
- **RQ3 (CDNA 3 MI300X Arithmetic & Serving Utilization):** Berapa rasio pencapaian TFLOPS BF16/FP16 GEMM serta efisiensi memori `vLLM` (FP8/BF16, *PagedAttention*) pada Instinct MI300X ($192\text{ GB}$ HBM3)?
- **RQ4 (CUDA-to-HIP Portability & Profiling Discipline):** Sejauh mana *toolchain* ROCm (`hipify`, `hipcc`, `rocprofv2`, `ROCgdb`) menjaga portabilitas kode CUDA tanpa penurunan kinerja signifikan pada arsitektur RDNA 3 dan CDNA 3?

---

## 3. Protokol Reprodusibilitas & Audit Publikasi

Setiap eksperimen yang berpotensi dikutip dalam naskah publikasi **WAJIB** mematuhi aturan audit berikut:
1. **Pencatatan Otomatis Terpusat:** Semua skrip uji wajib menulis hasil ke [`MASTER_EXPERIMENT_LOG.jsonl`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/00_RESEARCH_LIBRARY/MASTER_EXPERIMENT_LOG.jsonl) (audit trail *append-only* lengkap) dan [`MASTER_BENCHMARK.csv`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/00_RESEARCH_LIBRARY/MASTER_BENCHMARK.csv) melalui [`scripts/log_experiment.py`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/scripts/log_experiment.py).
2. **Kriptografi & Provenance:** Setiap baris eksperimen wajib menyertakan `experiment_id` (`UUIDv4`), `timestamp_utc` (`ISO-8601`), `git_commit`, serta `script_sha256`.
3. **Environment Snapshot Berkala:** Sebelum menjalankan sesi benchmark besar atau pembaruan driver/SDK, jalankan [`scripts/snapshot_env.ps1`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/scripts/snapshot_env.ps1) untuk mengarsipkan versi OS, BIOS, XRT, driver NPU, dan daftar paket Python ke [`ENVIRONMENT_SNAPSHOTS/`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/00_RESEARCH_LIBRARY/ENVIRONMENT_SNAPSHOTS/).
4. **Isolasi Lingkungan & Keamanan:**
   - NVIDIA GeForce RTX 3050 tetap terisolasi dan tidak dicampur dengan environment Ryzen AI `1.9.0`.
   - Kredensial API (`token_lab\.env`) dan artefak installer biner tidak pernah di-commit ke Git.
5. **Disiplin Kredit Cloud MI300X:** Setiap sesi pada MI300X wajib menjalankan `auto_shutdown_guard.sh` agar kredit $\$100$ tidak terkuras oleh instance *idle*.
