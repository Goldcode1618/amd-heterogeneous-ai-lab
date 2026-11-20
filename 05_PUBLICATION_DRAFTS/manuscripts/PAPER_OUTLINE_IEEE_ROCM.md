# Kerangka Naskah Publikasi (IEEE / ACM & AMD ROCm Blogs)

**Judul Kerja:**  
*Unified Edge-to-Cloud AI Acceleration on the AMD Ecosystem: Empirical Characterization of XDNA 1 NPU, RDNA 3 iGPU, Token Lab API, and Instinct MI300X*

**Node Eksperimen Lokal:** `GOLDENRATIO1618` (AMD Ryzen 5 8600G, Hawk Point XDNA 1 NPU, Radeon 760M RDNA 3 iGPU, 32 GB DDR5)  
**Pustaka Sitasi:** [`REFERENCES_BIBTEX.bib`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/00_RESEARCH_LIBRARY/REFERENCES_BIBTEX.bib)  
**Data Audit Utama:** [`MASTER_EXPERIMENT_LOG.jsonl`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/00_RESEARCH_LIBRARY/MASTER_EXPERIMENT_LOG.jsonl) & [`MASTER_BENCHMARK.csv`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/00_RESEARCH_LIBRARY/MASTER_BENCHMARK.csv)

---

## Abstract (Draft Outline)
Modern AI workloads increasingly span heterogeneous compute tiers—from low-power edge NPUs and integrated GPUs on desktop/client APUs to OpenAI-compatible cloud inference endpoints and high-bandwidth datacenter accelerators. This paper presents a reproducible empirical methodology and cross-tier characterization across the AMD AI ecosystem, unifying **Ryzen AI 1.9.0** (`VitisAIExecutionProvider` on Hawk Point XDNA 1, `'target': 'X1'`), **DirectML** on Radeon 760M (RDNA 3), **Zen 4 CPU** execution, **AMD Token Lab** hosted open-weights inference (`Gemma-4-31B-IT`, `DeepSeek-V4.1-Flash`, `Qwen3.8-27B`), and **AMD Instinct MI300X** (`192 GB HBM3`, CDNA 3). All measurements are cryptographically anchored with SHA-256 script provenance and Git commit metadata.

---

## 1. Introduction & Motivation
- Mengapa evaluasi *single-hardware* tidak cukup untuk aplikasi *agentic AI* modern.
- Kontribusi utama:
  1. Metodologi *edge-to-cloud* terpusat dan reprodusibel pada satu workstation konsumen-ke-cloud AMD.
  2. Karakterisasi empiris `VitisAIExecutionProvider` (`'target': 'X1'`) vs `DmlExecutionProvider` vs `CPUExecutionProvider` pada ONNX Runtime `1.30.0` (Ryzen AI `1.9.0`).
  3. Evaluasi latensi & *throughput* hybrid routing antara NPU lokal dan AMD Token Lab / Instinct MI300X.

## 2. Experimental Apparatus & Reproducibility Protocol
- Referensi otomatis ke [`latest_snapshot.md`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/00_RESEARCH_LIBRARY/ENVIRONMENT_SNAPSHOTS/latest_snapshot.md) (OS Windows 11 Pro `26300.9974`, BIOS `1.E3`, XRT `2.21.0`, NPU Driver `32.0.20102.4480`, FW `1.5.5.391`, WSL2 `6.18.33.2-microsoft-standard-WSL2`).
- Penjelasan isolasi RTX 3050 dGPU dan skema pencatatan `scripts/log_experiment.py`.

## 3. Empirical Results
- **Tabel LaTeX Otomatis:** [`benchmark_summary.tex`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/05_PUBLICATION_DRAFTS/tables/benchmark_summary.tex)
- **Grafik Vektor Otomatis:** [`edge_vs_cloud_benchmark.svg`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/05_PUBLICATION_DRAFTS/figures/edge_vs_cloud_benchmark.svg)
- **Sub-bab Analisis:**
  - *3.1 Local Edge Compute (RQ1):* XDNA 1 NPU vs Radeon 760M DirectML vs Zen 4 CPU (`p50`, `p95`, cold-start session init).
  - *3.2 Instant Cloud API (RQ2):* AMD Token Lab wall-clock latency & `tokens/s` across open-weight models.
  - *3.3 Datacenter CDNA 3 MI300X & CUDA-to-HIP Porting (RQ3–RQ4):* BF16/FP16 GEMM TFLOPS, `vLLM` PagedAttention serving, dan hasil lab *ROCm Certified Associate*.

## 4. Threats to Validity & Engineering Lessons Learned
- Ketidakhadiran `/dev/accel` pada WSL2 (perlu bare-metal Ubuntu 24.04 untuk `amdxdna` Linux).
- Resolusi `4x4.xclbin` untuk arsitektur Phoenix/Hawk Point (`'target': 'X1'`) pada Ryzen AI `1.9.0` agar tidak jatuh ke `AMD_AIE2P_4x4_Overlay` (Strix/XDNA 2).

## 5. Conclusion & Artifact Availability
- Seluruh kode benchmark, skrip audit SHA-256, dan dataset log terpusat di [`AMD AI Developer Program`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/README.md).
