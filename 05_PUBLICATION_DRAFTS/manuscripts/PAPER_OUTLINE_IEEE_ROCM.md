# Kerangka Naskah Publikasi (IEEE / ACM & AMD ROCm Blogs)

**Judul Kerja:**  
*Unified Edge-to-Cloud AI Acceleration on the AMD Ecosystem: Empirical Characterization of XDNA 1 NPU, RDNA 3 iGPU, Token Lab API, and Instinct MI300X*

**Node Eksperimen Lokal:** `GOLDENRATIO1618` (AMD Ryzen 5 8600G, Hawk Point XDNA 1 NPU, Radeon 760M RDNA 3 iGPU, 32 GB DDR5)  
**Pustaka Sitasi:** [`REFERENCES_BIBTEX.bib`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/00_RESEARCH_LIBRARY/REFERENCES_BIBTEX.bib)  
**Data Audit Utama:** [`MASTER_EXPERIMENT_LOG.jsonl`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/00_RESEARCH_LIBRARY/MASTER_EXPERIMENT_LOG.jsonl) & [`MASTER_BENCHMARK.csv`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/00_RESEARCH_LIBRARY/MASTER_BENCHMARK.csv)

---

## Abstract (Draft Outline)
Modern AI workloads increasingly span heterogeneous compute tiers—from low-power edge NPUs and integrated GPUs on desktop/client APUs to OpenAI-compatible cloud inference endpoints and high-bandwidth datacenter accelerators. This paper presents a reproducible empirical methodology and cross-tier characterization across the AMD AI ecosystem, unifying **Ryzen AI 1.9.0** (`VitisAIExecutionProvider` on Hawk Point XDNA 1, `'target': 'X1'`), **DirectML** on Radeon 760M (RDNA 3), **Zen 4 CPU** execution, **AMD Token Lab** hosted open-weights inference (`Gemma-4-31B-IT`, `DeepSeek-V4.1-Flash`, `Qwen3.8-27B`), and physical **AMD Instinct MI300X** (`gfx942`, `192 GB HBM3`, `304 CUs`, `torch 2.15.0+rocm7.14`) benchmarks. On physical MI300X silicon, Wave 2 empirical runs achieve **1,111.94 TFLOPS** (`FP8 float8_e4m3fnuz` via `torch._scaled_mm`), **643.67 TFLOPS** (`BF16` GEMM), **3,929.20 GB/s** HBM3 D2D copy bandwidth, and **291.45 TFLOPS** causal SDPA (`seq_len=8192`), alongside three native `gfx942` HIP C++ kernels profiled with `rocprofv3`. All measurements are cryptographically anchored with SHA-256 script provenance and Git commit metadata.

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
  - *3.3 Datacenter CDNA 3 MI300X & CUDA-to-HIP Porting (RQ3–RQ4):* Empirical Wave 1 & Wave 2 physical `AMD Instinct MI300X VF` (`gfx942`, `304 CUs`, `191.69 GiB HBM3`, `rocm:latest` + `torch 2.15.0+rocm7.14`, `HIP 7.14.60850`) measurements synchronized from `MASTER_BENCHMARK.csv` (`L16–L23`, all `PASS`):

| CSV Row | Workload / Sweep | Precision / Stack | p50 (ms) | p95 (ms) | Empirical Peak / Logged Metric | Resident Memory | Notes & Full Sweep Range |
|---|---|---|---:|---:|---:|---:|---|
| `L16` | `GEMM_8192x8192` | `FP32` (`PyTorch_ROCm_HIP_7.14.60850`) | `9.385` | `9.452` | **`117.066 TFLOPS`** | `848.00 MB` | Sweep `2K/4K/8K/16K`: `81.35 / 107.65 / 117.07 / 117.74 TFLOPS` |
| `L17` | `GEMM_8192x8192` | `BF16` (`PyTorch_ROCm_HIP_7.14.60850`) | `1.736` | `1.839` | **`626.138 TFLOPS`** | `464.00 MB` | Consistent with Wave 1 baseline (`622.41 TFLOPS`, `p50=1.772 ms`) |
| `L18` | `GEMM_16384x16384` | `BF16` (`PyTorch_ROCm_HIP_7.14.60850`) | `13.664` | `13.733` | **`643.669 TFLOPS`** | `1616.00 MB` | Sweep `2K/4K/8K/16K`: `268.14 / 573.01 / 626.14 / 643.67 TFLOPS` |
| `L19` | `GEMM_8192x8192` | `FP16` (`PyTorch_ROCm_HIP_7.14.60850`) | `1.874` | `1.932` | **`585.348 TFLOPS`** | `464.00 MB` | Wave 1 `572.84 TFLOPS`; `16K` sweep peak `601.143 TFLOPS` (`2K–16K`: `254.03 / 528.82 / 585.35 / 601.14`) |
| `L20` | `GEMM_16384x16384` | `FP8_E4M3FNUZ` (`torch._scaled_mm`) | `7.922` | `7.934` | **`1111.941 TFLOPS`** | `1104.00 MB` | Native CDNA 3 `torch.float8_e4m3fnuz`; sweep `4K/8K/16K`: `846.44 / 1086.29 / 1111.94 TFLOPS` |
| `L21` | `HBM3_D2D_Copy_4096MB` | `FP32` (`PyTorch_ROCm_HIP_7.14.60850`) | `2.186` | `2.197` | **`3929.200 GB/s`** | `8192.00 MB` | ~74.1% of 5.3 TB/s theoretical HBM3 peak; sweep `256/1024/4096 MB`: `3638.88 / 3838.01 / 3929.20 GB/s` |
| `L22` | `SDPA_Causal_4x32x8192x128` | `BF16` (`batch=4`) | `7.545` | `7.573` | **`291.450 TFLOPS`** | `1028.00 MB` | `4,343,012.9 tok/s`; sweep `seq 512..8192`: `57.50 / 134.99 / 202.94 / 255.33 / 291.45 TFLOPS` |
| `L23` | `SAXPY_FMA_4M` | `FP32` (`HIP_C++_clang24_gfx942`) | `0.660` | `0.660` | **`76.280 GB/s`** | `32.00 MB` | Track A physical HIP (`0.6598 ms` standalone; `81.33 GB/s` / `0.6188 ms` profiled via `rocprofv3`) |

    - **Mandatory Bandwidth Accounting (`saxpy_fma_kernel` vs. `HBM3_D2D_Copy_4096MB`):** When citing `saxpy_fma_kernel` (`N = 4,194,304`, `L23`), we explicitly distinguish its **`32 MiB` resident memory footprint** ($2 \times 4{,}194{,}304 \times 4\text{ B} = 33{,}554{,}432\text{ B}$ for FP32 vectors `x` and `y`) from its **`48 MiB` ($\approx 50.33\text{ MB}$) total read/write memory traffic** ($3 \times 4{,}194{,}304 \times 4\text{ B} = 50{,}331{,}648\text{ B}$ for `read x + read y + write y`), which over `0.6598 ms` yields `76.28 GB/s` (`81.33 GB/s` during warm `rocprofv3` dispatch). Unlike `HBM3_D2D_Copy_4096MB` (`3,929.20 GB/s`), small-$N$ single-launch SAXPY is dominated by kernel launch/warmup latency and severely under-saturates MI300X's 304 CUs and 8 HBM3 stacks.
    - **Additional Track A Native HIP (`gfx942`) & `rocprofv3` Kernels + Track C `vLLM` Probe:** `wavefront_reduction.hip.cpp` confirmed CDNA 3 `WavefrontSize=64` with exact reduction (`Reduced Sum=1048576.0`, `expected=1048576`), and `tiled_gemm_lds.hip.cpp` (`256x256` LDS tiled GEMM) verified `C[0]=512.0` exact; all three Track A binaries produced SQLite traces via `rocprofv3` (`5447_results.db`, `5517_results.db`, `5587_results.db`). Track C live `vLLM` probing on base `rocm:latest` recorded `VLLM_NOT_PREINSTALLED` (`vllm_live_probe.json`) and intentionally skipped heavy multi-gigabyte wheel downloads to preserve cloud credit budget (<38 min cutoff).

## 4. Threats to Validity & Engineering Lessons Learned
- Ketidakhadiran `/dev/accel` pada WSL2 (perlu bare-metal Ubuntu 24.04 untuk `amdxdna` Linux).
- Resolusi `4x4.xclbin` untuk arsitektur Phoenix/Hawk Point (`'target': 'X1'`) pada Ryzen AI `1.9.0` agar tidak jatuh ke `AMD_AIE2P_4x4_Overlay` (Strix/XDNA 2).
- **Resolusi GID `/dev/kfd` Dinamis pada Kontainer `rocm:latest`:** Image `rocm:latest` (Ubuntu 24.04) tidak memiliki grup bernama `render` secara bawaan, sehingga flag `--group-add render` gagal saat startup; binding harus membaca GID numerik host secara dinamis (`stat -c '%g' /dev/kfd`) bersama `--group-add video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined` untuk akses KFD dan `rocprofv3`.
- **Skema `FP8` Spesifik CDNA 3 (`gfx942`):** Akselerasi `torch._scaled_mm` pada MI300X membutuhkan `torch.float8_e4m3fnuz` (bukan `float8_e4m3fn`) guna menghindari penalti konversi perangkat lunak dan mencapai puncak `1,111.94 TFLOPS`.
- **Overhead Peluncuran Kernel Kecil vs. Saturasi HBM3:** Kernel `saxpy_fma_kernel` pada `N = 4,194,304` (`32 MiB` *resident*, `48 MiB` trafik baca/tulis, `76.28 GB/s`) tidak mewakili batas lebar pita memori MI300X karena saturasi 304 CU dan 8 stack HBM3 baru tercapai pada salinan D2D multi-gigabyte (`3,929.20 GB/s` pada `4096 MB`).

## 5. Conclusion & Artifact Availability
- Seluruh kode benchmark, skrip audit SHA-256, dan dataset log terpusat di [`AMD AI Developer Program`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/README.md).
