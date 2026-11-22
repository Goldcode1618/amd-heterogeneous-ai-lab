# AMD AI Academy — Course Progress & Publication Evidence Ledger (26 Kursus)

Ledger terpusat untuk melacak progres **26 kursus AMD AI Academy** lintas 3 tingkat (Beginner–Advanced), menautkan kode eksperimen ke [`MASTER_EXPERIMENT_LOG.jsonl`](../00_RESEARCH_LIBRARY/MASTER_EXPERIMENT_LOG.jsonl), serta merangkum temuan teknis yang dapat dikutip dalam publikasi.

---

## Ringkasan Progres

| Track | Direktori | Jumlah Kursus | Selesai (Verified Code/Lab) | Ready / Spec | Belum Mulai |
|---|---|---:|---:|---:|---:|
| **Track 1 — Beginner Basics** | [`01_beginner_basics/`](./01_beginner_basics/README.md) | 8 | 5 | 3 | 0 |
| **Track 2 — Intermediate Systems** | [`02_intermediate_systems/`](./02_intermediate_systems/README.md) | 11 | 5 | 6 | 0 |
| **Track 3 — Advanced RDNA & HIP** | [`03_advanced_rdna_hip/`](./03_advanced_rdna_hip/README.md) | 7 | 3 | 4 | 0 |
| **TOTAL** | — | **26** | **13** | **13** | **0** |

> *Status `COMPLETED` menandakan artefak kode/benchmark telah dieksekusi dan direkam secara kriptografis ke `MASTER_EXPERIMENT_LOG.jsonl`; `CODE_READY` menandakan kernel HIP/skrip referensi atau panduan readiness tersedia untuk eksekusi target ROCm/MI300X.*

---

## Track 1: Beginner Basics (`01_beginner_basics/` — 8 Kursus)

| Code | Course Title | Artifact / Script | Target Hardware | Experiment Log ID | Status | Key Takeaways for Publication |
|---|---|---|---|---|---|---|
| `ACAD-B01` | PyTorch on AMD | [`tensor_device_probe.py`](./01_beginner_basics/ACAD-B01_pytorch_rocm/tensor_device_probe.py) | Host / MI300X ROCm | `c1433137-7fda-43df-9046-e37649183a3a` | `COMPLETED` | GEMM+SiLU 512x512 probe (`1544.14 GFLOP/s`, `p50=0.168 ms`) & transparent HIP detection (`torch.version.hip`). |
| `ACAD-B02` | Model Quantization Fundamentals | [`int8_qdq_quantizer.py`](./01_beginner_basics/ACAD-B02_quantization/int8_qdq_quantizer.py) | XDNA 1 NPU / Zen 4 | `b8768013-ded0-4c58-a005-77fa7c061d44` | `COMPLETED` | Symmetric INT8 weights + asymmetric UINT8 activations achieve `38.62 dB` output SQNR (`cos_sim=0.999931`) & `4.0x` memory compression. |
| `ACAD-B03` | HIP Programming Basics | [`saxpy_fma_kernel.hip.cpp`](./02_intermediate_systems/ACAD-I01_cuda_to_hip/saxpy_fma_kernel.hip.cpp) | MI300X / RDNA 3 | `70a29a44-f1ff-4963-aff9-35b12e8e2e31` | `CODE_READY` | Portable `hipLaunchKernelGGL` + `__fmaf_rn` SAXPY kernel targeting `--offload-arch=gfx942,gfx1103`. |
| `ACAD-B04` | Hugging Face on AMD | [`test_tokenlab.py`](../04_CLOUD_AND_API_LABS/token_lab/test_tokenlab.py) | Token Lab / MI300X | `8f8d65cd-acd2-4baa-8e58-0de8be9500f8` | `COMPLETED` | Verified open-weight inference via `https://api.tokenlab.amd.com/v1` (`Gemma-4-31B-IT`, `2.337 tokens/s`, 10 active models). |
| `ACAD-B05` | AI Agents 101 | [`ACAD-B05_B06_pydantic_mcp_agent/`](./01_beginner_basics/ACAD-B05_B06_pydantic_mcp_agent/README.md) | AMD Token Lab / vLLM | `8/8 pytest PASS` | `COMPLETED` | Standalone ReAct agent (`PydanticAI 1.66.0`) with $M=3$ MCP servers (`time`, `weather`, `airbnb`) and `tool_prefix="weather"` collision fix. |
| `ACAD-B06` | AI Agents 201 | [`ACAD-B05_B06_pydantic_mcp_agent/`](./01_beginner_basics/ACAD-B05_B06_pydantic_mcp_agent/src/agent.py) | Token Lab / MI300X | `8/8 pytest PASS` | `COMPLETED` | Multi-backend `.env` switching across Token Lab, ROCm `vLLM`, Ollama, and LM Studio with structured `AgentExecutionError` diagnostics. |
| `ACAD-B07` | ONNX Runtime & Ryzen AI Intro | [`inspect_xdna_partitioning.py`](./01_beginner_basics/ACAD-B07_ryzen_ai_ort/inspect_xdna_partitioning.py) | Hawk Point XDNA 1 | `4d81f5c9-4ce4-4466-a3d7-305794749401` / `57124ff4-2d7b-4f12-8090-cdb89bfd8c4f` | `COMPLETED` | `VitisAIExecutionProvider` (`'target': 'X1'`, `phoenix/4x4.xclbin`) achieves `2.864 ms` p50 (`343.78 inf/s`), `4.86x` faster than Zen 4 CPU. |
| `ACAD-B08` | ROCm Container Basics | [`check_rocm_lab_readiness.ps1`](../03_ROCM_CERTIFIED_ASSOCIATE/check_rocm_lab_readiness.ps1) | WSL2 Docker / MI300X | `readiness_audit_latest.log` | `CODE_READY` | Docker + WSL2 readiness verification & bare-metal `/dev/accel` boundary documentation (`module_01_orientation_setup`). |

---

## Track 2: Intermediate Systems (`02_intermediate_systems/` — 11 Kursus)

| Code | Course Title | Artifact / Script | Target Hardware | Experiment Log ID | Status | Key Takeaways for Publication |
|---|---|---|---|---|---|---|
| `ACAD-I01` | CUDA-to-HIP Migration | [`cuda_to_hip_porting_lab.py`](./02_intermediate_systems/ACAD-I01_cuda_to_hip/cuda_to_hip_porting_lab.py) | MI300X / RDNA 3 | `70a29a44-f1ff-4963-aff9-35b12e8e2e31` | `COMPLETED` | Automated AST/regex CUDA->HIP translator achieving 100% clean port (`11` API calls + `1` triple-angle launch converted, 0 leftovers). |
| `ACAD-I02` | vLLM Serving on MI300X | [`vllm_serving_benchmark.py`](./02_intermediate_systems/ACAD-I02_vllm_mi300x/vllm_serving_benchmark.py) | Instinct MI300X | `c3bbf4bb-4c91-40cf-976b-786756de3c8d` | `COMPLETED` | `Llama-3.1-70B` FP8 uses `65.75 GiB` of `192 GB` HBM3, leaving `107.05 GiB` KV-cache for `85` concurrent `4K`-token sequences (`75.1 tok/s` ceiling). |
| `ACAD-I03` | Unsloth DeepSeek-R1 Fine-Tuning | [`setup_env.sh`](../04_CLOUD_AND_API_LABS/mi300x_dev_cloud/setup_env.sh) | Instinct MI300X | `mi300x_provisioning` | `CODE_READY` | MI300X single-node `192 GB HBM3` environment provisioning with `auto_shutdown_guard.sh` credit protection. |
| `ACAD-I04` | Custom HIP Kernels | [`saxpy_fma_kernel.hip.cpp`](./02_intermediate_systems/ACAD-I01_cuda_to_hip/saxpy_fma_kernel.hip.cpp) | Instinct MI300X | `hipcc_gfx942_gfx1103` | `CODE_READY` | Custom `__fmaf_rn` HIP kernel with `hipEventElapsedTime` bandwidth accounting. |
| `ACAD-I05` | Triton on ROCm | [`fused_rmsnorm_reference.py`](./02_intermediate_systems/ACAD-I05_triton_rocm/fused_rmsnorm_reference.py) | MI300X / Zen 4 Host | `5df12711-bda5-415e-9fa5-5cdd7cf8e681` | `COMPLETED` | Single-pass Fused RMSNorm (`max_diff=1.91e-06`) reduces global memory passes by `3.0x` and achieves `1.82x` host speedup (`8.775 ms`). |
| `ACAD-I06` | Kubernetes & AMD GPU Operator | [`module_05_profiling_optimization/`](../03_ROCM_CERTIFIED_ASSOCIATE/module_05_profiling_optimization/LAB_NOTES.md) | MI300X Cloud | `rocm_cert_m05` | `CODE_READY` | Deployment hygiene, `amd-smi` telemetry, and container scheduling checklist. |
| `ACAD-I07` | Distributed Training with RCCL | [`module_01_orientation_setup/`](../03_ROCM_CERTIFIED_ASSOCIATE/module_01_orientation_setup/LAB_NOTES.md) | Instinct MI300X | `rocm_cert_m01` | `CODE_READY` | NUMA topology (`rocm-smi --showtopo`) and multi-GPU RCCL ring/tree collective notes. |
| `ACAD-I08` | FlashAttention on CDNA 3 | [`roofline_analyzer.py`](./03_advanced_rdna_hip/ACAD-A02_rocprofv2_roofline/roofline_analyzer.py) | Instinct MI300X | `91c8f6eb-92a1-48dd-a082-7e2b90b356a5` | `CODE_READY` | SRAM tiling rationale for overcoming MI300X's `246.68 Ops/Byte` BF16 Roofline ridge point. |
| `ACAD-I09` | Mixed Precision & FP8 on MI300X | [`fp8_bf16_microbench.py`](./02_intermediate_systems/ACAD-I09_fp8_mixed_precision/fp8_bf16_microbench.py) | Instinct MI300X | `c2690881-2d65-4548-8d3c-4c01aca5caae` | `COMPLETED` | Per-tensor amax-scaled `FP8_E4M3` achieves `31.58 dB` SQNR (`0.0%` clipping) while unlocking `2614.9 TFLOPS` peak on CDNA 3 (`2x` BF16). |
| `ACAD-I10` | RAG Pipelines on AMD | [`edge_npu_cloud_rag_router.py`](./02_intermediate_systems/ACAD-I10_hybrid_rag/edge_npu_cloud_rag_router.py) | Edge + Token Lab | `373f3a3f-15d3-4e38-a923-58342acc0bcb` | `COMPLETED` | Hybrid Edge-to-Cloud RAG retrieves local embeddings in `2.299 ms` and routes multi-hop queries selectively to Cloud Token Lab / MI300X. |
| `ACAD-I11` | Multi-Agent Systems on ROCm | [`ACAD-B05_B06_pydantic_mcp_agent/`](./01_beginner_basics/ACAD-B05_B06_pydantic_mcp_agent/README.md) | Token Lab / MI300X | `8/8 pytest PASS` | `COMPLETED` | Coordinated multi-tool MCP orchestration (`Time` + `Weather` + `Airbnb`) validated for `vLLM` tool-calling endpoints. |

---

## Track 3: Advanced RDNA & HIP (`03_advanced_rdna_hip/` — 7 Kursus)

| Code | Course Title | Artifact / Script | Target Hardware | Experiment Log ID | Status | Key Takeaways for Publication |
|---|---|---|---|---|---|---|
| `ACAD-A01` | HIP Porting Overview for RDNA | [`cuda_to_hip_porting_lab.py`](./02_intermediate_systems/ACAD-I01_cuda_to_hip/cuda_to_hip_porting_lab.py) | Radeon 760M (`gfx1103`) | `70a29a44-f1ff-4963-aff9-35b12e8e2e31` | `CODE_READY` | `--offload-arch=gfx1103` target compilation and DirectML vs HIP trade-off on Windows vs Linux. |
| `ACAD-A02` | ROCm Profiling (`rocprofv2`/`omniperf`) | [`roofline_analyzer.py`](./03_advanced_rdna_hip/ACAD-A02_rocprofv2_roofline/roofline_analyzer.py) | MI300X / RDNA 3 / XDNA 1 | `91c8f6eb-92a1-48dd-a082-7e2b90b356a5` | `COMPLETED` | Cross-architecture Roofline engine (`MI300X BF16` ridge=`246.68`, `MI300X FP8`=`493.38`, `760M`=`95.98`, `XDNA 1`=`178.57 Ops/B`). |
| `ACAD-A03` | Debugging HIP with `ROCgdb` | [`module_05_profiling_optimization/`](../03_ROCM_CERTIFIED_ASSOCIATE/module_05_profiling_optimization/LAB_NOTES.md) | MI300X / RDNA | `rocm_cert_m05` | `CODE_READY` | Wavefront lane inspection & hardware counter checklist (`SQ_WAVES`, `TCC_MISS_sum`, `SQ_LDS_BANK_CONFLICT`). |
| `ACAD-A04` | Wavefront32 vs Wavefront64 Optimization | [`wavefront_reduction.hip.cpp`](./03_advanced_rdna_hip/ACAD-A04_wavefront32_vs_64/wavefront_reduction.hip.cpp) | RDNA 3 & CDNA 3 | `hipcc_wf32_wf64` | `CODE_READY` | Portable `__shfl_down` reduction parameterized by `__AMDGCN_WAVEFRONT_SIZE` (`32` on RDNA 3 vs `64` on CDNA 3). |
| `ACAD-A05` | LDS & Shared Memory Tuning | [`tiled_gemm_lds.hip.cpp`](./03_advanced_rdna_hip/ACAD-A05_lds_shared_memory/tiled_gemm_lds.hip.cpp) | RDNA 3 & CDNA 3 | `hipcc_lds_padded` | `CODE_READY` | Bank-conflict-free `16x16` tiled GEMM using `[TILE][TILE + 1]` LDS stride padding. |
| `ACAD-A06` | AIE/XDNA Kernel Offload Architecture | [`benchmark_directml_vs_npu.py`](../01_LOCAL_RYZEN_AI_1_9/directml_benchmarks/benchmark_directml_vs_npu.py) | Hawk Point XDNA 1 | `57124ff4-2d7b-4f12-8090-cdb89bfd8c4f` | `COMPLETED` | Empirical head-to-head: XDNA 1 NPU (`2.864 ms`) beats Radeon 760M DirectML (`8.074 ms`) by `2.82x` and Zen 4 CPU (`13.349 ms`) by `4.66x` (p50). |
| `ACAD-A07` | Production Inference Deployment & Telemetry | [`log_experiment.py`](../scripts/log_experiment.py) | Edge + MI300X | `MASTER_EXPERIMENT_LOG.jsonl` | `COMPLETED` | Cryptographic provenance pipeline (UUIDv4, Git commit, script SHA-256) exporting publication LaTeX tables and vector SVGs. |
