# Track 2: Intermediate Systems (`ACAD-I01` – `ACAD-I11`)

Direktori ini menampung eksperimen sistem skala menengah-tinggi untuk 11 kursus AMD AI Academy (*CUDA-to-HIP Migration, vLLM Serving on MI300X, Unsloth DeepSeek-R1 Fine-Tuning, Custom HIP Kernels, Triton on ROCm, Kubernetes & AMD GPU Operator, RCCL, FlashAttention CDNA 3, FP8 MI300X, RAG, Multi-Agent Systems*).

## Protokol Efisiensi Kredit MI300X
- Siapkan dan verifikasi sintaks kode secara lokal terlebih dahulu sebelum *provisioning* instance MI300X (`$1.99/jam`).
- Jalankan [`auto_shutdown_guard.sh`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/04_CLOUD_AND_API_LABS/mi300x_dev_cloud/auto_shutdown_guard.sh) selama eksperimen berlangsung.
- Catat setiap hasil *benchmark* ke [`MASTER_BENCHMARK.csv`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/00_RESEARCH_LIBRARY/MASTER_BENCHMARK.csv) melalui [`scripts/log_experiment.py`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/scripts/log_experiment.py).
