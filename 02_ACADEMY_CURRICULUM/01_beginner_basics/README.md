# Track 1: Beginner Basics (`ACAD-B01` – `ACAD-B08`)

Direktori ini menampung materi, kode praktikum, dan catatan riset untuk 8 kursus tingkat dasar AMD AI Academy (*PyTorch on AMD, Model Quantization Fundamentals, HIP Programming Basics, Hugging Face on AMD, AI Agents 101/201, ONNX Runtime & Ryzen AI Intro, ROCm Container Basics*).

## Konvensi Penamaan Sub-Lab
- Buat folder per kursus: `ACAD-B01_pytorch_on_amd/`, `ACAD-B02_quantization/`, dst.
- Rekam setiap uji coba empiris ke [`MASTER_EXPERIMENT_LOG.jsonl`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/00_RESEARCH_LIBRARY/MASTER_EXPERIMENT_LOG.jsonl) menggunakan [`scripts/log_experiment.py`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/scripts/log_experiment.py):

```powershell
python ..\..\scripts\log_experiment.py record `
  --track "academy_beginner" `
  --hardware-tier "Edge_NPU" `
  --device-name "AMD Ryzen AI NPU Phoenix (XDNA 1)" `
  --execution-provider "VitisAIExecutionProvider" `
  --model-or-workload "ACAD-B07_demo" `
  --precision "INT8" `
  --status "PASS" `
  --notes "Completed beginner module lab"
```
