# Module 04: LLM Inference Optimization — Lab Notes

## Objectives
- Deploy high-throughput LLM serving on MI300X (192 GB HBM3) with `vLLM` on ROCm.
- Evaluate FP8 / BF16 quantization, PagedAttention, and tensor parallelism.

## Key Commands & Exercises
```bash
# Example vLLM serving check on MI300X
# python -m vllm.entrypoints.openai.api_server --model <hf-model> --dtype bfloat16
```

## Notes & Observations
- _Record TTFT, tokens/sec, and KV-cache utilization metrics here._
