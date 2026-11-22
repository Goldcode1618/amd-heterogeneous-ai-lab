# Module 04: CUDA-to-HIP Porting & High-Throughput LLM Serving (`vLLM`) — Lab Notes

## Objectives
- Port CUDA kernels and `cuBLAS` calls to HIP / `rocBLAS` using `hipify-perl`, `hipify-clang`, and deterministic portability audits.
- Deploy high-throughput LLM serving on **AMD Instinct MI300X (192 GB HBM3)** with `vLLM` PagedAttention, BF16, and FP8 quantization.

## PagedAttention HBM3 Capacity Equation (192 GB MI300X)
For a Transformer with $L$ layers, $H_{kv}$ GQA KV heads, head dimension $d_h$, and precision $b_{kv}$ bytes/element:
$$\text{KV}_{\text{token}} = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b_{kv}$$
$$\text{Max Concurrent Sequences} = \left\lfloor \frac{0.90 \times 192\text{ GiB} - W_{\text{model}}}{\text{KV}_{\text{token}} \times S_{\text{seq}}} \right\rfloor$$

Using [`vllm_serving_benchmark.py`](../../02_ACADEMY_CURRICULUM/02_intermediate_systems/ACAD-I02_vllm_mi300x/vllm_serving_benchmark.py), a single MI300X (`192 GB HBM3`) comfortably hosts `Llama-3.1-70B` in FP8 (`~65.75 GiB` weights) leaving `~107 GiB` for PagedAttention KV-cache (`83+` concurrent `4096`-token sequences).

## Key Commands & Exercises
```bash
# 1. Automated CUDA-to-HIP source translation & audit
python ../../02_ACADEMY_CURRICULUM/02_intermediate_systems/ACAD-I01_cuda_to_hip/cuda_to_hip_porting_lab.py

# 2. Analytical MI300X vLLM KV-cache capacity sizing
python ../../02_ACADEMY_CURRICULUM/02_intermediate_systems/ACAD-I02_vllm_mi300x/vllm_serving_benchmark.py --mode analytical

# 3. Launch vLLM OpenAI-compatible server on MI300X (connects directly with pydantic-ai-multi-mcp-agent)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B --dtype bfloat16 --enable-auto-tool-choice --port 8000
```
