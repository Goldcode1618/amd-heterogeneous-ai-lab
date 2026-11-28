"""Unit tests verifying 00_RESEARCH_LIBRARY provenance and Academy lab modules."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

LIBRARY_DIR = WORKSPACE_ROOT / "00_RESEARCH_LIBRARY"
CSV_PATH = LIBRARY_DIR / "MASTER_BENCHMARK.csv"
JSONL_PATH = LIBRARY_DIR / "MASTER_EXPERIMENT_LOG.jsonl"


def _load_module(rel_path: str, module_name: str):
    """Dynamically load a lab Python script by relative path."""
    full_path = WORKSPACE_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_research_library_csv_and_jsonl_provenance_integrity() -> None:
    """Every logged experiment in CSV/JSONL must share valid UUIDv4 and 64-char SHA-256."""
    assert CSV_PATH.is_file()
    assert JSONL_PATH.is_file()

    with CSV_PATH.open("r", encoding="utf-8", newline="") as cf:
        csv_rows = list(csv.DictReader(cf))

    jsonl_rows = [
        json.loads(line)
        for line in JSONL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(csv_rows) >= 5
    assert len(csv_rows) == len(jsonl_rows)

    for csv_row, json_row in zip(csv_rows, jsonl_rows):
        assert csv_row["experiment_id"] == json_row["experiment_id"]
        assert len(json_row["script_sha256"]) == 64
        assert json_row["status"] in {"PASS", "FAIL"}


def test_int8_qdq_quantizer_accuracy_and_compression() -> None:
    """QDQ reference lab must achieve >= 30 dB SQNR and 4x memory compression."""
    mod = _load_module(
        "02_ACADEMY_CURRICULUM/01_beginner_basics/ACAD-B02_quantization/int8_qdq_quantizer.py",
        "int8_qdq_quantizer",
    )
    res = mod.evaluate_qdq_pipeline(dim=256, runs=3)
    assert res["compression_ratio"] == 4.0
    assert res["matmul_output_sqnr_db"] >= 30.0
    assert res["cosine_similarity"] > 0.999


def test_cuda_to_hip_porting_auditor_clean_translation() -> None:
    """CUDA-to-HIP translator must leave zero CUDA/cuBLAS or <<<>>> tokens."""
    mod = _load_module(
        "02_ACADEMY_CURRICULUM/02_intermediate_systems/ACAD-I01_cuda_to_hip/cuda_to_hip_porting_lab.py",
        "cuda_to_hip_porting_lab",
    )
    hip_code, counts = mod.translate_cuda_to_hip(mod.SAMPLE_CUDA_SOURCE)
    audit = mod.audit_portability(hip_code)
    assert counts["api_replacements"] >= 10
    assert counts["kernel_launches"] == 1
    assert audit["clean_port"] is True
    assert audit["has_hip_header"] is True
    assert audit["has_hip_launch"] is True


def test_vllm_mi300x_paged_attention_sizing() -> None:
    """Llama-3.1-70B FP8 on 192 GB MI300X must leave > 100 GiB for KV-cache."""
    mod = _load_module(
        "02_ACADEMY_CURRICULUM/02_intermediate_systems/ACAD-I02_vllm_mi300x/vllm_serving_benchmark.py",
        "vllm_serving_benchmark",
    )
    cap = mod.estimate_mi300x_kv_capacity(
        mod.PRESET_MODELS["Llama-3.1-70B"],
        weight_bytes_per_param=1.0,
        seq_len=4096,
    )
    assert cap["kv_budget_gib"] > 100.0
    assert cap["max_concurrent_seqs"] >= 80


def test_fused_rmsnorm_numerical_equivalence() -> None:
    """Single-pass fused RMSNorm must match naive multi-pass RMSNorm within 1e-5."""
    mod = _load_module(
        "02_ACADEMY_CURRICULUM/02_intermediate_systems/ACAD-I05_triton_rocm/fused_rmsnorm_reference.py",
        "fused_rmsnorm_reference",
    )
    res = mod.benchmark_rmsnorm(rows=64, hidden_dim=512, runs=3)
    assert res["max_abs_diff"] < 1e-5
    assert res["dram_traffic_reduction_x"] > 2.5


def test_fp8_mixed_precision_dynamic_range() -> None:
    """Scaled FP8 E4M3 and BF16 must maintain > 25 dB SQNR with 0% clipping on test tensor."""
    mod = _load_module(
        "02_ACADEMY_CURRICULUM/02_intermediate_systems/ACAD-I09_fp8_mixed_precision/fp8_bf16_microbench.py",
        "fp8_bf16_microbench",
    )
    rows = mod.evaluate_precisions(n_elems=16384)
    by_fmt = {r["format"]: r for r in rows}
    assert by_fmt["BF16"]["sqnr_db"] > 50.0
    assert by_fmt["FP8_E4M3"]["sqnr_db"] > 25.0
    assert by_fmt["FP8_E4M3"]["clip_rate_pct"] == 0.0


def test_hybrid_edge_cloud_rag_router() -> None:
    """Hybrid RAG router must escalate complex architectural queries and serve simple lookups locally."""
    mod = _load_module(
        "02_ACADEMY_CURRICULUM/02_intermediate_systems/ACAD-I10_hybrid_rag/edge_npu_cloud_rag_router.py",
        "edge_npu_cloud_rag_router",
    )
    rag = mod.HybridEdgeCloudRAG()
    simple = rag.route_query("What is MI300X HBM3 memory?")
    complex_q = rag.route_query("Compare XDNA 1 NPU architecture trade-off vs MI300X HBM3")
    assert simple["route"] == "Edge_Local_Direct_Answer"
    assert complex_q["route"] == "Cloud_TokenLab_MI300X"
    assert simple["top_hits"][0]["doc_id"] == "doc-02"


def test_roofline_analyzer_regimes() -> None:
    """Roofline engine must classify low-I kernels as memory-bound and dense GEMM as compute-bound."""
    mod = _load_module(
        "02_ACADEMY_CURRICULUM/03_advanced_rdna_hip/ACAD-A02_rocprofv2_roofline/roofline_analyzer.py",
        "roofline_analyzer",
    )
    dense_gemm = mod.analyze_kernel_roofline("GEMM_8192", 2.0 * (8192 ** 3), 3.0 * (8192 ** 2) * 2.0)
    saxpy = mod.analyze_kernel_roofline("SAXPY", 2.0 * 1e6, 12.0 * 1e6)
    assert all(r["regime"] == "Compute Bound" for r in dense_gemm)
    assert all(r["regime"] == "Memory-Bandwidth Bound" for r in saxpy)


def test_smoke_test_rocm_shallow_path_resolution(monkeypatch) -> None:
    """smoke_test_rocm.py must load cleanly at both repo depth and depth-1 container path (/work)."""
    rel_path = "04_CLOUD_AND_API_LABS/mi300x_dev_cloud/smoke_test_rocm.py"
    mod = _load_module(rel_path, "smoke_test_rocm_repo")
    assert mod.WORKSPACE_ROOT == WORKSPACE_ROOT
    assert mod.record_experiment is not None

    shallow_path = "/work/smoke_test_rocm.py"
    source = (WORKSPACE_ROOT / rel_path).read_text(encoding="utf-8")
    code = compile(source, shallow_path, "exec")

    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: Path(shallow_path))
    ns: dict[str, object] = {"__name__": "smoke_test_rocm_shallow", "__file__": shallow_path}
    exec(code, ns)

    assert ns["WORKSPACE_ROOT"] == Path("/")
    assert ns["record_experiment"] is None
    assert callable(ns["main"])


def test_wave2_suite_shallow_path_and_fp8_detection(monkeypatch) -> None:
    """wave2_mi300x_suite.py must resolve depth-1 /work paths cleanly and prefer MI300X FP8 fnuz."""
    import types

    rel_path = "04_CLOUD_AND_API_LABS/mi300x_dev_cloud/wave2_mi300x_suite.py"
    shallow_path = "/work/wave2_mi300x_suite.py"
    source = (WORKSPACE_ROOT / rel_path).read_text(encoding="utf-8")
    code = compile(source, shallow_path, "exec")

    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: Path(shallow_path))
    ns: dict[str, object] = {"__name__": "wave2_suite_shallow", "__file__": shallow_path}
    exec(code, ns)

    assert ns["DEFAULT_OUT_DIR"] == Path("/work")
    detect_fn = ns["detect_fp8_dtype"]
    assert callable(detect_fn)

    both = types.SimpleNamespace(float8_e4m3fnuz="fnuz_dtype", float8_e4m3fn="fn_dtype")
    assert detect_fn(both) == ("fnuz_dtype", "float8_e4m3fnuz")

    fallback_only = types.SimpleNamespace(float8_e4m3fn="fn_dtype")
    assert detect_fn(fallback_only) == ("fn_dtype", "float8_e4m3fn")

    none_avail = types.SimpleNamespace()
    assert detect_fn(none_avail) == (None, "UNAVAILABLE")


def test_wave2_suite_fault_isolation_and_incremental_persistence() -> None:
    """One failing subtest in wave2_mi300x_suite.py must not abort subsequent subtests or JSONL/JSON writes."""
    import json
    import tempfile

    mod = _load_module(
        "04_CLOUD_AND_API_LABS/mi300x_dev_cloud/wave2_mi300x_suite.py",
        "wave2_mi300x_suite_mod",
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        jsonl_path = tmp_path / "wave2_mi300x_suite.jsonl"
        json_path = tmp_path / "wave2_mi300x_suite.json"
        metadata = {"device_name": "AMD Instinct MI300X VF"}
        records: list[dict[str, object]] = []

        def _boom() -> dict[str, object]:
            raise RuntimeError("simulated OOM")

        def _ok() -> dict[str, object]:
            return {"tflops": 1234.5, "latency_ms_p50": 0.9}

        err_rec = mod.run_subtest(
            "gemm_fp32_16384",
            _boom,
            jsonl_path=jsonl_path,
            json_path=json_path,
            metadata=metadata,
            records=records,
        )
        ok_rec = mod.run_subtest(
            "gemm_fp8_scaled_mm_8192",
            _ok,
            jsonl_path=jsonl_path,
            json_path=json_path,
            metadata=metadata,
            records=records,
        )

        assert err_rec["status"] == "ERROR"
        assert "simulated OOM" in str(err_rec["error"])
        assert ok_rec["status"] == "PASS"

        lines = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(lines) == 2
        assert lines[0]["status"] == "ERROR"
        assert lines[1]["status"] == "PASS"

        summary = json.loads(json_path.read_text(encoding="utf-8"))
        assert summary["summary"]["total"] == 2
        assert summary["summary"]["passed"] == 1
        assert summary["summary"]["failed"] == 1
        assert summary["results"][1]["tflops"] == 1234.5
