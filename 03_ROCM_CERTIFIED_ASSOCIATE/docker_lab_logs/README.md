# ROCm Certified Associate — Docker & Environment Audit Logs

Direktori ini menyimpan bukti eksekusi kontainer Docker, audit lingkungan WSL2/ROCm, dan output terminal terverifikasi untuk modul sertifikasi **ROCm Certified Associate** (`module_01_orientation_setup` s.d. `module_05_profiling_optimization`).

## Konvensi Penyimpanan Bukti Lab
1. **`readiness_audit_latest.log`**: Dihasilkan otomatis oleh [`check_rocm_lab_readiness.ps1`](file:///c:/Antigravity/AMD%20AI%20Developer%20Program/03_ROCM_CERTIFIED_ASSOCIATE/check_rocm_lab_readiness.ps1).
2. **Bukti Lab Kontainer (`module_0X_*.log`)**: Saat menjalankan lab di dalam kontainer ROCm (misalnya `rocm/pytorch`), simpan transkrip terminal dengan format `module_01_rocminfo.log`, `module_04_hipify_build.log`, `module_05_rocprofv2.log`, dll.
3. File `.log` di folder ini **dilacak di Git** sebagai lampiran audit sertifikasi dan reprodusibilitas riset.
