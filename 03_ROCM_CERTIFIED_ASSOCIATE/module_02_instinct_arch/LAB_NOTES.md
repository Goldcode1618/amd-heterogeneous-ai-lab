# Module 02: HIP Programming — Lab Notes

## Objectives
- Compile and execute HIP kernels (`hipcc`) and understand CUDA-to-HIP porting (`hipify-perl`, `hipify-clang`).
- Manage memory allocation, streams, and wavefront execution (`64`-thread wavefront on CDNA / MI300X).

## Key Commands & Exercises
```bash
hipcc --version
# Sample vector add compile
# hipcc -O3 vector_add.hip -o vector_add && ./vector_add
```

## Notes & Observations
- _Record HIP kernel compilation flags and timing results here._
