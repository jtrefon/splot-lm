# Optimized kernels

The current reference uses native PyTorch/CUDA FP8 GEMMs for selected projections; see [FP8 qualification](../experiments/results/T0007/attempt-001/summary.md) and [autograd](../research/implementation/fp8-autograd.md). Custom fused GDN, grouped expert, quantization and attention kernels are not implemented here. Profile first, then qualify each optimized path against the existing numerical and causal reference. No claim of native FP4 on Ada.
