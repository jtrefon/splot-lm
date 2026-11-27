# Native FP8 scaled-GEMM probe

Status: T0007 operator-feasibility implementation, issue [#51](https://github.com/jtrefon/splot-lm/issues/51). This is not FP8 training, an autograd integration or a language-quality result.

## Scope

Qualify the existing private `torch._scaled_mm` path on the owner RTX 4090 (`sm_89`) with the installed Python 3.12.7 / torch 2.10.0+cu128 stack. No package installation, CUDA/driver change, model integration, harness edit or real corpus is involved.

The probe distinguishes native FP8 dispatch from simulated quantization:

- actual successful `torch._scaled_mm` calls receive `torch.float8_e4m3fn` forward operands;
- dgrad and wgrad receive `torch.float8_e5m2` output-gradient operands and `torch.float8_e4m3fn` other operands;
- output is BF16 (FP8 output is not used);
- unsupported formats/shapes record the exact exception and are not silently cast or replaced by a fake-quant matmul.

## API and layout contract

Installed schema:

```text
aten::_scaled_mm(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b,
                 Tensor? bias=None, Tensor? scale_result=None,
                 ScalarType? out_dtype=None, bool use_fast_accum=False) -> Tensor
```

- LHS is row-major contiguous `[M,K]`.
- RHS is the logical `[K,N]` matrix in column-major layout: for a row-major `[N,K]` weight this is `W.T` (stride `[1,K]`); other orientations use `W.T.contiguous().T` or `X.T.contiguous().T` and record the copy.
- Per-tensor FP32 scalar scales are passed as 0-dimensional CUDA tensors.
- Quantization uses current scale `s = max(abs(x)) / finfo(dtype).max`; all-zero input uses `s=1`. Values are clamped to the finite FP8 range, cast to E4M3FN or E5M2, and dequantized as `q.float() * s`.

The probe records actual input dtypes, shapes, strides, scales, output dtype/stride, layout-copy flags and the native operator call. It also records saturation, underflow and quantized-zero counts with explicit denominators.

## Cases and gates

Six aligned shapes: `(M,K,N) = (16,384,512), (64,384,512), (512,384,512), (64,384,384), (64,512,384), (64,384,1152)`.

Four routed-token-like shapes: `M = 1,3,7,17`, `K=384`, `N=512`, padded with zero rows to a multiple of 16; forward/dgrad outputs are trimmed and the padded work/overhead is explicit. This is not a sparse-execution measurement.

Each case computes explicit forward `Y = X W^T`, dgrad `dX = dY W` and wgrad `dW = dY^T X`. FP32 references use TF32 disabled. A BF16 matmul comparator uses the same inputs. For random cases:

- relative Frobenius error versus FP32 `<= 0.15`;
- relative Frobenius error versus the FP32 product of dequantized operands `<= 0.025`;
- output finite and native FP8 dispatch observed.

These are provisional smoke tolerances only. An all-zero case must be exactly zero, and a nonfinite input must be rejected.

## Benchmarks and profiling

For each supported case/op, `tools/fp8_probe.py` measures:

- prequantized native FP8 GEMM;
- quantization-inclusive work (scale/cast/layout plus GEMM);
- paired BF16 matmul baseline.

Five warmups and 20 synchronized measured iterations are used; median/p95, allocator peaks and the timing method are recorded. No speed threshold or superiority claim is applied. A bounded representative `torch.profiler` call records operator/kernel names when available; no Tensor Core instruction claim is made unless verified.

## Files and commands

- `tools/fp8_probe.py` — pure quantization/layout/error helpers plus the CUDA probe.
- `tools/test_fp8_probe.py` — CPU-only deterministic helper tests.
- `research/implementation/fp8-probe.md` — this contract.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:tools python tools/test_fp8_probe.py
PYTHONPATH=src:tools CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
  timeout --kill-after=15s 180s python tools/fp8_probe.py \
  --device cuda --seed 42 --out "$RUNS/fp8-probe.json"
```

## Limitations

- Operator feasibility only; no model/training integration, autograd, optimizer, checkpoint or language-quality claim.
- Current per-tensor scaling only; no delayed scaling, MXFP8, NVFP4 or block scaling.
- Mixed E5M2/E4M3 support is recorded as observed or as an exact operator error; it is not assumed from the dtype label.
- FP8 output is BF16. Quantization-inclusive timings include Python/layout overhead; no production throughput claim.
- No packages, CUDA/driver/toolkit changes or model/harness edits.
- The synthetic normal inputs and representative shapes are not a corpus benchmark.
