# RFC 0001 architecture accounting validation
2026-11-26 | #3 | CPU/meta only, not a model or hardware benchmark.
Command: python tools/test_accounting.py
Local environment Python 3.13.5, PyTorch 2.10.0+cpu. Independent ParameterList of meta tensors matching the proposed projection/normalization/expert shapes, not actual model modules or forward execution. No weights allocated, no data, no GPU.
Observed 6/6 tests passed (0.012s test runner, not a benchmark): meta inventory for Nano/Mini with and without MTP, exact Mini counts, depth/active bounds, extra readout cost, causal prefill MAC summation, invalid config rejection.
Local tested Git blobs: accounting.py fde7bd453af12df48cfb5b6dda8893cfe4e47bf2; test_accounting.py 27b13aac3c3da6dcf36b83f3dcb9a58545ff13b5.
Analytic contract scope: fixed 2/4/2 hybrid topology, equal GDN head dimensions, shared one expert, no optional depth-specific parameters. MACs are contraction counts only. Full forward-model numel and kernel profiler comparisons are required later. No performance claim.
