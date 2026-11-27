"""Native FP8 first-order linear autograd for selected SPLOT projections.

The adapter preserves the original FP32 master Parameter objects and state_dict
keys. Only the approved bias-free projection roles are converted.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.autograd.function import once_differentiable

from .dense import GQA, Block
from .gdn import GatedDeltaNet
from .moe import Expert, SparseFFN
from .mtp import MTPLM


FP8_E4M3 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2
ALIGNMENT = 16
SCALE_DTYPE = torch.float32


def padding_plan(real_m, multiple=ALIGNMENT):
    if type(real_m) is not int or real_m < 0 or type(multiple) is not int or multiple < 1:
        raise ValueError('invalid padding dimensions')
    padded = ((real_m + multiple - 1) // multiple) * multiple
    return {'real_M': real_m, 'padded_M': padded, 'padding_rows': padded - real_m,
            'padding_fraction': (padded - real_m) / padded if padded else 0.,
            'overhead_ratio': padded / real_m if real_m else 1., 'multiple': multiple,
            'padding_value': 0}


def pad_rows(x, padded_m):
    if x.ndim < 1 or padded_m < x.shape[0]:
        raise ValueError('invalid row padding')
    if padded_m == x.shape[0]:
        return x
    out = torch.zeros((padded_m, *x.shape[1:]), dtype=x.dtype, device=x.device)
    out[:x.shape[0]] = x
    return out


def _row_major(x):
    if x.stride() == (x.shape[1], 1):
        return x
    return x.contiguous()


def _column_major(x):
    if x.stride() == (1, x.shape[0]):
        return x
    return x.t().contiguous().t()


def _quantize_audit(x, dtype):
    """Device-resident current per-tensor quantization plus audit counters."""
    x = x.float()
    maximum = float(torch.finfo(dtype).max)
    absmax = x.abs().amax()
    scale = absmax / maximum
    scale = torch.where(absmax > 0, scale.clamp_min(torch.finfo(torch.float32).tiny),
                        torch.ones_like(scale))
    scaled = x / scale
    saturation = (scaled.abs() > maximum).sum().to(torch.int64)
    q = torch.clamp(scaled, -maximum, maximum).to(dtype)
    q_float = q.float()
    nonzero = (x != 0).sum().to(torch.int64)
    underflow = ((q_float == 0) & (x != 0)).sum().to(torch.int64)
    finite = torch.isfinite(x).all()
    return q, scale, {'saturation': saturation, 'underflow': underflow,
                      'nonzero': nonzero, 'finite': finite}


def _quantize(x, dtype):
    q, scale, _ = _quantize_audit(x, dtype)
    return q, scale


def _audit_add(module, name, value):
    old = getattr(module, name, None)
    setattr(module, name, value if old is None else old + value)


def _audit_or(module, name, value):
    old = getattr(module, name, None)
    setattr(module, name, value if old is None else torch.logical_or(old, value))


def _record_quant(module, role, stats):
    for key in ('saturation', 'underflow', 'nonzero'):
        _audit_add(module, f'audit_{role}_{key}', stats[key])
    _audit_or(module, 'audit_nonfinite', ~stats['finite'])


def _record_output(module, value):
    _audit_or(module, 'audit_nonfinite', ~torch.isfinite(value).all())


def _mm(a, b, scale_a, scale_b, out_dtype):
    return torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=out_dtype, use_fast_accum=False)



def quantize_device(x, dtype):
    return _quantize(x, dtype)


def dequantize(q, scale):
    return q.float() * scale

def relative_frobenius(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError(f'shape mismatch: {tuple(actual.shape)} != {tuple(reference.shape)}')
    numerator = torch.linalg.vector_norm((actual.float() - reference.float()).flatten())
    denominator = torch.linalg.vector_norm(reference.float().flatten())
    return float((numerator / (denominator + 1e-12)).detach())


class _FP8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, module):
        if x.device.type != 'cuda' or weight.device.type != 'cuda':
            raise RuntimeError('native FP8 linear requires a CUDA device')
        if weight.dtype != torch.float32:
            raise TypeError('FP8 master weight must remain float32')
        if x.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError(f'unsupported FP8 linear input dtype: {x.dtype}')
        if x.ndim < 1 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
            raise ValueError('invalid FP8 linear shapes')
        n, k = weight.shape
        if k % ALIGNMENT or n % ALIGNMENT:
            raise ValueError(f'FP8 linear K/N must be multiples of {ALIGNMENT}: K={k}, N={n}')
        shape = tuple(x.shape)
        m = math.prod(shape[:-1])
        module.forward_calls += 1
        if m == 0:
            ctx.empty = True
            ctx.input_shape = shape
            ctx.input_dtype = x.dtype
            ctx.k, ctx.n = k, n
            ctx.module = module
            module.empty_forward_calls += 1
            module.last_forward_dispatched = False
            return torch.empty((*shape[:-1], n), device=x.device, dtype=torch.bfloat16)
        plan = padding_plan(m)
        padded_m = plan['padded_M']
        x_pad = pad_rows(x.reshape(m, k).float(), padded_m)
        qx, sx, input_stats = _quantize_audit(x_pad, FP8_E4M3)
        qw, sw, weight_stats = _quantize_audit(weight, FP8_E4M3)
        _record_quant(module, 'input', input_stats)
        _record_quant(module, 'weight', weight_stats)
        rhs = _column_major(qw.t())
        out_pad = _mm(qx, rhs, sx, sw, torch.bfloat16)
        _record_output(module, out_pad)
        module.audit_native_forward_calls += 1
        ctx.save_for_backward(qx, qw, sx, sw)
        ctx.empty = False
        ctx.input_shape = shape
        ctx.input_dtype = x.dtype
        ctx.m, ctx.k, ctx.n, ctx.padded_m = m, k, n, padded_m
        ctx.module = module
        module.padded_forward_rows += padded_m - m
        module.last_forward_padded_m = padded_m
        module.last_forward_scales = (sx.detach(), sw.detach())
        module.last_forward_dispatched = True
        return out_pad[:m].reshape(*shape[:-1], n)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        module = ctx.module
        module.backward_calls += 1
        if ctx.empty:
            dx = torch.zeros(ctx.input_shape, device=grad_out.device, dtype=ctx.input_dtype)
            dw = torch.zeros((ctx.n, ctx.k), device=grad_out.device, dtype=torch.float32)
            module.empty_backward_calls += 1
            return dx, dw, None
        qx, qw, sx, sw = ctx.saved_tensors
        m = ctx.m
        grad = grad_out.reshape(m, ctx.n).float()
        grad_pad = pad_rows(grad, ctx.padded_m)
        qdy, sdy, grad_stats = _quantize_audit(grad_pad, FP8_E5M2)
        _record_quant(module, 'grad', grad_stats)
        dx_pad = _mm(qdy, _column_major(qw), sdy, sw, torch.bfloat16)
        dx = dx_pad[:m].to(ctx.input_dtype).reshape(ctx.input_shape)
        _record_output(module, dx_pad)
        dw = _mm(_row_major(qdy.t()), _column_major(qx), sdy, sx, torch.float32)
        _record_output(module, dw)
        module.audit_native_backward_calls += 1
        module.audit_wgrad_fp32_calls += 1
        module.padded_backward_rows += ctx.padded_m - m
        module.last_backward_padded_m = ctx.padded_m
        module.last_backward_scales = (sdy.detach(), sx.detach(), sw.detach())
        return dx, dw, None


class FP8Linear(nn.Linear):
    """nn.Linear-compatible adapter around a preserved FP32 master Parameter."""
    def __init__(self, in_features, out_features):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = None
        self.reset_stats()

    @classmethod
    def from_linear(cls, linear):
        if not isinstance(linear, nn.Linear) or linear.bias is not None:
            raise ValueError('FP8 conversion requires a bias-free nn.Linear')
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module.in_features = linear.in_features
        module.out_features = linear.out_features
        module.weight = linear.weight
        module.bias = None
        module.reset_stats()
        return module

    def reset_stats(self):
        self.forward_calls = 0
        self.backward_calls = 0
        self.empty_forward_calls = 0
        self.empty_backward_calls = 0
        self.padded_forward_rows = 0
        self.padded_backward_rows = 0
        self.audit_native_forward_calls = 0
        self.audit_native_backward_calls = 0
        self.audit_wgrad_fp32_calls = 0
        for role in ('input', 'weight', 'grad'):
            for key in ('saturation', 'underflow', 'nonzero'):
                setattr(self, f'audit_{role}_{key}', None)
        self.audit_nonfinite = None
        self.last_forward_padded_m = None
        self.last_backward_padded_m = None
        self.last_forward_scales = None
        self.last_backward_scales = None
        self.last_forward_dispatched = False

    def forward(self, x):
        if self.bias is not None:
            raise ValueError('FP8Linear is bias-free')
        return _FP8LinearFunction.apply(x, self.weight, self)


def classify_linear(model, module_path):
    parent_path, local = module_path.rsplit('.', 1) if '.' in module_path else ('', module_path)
    parent = model.get_submodule(parent_path) if parent_path else model
    if isinstance(parent, GQA):
        if local in ('q', 'k', 'v', 'o'):
            return 'selected', 'gqa_projection'
        if local == 'output_gate':
            return 'excluded', 'gqa_output_gate'
    if isinstance(parent, GatedDeltaNet):
        if local in ('q', 'k', 'v', 'z', 'o'):
            return 'selected', 'gdn_projection'
        if local in ('a', 'b'):
            return 'excluded', 'gdn_scalar_gate'
    if isinstance(parent, Block) and local in ('gate', 'up', 'down'):
        return 'selected', 'dense_ffn'
    if isinstance(parent, Expert) and local in ('gate', 'up', 'down'):
        return 'selected', 'expert_ffn'
    if isinstance(parent, SparseFFN) and local == 'router':
        return 'excluded', 'moe_router'
    if isinstance(parent, MTPLM) and local == 'offset_projection':
        return 'selected', 'mtp_offset_projection'
    return 'unknown', type(parent).__name__


def convert_selected_linears(model):
    before_ids = {id(p) for p in model.parameters()}
    before_keys = set(model.state_dict())
    selected, excluded, unknown = [], [], []
    for path, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, FP8Linear):
            continue
        status, role = classify_linear(model, path)
        if status == 'selected':
            if module.bias is not None:
                raise ValueError(f'selected projection has bias: {path}')
            parent_path, local = path.rsplit('.', 1) if '.' in path else ('', path)
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, local, FP8Linear.from_linear(module))
            selected.append({'name': path, 'role': role})
        elif status == 'excluded':
            excluded.append({'name': path, 'role': role})
        else:
            unknown.append({'name': path, 'owner': role})
    if unknown:
        raise ValueError(f'unknown eligible linear roles: {unknown}')
    if not selected:
        raise ValueError('no selected FP8 projections')
    after_ids = {id(p) for p in model.parameters()}
    if before_ids != after_ids:
        raise RuntimeError('FP8 conversion changed parameter identity')
    if before_keys != set(model.state_dict()):
        raise RuntimeError('FP8 conversion changed state_dict keys')
    return {
        'format': 'e4m3fn-forward/e5m2-backward',
        'backend': 'torch._scaled_mm',
        'scales': 'per-tensor-current-fp32-device',
        'output_dtype': 'bfloat16',
        'padding_multiple': ALIGNMENT,
        'selected': selected, 'excluded': excluded,
        'selected_count': len(selected), 'excluded_count': len(excluded),
        'parameter_count': len(before_ids),
        'native_dispatch': True, 'adamw_only': True,
        'no_host_scale_roundtrip': True,
    }


def fp8_runtime_probe(device):
    if str(device) != 'cuda' and torch.device(device).type != 'cuda':
        return {'available': False, 'error': 'FP8 requires CUDA'}
    if not torch.cuda.is_available() or not hasattr(torch, '_scaled_mm'):
        return {'available': False, 'error': 'CUDA or torch._scaled_mm unavailable'}
    try:
        a = torch.ones(16, 16, device=device, dtype=FP8_E4M3)
        b = torch.ones(16, 16, device=device, dtype=FP8_E4M3).t()
        scale = torch.tensor(1., device=device, dtype=torch.float32)
        out = _mm(a, b, scale, scale, torch.bfloat16)
        return {'available': True, 'operator': 'torch._scaled_mm',
                'schema': str(torch.ops.aten._scaled_mm.default._schema),
                'output_dtype': str(out.dtype).replace('torch.', ''),
                'float8_dtypes': [str(FP8_E4M3).replace('torch.', ''), str(FP8_E5M2).replace('torch.', '')]}
    except Exception as exc:
        return {'available': False, 'error': f'{type(exc).__name__}: {exc}'}


def has_fp8(model):
    return any(isinstance(module, FP8Linear) for module in model.modules())


def reset_fp8_audit(model):
    for module in model.modules():
        if isinstance(module, FP8Linear):
            module.reset_stats()


def _audit_total(modules, name):
    values = [getattr(module, name, None) for module in modules
              if getattr(module, name, None) is not None]
    return sum(int(value.item()) for value in values)


def fp8_assert_finite(model, stage='boundary'):
    modules = [module for module in model.modules() if isinstance(module, FP8Linear)]
    if not modules:
        return
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f'nonfinite FP8 master parameter at {stage}')
    for module in modules:
        if module.audit_nonfinite is not None and bool(module.audit_nonfinite.item()):
            raise FloatingPointError(f'nonfinite FP8 quantization/output at {stage}')
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f'nonfinite FP8 gradient at {stage}')


def fp8_diagnostics(model):
    modules = [module for module in model.modules() if isinstance(module, FP8Linear)]
    forward = sum(module.forward_calls for module in modules)
    backward = sum(module.backward_calls for module in modules)
    selected_scales = [scale for module in modules
                       if module.last_forward_scales is not None
                       for scale in module.last_forward_scales]
    quantization = {
        'input_saturation': _audit_total(modules, 'audit_input_saturation'),
        'weight_saturation': _audit_total(modules, 'audit_weight_saturation'),
        'grad_saturation': _audit_total(modules, 'audit_grad_saturation'),
        'input_underflow': _audit_total(modules, 'audit_input_underflow'),
        'weight_underflow': _audit_total(modules, 'audit_weight_underflow'),
        'grad_underflow': _audit_total(modules, 'audit_grad_underflow'),
        'input_nonzero': _audit_total(modules, 'audit_input_nonzero'),
        'weight_nonzero': _audit_total(modules, 'audit_weight_nonzero'),
        'grad_nonzero': _audit_total(modules, 'audit_grad_nonzero'),
        'nonfinite': any(module.audit_nonfinite is not None and bool(module.audit_nonfinite.item())
                         for module in modules),
    }
    total_underflow = (quantization['input_underflow'] + quantization['weight_underflow']
                       + quantization['grad_underflow'])
    total_nonzero = (quantization['input_nonzero'] + quantization['weight_nonzero']
                     + quantization['grad_nonzero'])
    quantization['underflow_fraction'] = total_underflow / total_nonzero if total_nonzero else 0.
    return {
        'selected_modules': len(modules),
        'forward_calls': forward,
        'backward_calls': backward,
        'forward_coverage': (sum(module.forward_calls > 0 for module in modules) / len(modules)) if modules else None,
        'backward_coverage': (sum(module.backward_calls > 0 for module in modules) / len(modules)) if modules else None,
        'empty_forward_calls': sum(module.empty_forward_calls for module in modules),
        'empty_backward_calls': sum(module.empty_backward_calls for module in modules),
        'padded_forward_rows': sum(module.padded_forward_rows for module in modules),
        'padded_backward_rows': sum(module.padded_backward_rows for module in modules),
        'native_forward_calls': sum(module.audit_native_forward_calls for module in modules),
        'native_backward_calls': sum(module.audit_native_backward_calls for module in modules),
        'wgrad_fp32_calls': sum(module.audit_wgrad_fp32_calls for module in modules),
        'quantization': quantization,
        'scales_resident_cuda': bool(selected_scales) and all(s.device.type == 'cuda' for s in selected_scales),
        'scale_devices': sorted({s.device.type for s in selected_scales}),
        'last_forward_padded_m': sorted({module.last_forward_padded_m for module in modules
                                         if module.last_forward_padded_m is not None}),
    }
