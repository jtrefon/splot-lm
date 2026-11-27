"""Semantic Muon/AdamW partition and adapter for the synthetic harness.

Native Muon is used for hidden 2-D linear matrices only. The partition is
derived from the actual owning module and parameter role, never from ndim or
shape heuristics alone. Unknown parameters fail closed.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import torch
from torch import nn

from splot.dense import GQA, RMSNorm, Block
from splot.gdn import GatedDeltaNet
from splot.mtp import MTPLM
from splot.moe import SparseFFN, Expert
from splot.recurrent import RecurrentLM


PARTITION_VERSION = 1
# torch/optim/_muon.py: _zeropower_via_newtonschulz casts the gradient to
# bfloat16 for the Newton-Schulz iterations and returns that BF16 matrix.
NS_INTERMEDIATE_DTYPE = 'bfloat16'
MUON_IMPLEMENTATION = 'torch.optim.Muon'
MUON_DEFAULTS = {
    'lr': .003, 'weight_decay': 0., 'momentum': .95, 'nesterov': True,
    'ns_coefficients': (3.4445, -4.775, 2.0315), 'ns_steps': 5,
    'eps': 1e-7, 'adjust_lr_fn': 'match_rms_adamw',
}
ADAMW_DEFAULTS = {'betas': (.9, .999), 'eps': 1e-8}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _classify(model, name):
    """Return (group, role) for one canonical parameter name."""
    parent_path, local = name.rsplit('.', 1) if '.' in name else ('', name)
    parent = model.get_submodule(parent_path) if parent_path else model
    if isinstance(parent, nn.Embedding):
        return 'adamw', 'tied_embedding'
    if isinstance(parent, RMSNorm):
        return 'adamw', 'rmsnorm'
    if isinstance(parent, GatedDeltaNet):
        if local in ('conv', 'log_A', 'dt_bias'):
            return 'adamw', 'gdn_conv_or_scalar_state'
        raise ValueError(f'unclassified GatedDeltaNet parameter: {name}')
    if isinstance(parent, RecurrentLM):
        if local == 'depth':
            return 'adamw', 'depth_embedding'
        raise ValueError(f'unclassified recurrent parameter: {name}')
    if isinstance(parent, nn.Linear):
        if '.' in parent_path:
            owner_path, module_name = parent_path.rsplit('.', 1)
        else:
            owner_path, module_name = '', parent_path
        owner = model.get_submodule(owner_path) if owner_path else model
        if module_name in ('q', 'k', 'v', 'o'):
            if not isinstance(owner, (GQA, GatedDeltaNet)):
                raise ValueError(f'attention projection outside GQA/GDN: {name}')
            return 'muon', 'attention_projection'
        if module_name == 'z':
            if not isinstance(owner, GatedDeltaNet):
                raise ValueError(f'GDN z projection outside GDN: {name}')
            return 'muon', 'gdn_z_projection'
        if module_name == 'output_gate':
            if not isinstance(owner, GQA):
                raise ValueError(f'output gate outside GQA: {name}')
            return 'muon', 'gqa_output_gate'
        if module_name in ('a', 'b'):
            if not isinstance(owner, GatedDeltaNet):
                raise ValueError(f'GDN scalar gate outside GDN: {name}')
            return 'adamw', 'gdn_scalar_gate'
        if module_name == 'router':
            if not isinstance(owner, SparseFFN):
                raise ValueError(f'router outside SparseFFN: {name}')
            return 'adamw', 'moe_router'
        if module_name in ('gate', 'up', 'down'):
            if not isinstance(owner, (Block, Expert)):
                raise ValueError(f'FFN matrix outside Block/Expert: {name}')
            return 'muon', 'ffn_matrix'
        if module_name == 'offset_projection':
            if not isinstance(owner, MTPLM):
                raise ValueError(f'offset projection outside MTPLM: {name}')
            return 'muon', 'mtp_offset_projection'
        raise ValueError(f'unclassified linear parameter: {name}')
    raise ValueError(f'unclassified parameter owner {type(parent).__name__}: {name}')


def _owner_info(model, name):
    parent_path, _ = name.rsplit('.', 1) if '.' in name else ('', name)
    parent = model.get_submodule(parent_path) if parent_path else model
    if isinstance(parent, nn.Linear):
        if '.' in parent_path:
            owner_path, _ = parent_path.rsplit('.', 1)
        else:
            owner_path = ''
        owner = model.get_submodule(owner_path) if owner_path else model
        return type(owner).__name__, owner_path
    return type(parent).__name__, parent_path


@dataclass
class Partition:
    muon: list
    adamw: list
    metadata: dict


def build_semantic_partition(model):
    """Build a stable, exhaustive, disjoint semantic partition."""
    entries = {}
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        key = id(parameter)
        if key in entries:
            aliases.setdefault(key, []).append(name)
            continue
        group, role = _classify(model, name)
        owner, module = _owner_info(model, name)
        entries[key] = {
            'name': name, 'shape': list(parameter.shape), 'numel': parameter.numel(),
            'role': role, 'owner': owner, 'module': module,
            'group': group, 'parameter': parameter,
        }
    for key, names in aliases.items():
        entries[key]['aliases'] = sorted(names)
    ordered = sorted(entries.values(), key=lambda entry: entry['name'])
    for entry in ordered:
        if entry['group'] == 'muon' and entry['parameter'].ndim != 2:
            raise ValueError(f'Muon parameter is not 2-D: {entry["name"]}')
    muon = [entry['parameter'] for entry in ordered if entry['group'] == 'muon']
    adamw = [entry['parameter'] for entry in ordered if entry['group'] == 'adamw']
    inventory = sum(parameter.numel() for parameter in model.parameters())
    unique = sum(entry['numel'] for entry in ordered)
    if unique != inventory:
        raise ValueError(f'partition inventory mismatch: {unique} != {inventory}')
    if len({id(parameter) for parameter in muon + adamw}) != len(muon) + len(adamw):
        raise ValueError('partition contains duplicate parameter identities')
    if any(entry['group'] == 'muon' and entry['parameter'].ndim != 2 for entry in ordered):
        raise ValueError('Muon group contains a non-2-D parameter')
    metadata = {
        'version': PARTITION_VERSION,
        'inventory_numel': inventory,
        'parameter_count': len(ordered),
        'native_implementation': MUON_IMPLEMENTATION,
        'ns_intermediate_dtype': NS_INTERMEDIATE_DTYPE,
        'groups': {
            'muon': {
                'count': len(muon), 'numel': sum(parameter.numel() for parameter in muon),
                'entries': [{key: entry[key] for key in
                             ('name', 'shape', 'numel', 'role', 'owner', 'module')}
                            | ({'aliases': entry['aliases']} if 'aliases' in entry else {})
                            for entry in ordered if entry['group'] == 'muon'],
            },
            'adamw': {
                'count': len(adamw), 'numel': sum(parameter.numel() for parameter in adamw),
                'entries': [{key: entry[key] for key in
                             ('name', 'shape', 'numel', 'role', 'owner', 'module')}
                            | ({'aliases': entry['aliases']} if 'aliases' in entry else {})
                            for entry in ordered if entry['group'] == 'adamw'],
            },
        },
    }
    metadata['plan_sha256'] = _digest(metadata)
    return Partition(muon, adamw, metadata)


class MuonAdamW:
    """Native Muon plus AdamW adapter with a validated nested checkpoint state."""
    adapter_version = 1

    def __init__(self, model, *, muon_lr=.003, muon_weight_decay=0.,
                 muon_momentum=.95, muon_nesterov=True,
                 muon_ns_coefficients=MUON_DEFAULTS['ns_coefficients'], muon_ns_steps=5,
                 muon_eps=1e-7, muon_adjust_lr_fn='match_rms_adamw',
                 adamw_lr=.003, adamw_weight_decay=0., adamw_betas=(.9, .999),
                 adamw_eps=1e-8):
        self.partition = build_semantic_partition(model)
        self.settings = {
            'muon': {'lr': muon_lr, 'weight_decay': muon_weight_decay,
                     'momentum': muon_momentum, 'nesterov': muon_nesterov,
                     'ns_coefficients': list(muon_ns_coefficients),
                     'ns_steps': muon_ns_steps, 'eps': muon_eps,
                     'adjust_lr_fn': muon_adjust_lr_fn},
            'adamw': {'lr': adamw_lr, 'weight_decay': adamw_weight_decay,
                      'betas': list(adamw_betas), 'eps': adamw_eps},
            'ns_intermediate_dtype': NS_INTERMEDIATE_DTYPE,
        }
        self.muon = torch.optim.Muon(
            self.partition.muon, lr=muon_lr, weight_decay=muon_weight_decay,
            momentum=muon_momentum, nesterov=muon_nesterov,
            ns_coefficients=tuple(muon_ns_coefficients), ns_steps=muon_ns_steps,
            eps=muon_eps, adjust_lr_fn=muon_adjust_lr_fn)
        self.adamw = torch.optim.AdamW(
            self.partition.adamw, lr=adamw_lr, weight_decay=adamw_weight_decay,
            betas=tuple(adamw_betas), eps=adamw_eps)

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        # Muon first, then AdamW; every gradient was checked before this call.
        self.muon.step(closure=None)
        self.adamw.step(closure=closure)

    def state_dict(self):
        return {
            'adapter_version': self.adapter_version,
            'partition': self.partition.metadata,
            'settings': self.settings,
            'muon': self.muon.state_dict(),
            'adamw': self.adamw.state_dict(),
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get('adapter_version') != self.adapter_version:
            raise ValueError('optimizer adapter version or mode mismatch')
        if state.get('partition') != self.partition.metadata:
            raise ValueError('optimizer partition metadata mismatch')
        if state.get('settings') != self.settings:
            raise ValueError('optimizer settings mismatch')
        try:
            self.muon.load_state_dict(state['muon'])
            self.adamw.load_state_dict(state['adamw'])
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f'invalid MuonAdamW state: {exc}') from exc

    def _inventory(self, state):
        elements = bytes_ = 0
        for entry in state.get('state', {}).values():
            for value in entry.values():
                if torch.is_tensor(value):
                    elements += value.numel()
                    bytes_ += value.numel() * value.element_size()
        return {'elements': elements, 'bytes': bytes_}

    def state_inventory(self):
        muon = self._inventory(self.muon.state_dict())
        adamw = self._inventory(self.adamw.state_dict())
        return {'muon': muon, 'adamw': adamw,
                'total': {'elements': muon['elements'] + adamw['elements'],
                          'bytes': muon['bytes'] + adamw['bytes']}}


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _parse_adamw(spec, training):
    if spec is None:
        spec = {}
    if not isinstance(spec, dict):
        raise ValueError('optimizer.adamw must be an object')
    unknown = set(spec) - {'lr', 'weight_decay', 'betas', 'eps'}
    if unknown:
        raise ValueError(f'unknown AdamW settings: {sorted(unknown)}')
    lr = _finite(spec.get('lr', training['lr']), 'AdamW lr')
    weight_decay = _finite(spec.get('weight_decay', training['weight_decay']), 'AdamW weight_decay')
    betas = spec.get('betas', list(ADAMW_DEFAULTS['betas']))
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError('AdamW betas must contain two values')
    betas = tuple(_finite(value, 'AdamW beta') for value in betas)
    if not all(0 <= value < 1 for value in betas):
        raise ValueError('AdamW betas must be in [0, 1)')
    eps = _finite(spec.get('eps', ADAMW_DEFAULTS['eps']), 'AdamW eps')
    if lr < 0 or weight_decay < 0 or eps <= 0:
        raise ValueError('invalid AdamW hyperparameters')
    return {'lr': lr, 'weight_decay': weight_decay, 'betas': betas, 'eps': eps}


def _parse_muon(spec, training):
    if spec is None:
        spec = {}
    if not isinstance(spec, dict):
        raise ValueError('optimizer.muon must be an object')
    unknown = set(spec) - set(MUON_DEFAULTS)
    if unknown:
        raise ValueError(f'unknown Muon settings: {sorted(unknown)}')
    settings = dict(MUON_DEFAULTS)
    settings.update(spec)
    settings['lr'] = _finite(settings['lr'], 'Muon lr')
    settings['weight_decay'] = _finite(settings['weight_decay'], 'Muon weight_decay')
    settings['momentum'] = _finite(settings['momentum'], 'Muon momentum')
    settings['eps'] = _finite(settings['eps'], 'Muon eps')
    if not isinstance(settings['nesterov'], bool):
        raise ValueError('Muon nesterov must be boolean')
    if settings['lr'] < 0 or settings['weight_decay'] < 0 or not 0 <= settings['momentum'] <= 1 or settings['eps'] <= 0:
        raise ValueError('invalid Muon hyperparameters')
    coefficients = settings['ns_coefficients']
    if not isinstance(coefficients, (list, tuple)) or len(coefficients) != 3:
        raise ValueError('Muon ns_coefficients must contain three values')
    settings['ns_coefficients'] = tuple(_finite(value, 'Muon ns_coefficient') for value in coefficients)
    if type(settings['ns_steps']) is not int or not 1 <= settings['ns_steps'] < 100:
        raise ValueError('Muon ns_steps must be an integer in [1, 99]')
    if settings['adjust_lr_fn'] not in (None, 'original', 'match_rms_adamw'):
        raise ValueError('invalid Muon adjust_lr_fn')
    return settings


def build_optimizer(model, training):
    """Build the default AdamW path or the opt-in native Muon/AdamW hybrid."""
    spec = training.get('optimizer')
    if spec is None:
        opt = torch.optim.AdamW(model.parameters(), lr=training['lr'],
                                weight_decay=training['weight_decay'])
        return opt, {'name': 'adamw', 'implementation': 'torch.optim.AdamW',
                     'source': 'default', 'lr': training['lr'],
                     'weight_decay': training['weight_decay'],
                     'betas': list(ADAMW_DEFAULTS['betas']), 'eps': ADAMW_DEFAULTS['eps']}
    if isinstance(spec, str):
        if spec not in ('adamw', 'muon'):
            raise ValueError(f'unknown optimizer selector: {spec}')
        spec = {'name': spec}
    if not isinstance(spec, dict):
        raise ValueError('optimizer must be an object or adamw/muon selector')
    unknown = set(spec) - {'name', 'muon', 'adamw'}
    if unknown:
        raise ValueError(f'unknown optimizer settings: {sorted(unknown)}')
    name = spec.get('name')
    if name not in ('adamw', 'muon'):
        raise ValueError(f'unknown optimizer name: {name}')
    adamw = _parse_adamw(spec.get('adamw'), training)
    if name == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), **adamw)
        return opt, {'name': 'adamw', 'implementation': 'torch.optim.AdamW',
                     'source': 'config', 'lr': adamw['lr'],
                     'weight_decay': adamw['weight_decay'],
                     'betas': list(adamw['betas']), 'eps': adamw['eps']}
    muon = _parse_muon(spec.get('muon'), training)
    opt = MuonAdamW(
        model, muon_lr=muon['lr'], muon_weight_decay=muon['weight_decay'],
        muon_momentum=muon['momentum'], muon_nesterov=muon['nesterov'],
        muon_ns_coefficients=muon['ns_coefficients'], muon_ns_steps=muon['ns_steps'],
        muon_eps=muon['eps'], muon_adjust_lr_fn=muon['adjust_lr_fn'],
        adamw_lr=adamw['lr'], adamw_weight_decay=adamw['weight_decay'],
        adamw_betas=adamw['betas'], adamw_eps=adamw['eps'])
    return opt, {'name': 'muon', 'implementation': 'torch.optim.Muon+torch.optim.AdamW',
                 'source': 'config', 'adapter_version': opt.adapter_version,
                 'muon': opt.settings['muon'], 'adamw': opt.settings['adamw'],
                 'partition': opt.partition.metadata,
                 'ns_intermediate_dtype': NS_INTERMEDIATE_DTYPE}


def optimizer_state_inventory(opt):
    if isinstance(opt, MuonAdamW):
        return opt.state_inventory()
    state = opt.state_dict().get('state', {})
    elements = bytes_ = 0
    for entry in state.values():
        for value in entry.values():
            if torch.is_tensor(value):
                elements += value.numel()
                bytes_ += value.numel() * value.element_size()
    return {'adamw': {'elements': elements, 'bytes': bytes_},
            'total': {'elements': elements, 'bytes': bytes_}}
