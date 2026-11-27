"""SPLOT research reference models."""
from .dense import DenseConfig, DenseLM, ntp_loss
from .recurrent import RecurrentConfig, RecurrentLM, RecurrentCache, RecurrentOutput
from .mtp import MTPLM, MTPOutput, Objective, training_objective
from .speculative import ModelRunner, Prefix, Result, generate
from .gdn import GatedDeltaNet
from .hybrid import HybridLM
from .moe import SparseFFN, balance_loss, routing_summary
from .sparse import SPLOTLM, SparseOutput
