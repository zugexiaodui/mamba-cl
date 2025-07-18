"""PyTorch MAMBA model."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available
from transformers.models.mamba.configuration_mamba import MambaConfig
from timm.models.layers import DropPath, drop_path
from timm.models.vision_transformer import Mlp


logger = logging.get_logger(__name__)

if is_mamba_ssm_available():
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
else:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, mamba_inner_fn)
)


class IntermReader(nn.Module):
    def __init__(self, dst_param_id: int, module_name: str, other_args: dict = {}) -> None:
        super().__init__()
        self.dst_param_id = dst_param_id
        self.module_name = module_name
        self.other_args = other_args

    def forward(self, x: Tensor):
        return x

    def extra_repr(self) -> str:
        return f"dst_param_id={self.dst_param_id}, module_name={self.module_name}"


class MambaMixer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = config.time_step_rank
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.use_out_norm = config.use_out_norm

        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.config = config

        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)

        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size*2, bias=False)

        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)

        self.use_bias = config.use_bias

        self.group_norm_size = config.group_norm_size        
        if self.group_norm_size > 0:
            # e.g., use group_norm_size = 64
            assert self.use_out_norm is False

        if self.config.use_rope:
            freq = (100**(-torch.arange(1, self.ssm_state_size+1)/self.ssm_state_size) - 100**(-1)) / (1 - 100**(-1))
            self.freq = nn.Parameter(freq)

        assert is_fast_path_available, "The fast path is not available because one" \
                " of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`" \
                " is None. Falling back to the naive implementation." \
                "To install follow https://github.com/state-spaces/mamba/#installation and" \
                " https://github.com/Dao-AILab/causal-conv1d"

    def split_parameters(self):
        self.x_proj_step = nn.Linear(self.intermediate_size, self.time_step_rank, bias=False)
        self.x_proj_B = nn.Linear(self.intermediate_size, self.ssm_state_size, bias=False)
        self.x_proj_C = nn.Linear(self.intermediate_size, self.ssm_state_size, bias=False)
        w_step, w_B, w_C = torch.split(self.x_proj.weight.data, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=0)
        self.x_proj_step.weight.data = w_step.data.clone()
        self.x_proj_B.weight.data = w_B.data.clone()
        self.x_proj_C.weight.data = w_C.data.clone()
        delattr(self, 'x_proj')

        self.interm_reader_A = IntermReader(id(self.A_log), 'interm_reader_A')
        self.interm_reader_step = IntermReader(id(self.x_proj_step.weight), 'interm_reader_step')
        self.interm_reader_B = IntermReader(id(self.x_proj_B.weight), 'interm_reader_B')
        self.interm_reader_C = IntermReader(id(self.x_proj_C.weight), 'interm_reader_C')
        self.interm_reader_out = IntermReader(id(self.out_proj.weight), 'interm_reader_out')

    def forward_ssm(self, hidden_states, gate):
        time_step = self.x_proj_step(self.interm_reader_step(hidden_states.transpose(1, 2)))
        B = self.x_proj_B(hidden_states.transpose(1, 2))
        C = self.x_proj_C(self.interm_reader_C(hidden_states.transpose(1, 2)))

        B = B.transpose(1, 2)
        C = C.transpose(1, 2)

        if self.config.use_rope:
            B, C = self.add_rope(B.transpose(1, 2), C.transpose(1, 2))
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)

        discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)
        time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None

        A = -torch.exp(self.A_log.float())
        delta = F.softplus(discrete_time_step + time_proj_bias.view(1, -1, 1)).to(discrete_time_step.dtype)

        delta = delta / self.config.delta_scale_factor

        if not self.x_proj_B.weight.requires_grad:
            self.interm_reader_A(delta)
            self.interm_reader_B(torch.matmul(delta, hidden_states.transpose(1, 2)))

        scan_outputs, ssm_state = selective_scan_fn(
            hidden_states,
            delta,
            A,
            B,
            C,
            None, 
            None, 
            None, 
            delta_softplus=False,
            return_last_state=True,
        )

        batch, dim, tokennum = scan_outputs.shape[0], scan_outputs.shape[1], scan_outputs.shape[2]
        org_type = scan_outputs.dtype

        scan_outputs = scan_outputs.view(batch, dim//self.group_norm_size, self.group_norm_size, tokennum).float()

        scan_outputs = scan_outputs * torch.rsqrt(scan_outputs.pow(2).mean(2, keepdim=True) + 1e-6)
        scan_outputs = scan_outputs.view(batch, dim, tokennum)

        scan_outputs = scan_outputs + (hidden_states * self.D.float()[None, :, None])
        scan_outputs = scan_outputs * F.silu(gate.float())
        scan_outputs = scan_outputs.to(dtype=org_type)

        return scan_outputs

    def cuda_kernels_forward(self, hidden_states: torch.Tensor):
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        hidden_states, gate = projected_states.chunk(2, dim=1)

        if causal_conv1d_fn is not None:
            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            hidden_states = causal_conv1d_fn(
                hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
            )
        else:
            hidden_states = self.conv1d(hidden_states)[:, :, :-self.conv1d.padding[0]]
            hidden_states = F.silu(hidden_states)

        scan_outputs = self.forward_ssm(hidden_states, gate)

        contextualized_states = self.out_proj(self.interm_reader_out(scan_outputs.transpose(1, 2)))

        return contextualized_states
    
    def add_rope(self, B, C):
        n_idx = torch.arange(0, B.shape[1], device=B.device, dtype=B.dtype) / self.config.rope_scale_factor
        theta = n_idx.view(-1, 1) @ self.freq.view(1, -1)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        B = self.theta_shift(B, sin, cos)
        C = self.theta_shift(C, sin, cos)

        return B, C
    
    def theta_shift(self, x, sin, cos):
        return (x * cos) + (self.rotate_every_two(x) * sin)

    def rotate_every_two(self, x):
        x1 = x[:, :, ::2]
        x2 = x[:, :, 1::2]
        x = torch.stack((-x2, x1), dim=-1)
        return x.flatten(-2)

    def forward(self, hidden_states):
        return self.cuda_kernels_forward(hidden_states)


class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MambaBlock(nn.Module):
    def __init__(self, config, layer_idx, drop_path=0., use_checkpoint=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.use_checkpoint = use_checkpoint
        self.mixer = MambaMixer(config, layer_idx=layer_idx)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _forward(self, hidden_states, drop_path_rate=0.0):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states)
        hidden_states = residual + drop_path(hidden_states, drop_prob=drop_path_rate, training=self.training)

        return hidden_states
    
    def forward(self, hidden_states, drop_path_rate=0.0):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, hidden_states, drop_path_rate, use_reentrant=False)
        else:
            return self._forward(hidden_states, drop_path_rate)