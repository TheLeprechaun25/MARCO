import math
import torch
from torch import nn
import torch.nn.functional as F


class GTLayer(nn.Module):
    def __init__(self, **model_params):
        super(GTLayer, self).__init__()
        self.n_heads = model_params['n_heads']
        self.hidden_dim = model_params['hidden_dim']
        self.head_dim = self.hidden_dim // self.n_heads

        self.W_h = nn.Linear(self.hidden_dim, 3 * self.hidden_dim, bias=model_params['bias'])

        self.norm1 = Norm(**model_params)
        self.norm2 = Norm(**model_params)

        self.mlp = MLP(**model_params)

    def forward(self, h, e1, e2):
        batch_size, n_nodes, _ = h.shape
        h_in = h.clone()

        # Initial normalization
        h = self.norm1(h)

        # Linear transformation
        q, k, v = self.W_h(h).split(self.hidden_dim, dim=2)
        k = k.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att + e1
        att = F.softmax(att, dim=-1)
        att = att * e2
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Residual connection
        y = y.transpose(1, 2).reshape(batch_size, n_nodes, self.hidden_dim)
        y = y + h_in

        # Normalization and MLP
        out = self.mlp(self.norm2(y))

        # Final residual connection
        return out + y


class MLP(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.c_fc = nn.Linear(model_params['hidden_dim'], 4 * model_params['hidden_dim'], bias=model_params['bias'])
        self.c_proj = nn.Linear(4 * model_params['hidden_dim'], model_params['hidden_dim'], bias=model_params['bias'])
        self.dropout = nn.Dropout(model_params['dropout'])
        self.act = Activation(model_params['activation'])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


def rms_norm(x, weight=None, eps=1e-05):
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return output * weight
    return output


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-05, weight=True, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=dtype, device=device))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        return rms_norm(x.float(), self.weight, self.eps).to(dtype=x.dtype)


class Norm(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.normalization = model_params['normalization']
        hidden_dim = model_params['hidden_dim']
        if self.normalization == 'layer':
            self.norm = nn.LayerNorm(hidden_dim)
        elif self.normalization == 'batch':
            self.norm = nn.BatchNorm1d(hidden_dim, affine=True, track_running_stats=False)
        elif self.normalization == 'rms':
            self.norm = RMSNorm(hidden_dim)
        elif self.normalization == 'instance':
            self.norm = nn.InstanceNorm1d(hidden_dim, affine=True, track_running_stats=False)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.normalization in ['instance', 'batch']:
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.norm(x)
        return x


class Activation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.act(x)
