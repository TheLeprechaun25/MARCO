import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GTLayer(nn.Module):
    def __init__(self, **model_params):
        super(GTLayer, self).__init__()

        self.n_heads = model_params['head_num']
        hidden_dim = model_params['embedding_dim']
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // self.n_heads
        model_params['bias'] = False
        self.W_h = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.W_e = nn.Linear(hidden_dim, 2 * self.n_heads, bias=False)
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.norm1 = Norm(**model_params)
        self.norm2 = Norm(**model_params)

        if model_params['activation'] == 'swiglu':
            self.mlp = MLP_swiglu(hidden_dim, hidden_dim, ff_hidden_dim, **model_params)
        else:
            self.mlp = MLP(hidden_dim, hidden_dim, ff_hidden_dim, **model_params)

    def forward(self, h, e):
        """
        Training: h (B, N, d)  edge_weights (B, N, N, 1)
        Inference: h (B, N, d)  edge_weights (1, N, N, 1)
        """
        batch_size, n_nodes, _ = h.shape
        h_in = h.clone()
        # Initial normalization
        h = self.norm1(h)

        # Linear transformation
        q, k, v = self.W_h(h).split(self.hidden_dim, dim=2)
        k = k.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        e1, e2 = self.W_e(e).split(self.n_heads, dim=3)
        e1 = e1.transpose(2, 3).transpose(1, 2) # (B, nh, T, T)
        e2 = e2.transpose(2, 3).transpose(1, 2) # (B, nh, T, T)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att + e1
        att = F.softmax(att, dim=-1)

        att = att * e2

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # residual connection
        y = y.transpose(1, 2).reshape(batch_size, n_nodes, self.hidden_dim)

        # Add, Normalization and MLP
        out = self.mlp(self.norm2(y + h_in))

        # Final residual connection
        return out + y


def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


class MHAbias(nn.Module):
    def __init__(self, model_params):
        super(MHAbias, self).__init__()
        self.n_heads = model_params['head_num']
        hidden_dim = model_params['embedding_dim']
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // self.n_heads
        self.scale_factor = 1 / math.sqrt(self.head_dim)

    def forward(self, q, k, v, bias, mask=None):
        # q shape: (batch, head_num, pomo, key_dim)
        # k,v shape: (batch, head_num, problem, key_dim)
        # bias.shape: (batch, pomo, problem, head_num)
        # rank3_ninf_mask.shape: (batch, pomo, problem)

        batch_s, head_num, pomo, key_dim = q.size()
        input_s = k.size(2)

        score = q @ k.transpose(-2, -1) * self.scale_factor
        # shape: (batch, head_num, pomo, problem)

        if mask is not None:
            score = score + mask[:, None, :, :].expand(batch_s, head_num, pomo, input_s)

        bias = bias.permute(0, 3, 1, 2)
        # shape: (batch, head_num, pomo, problem)

        score = nn.Softmax(dim=3)(score + bias)
        # shape: (batch, head_num, pomo, problem)

        score = score @ v
        # shape: (batch, head_num, n, key_dim)

        score = score.transpose(1, 2)
        # shape: (batch, n, head_num, key_dim)

        score = score.reshape(batch_s, pomo, head_num * key_dim)
        # shape: (batch, n, head_num*key_dim)

        return score


def multi_head_attention_with_bias_old(q, k, v, bias, rank3_ninf_mask=None):
    # q shape: (batch, head_num, pomo, key_dim)
    # k,v shape: (batch, head_num, problem, key_dim)
    # bias.shape: (batch, pomo, problem, head_num)
    # rank3_ninf_mask.shape: (batch, pomo, problem)

    batch_s, head_num, pomo, key_dim = q.size()
    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, pomo, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, pomo, input_s)

    bias = bias.permute(0, 3, 1, 2)
    # shape: (batch, head_num, pomo, problem)

    weights = nn.Softmax(dim=3)(score_scaled + bias)
    # shape: (batch, head_num, pomo, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, pomo, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


def multi_head_attention_with_bias(q, k, v, bias, rank3_ninf_mask=None):
    # q shape: (batch, head_num, pomo, key_dim)
    # k,v shape: (batch, head_num, problem, key_dim)
    # bias.shape: (batch, pomo, problem, head_num)
    # rank3_ninf_mask.shape: (batch, pomo, problem)

    batch_s, head_num, pomo, key_dim = q.size()
    input_s = k.size(2)

    scale_factor = 1 / math.sqrt(key_dim)
    score = q @ k.transpose(-2, -1) * scale_factor
    # shape: (batch, head_num, pomo, problem)

    if rank3_ninf_mask is not None:
        score = score + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, pomo, input_s)

    bias = bias.permute(0, 3, 1, 2)
    # shape: (batch, head_num, pomo, problem)

    score = nn.Softmax(dim=3)(score + bias)
    # shape: (batch, head_num, pomo, problem)

    score = score @ v
    # shape: (batch, head_num, n, key_dim)

    score = score.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    score = score.reshape(batch_s, pomo, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)
    return score


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, **model_params):
        super().__init__()
        dropout = model_params['dropout']
        bias = model_params['bias']
        activation = model_params['activation']
        self.c_fc = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.c_proj = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.act = Activation(activation)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class MLP_swiglu(nn.Module):
    def __init__(self,  in_dim, out_dim, hidden_dim, **model_params):
        super().__init__()
        dropout = model_params['dropout']
        bias = model_params['bias']

        self.c_fc = nn.Linear(in_dim, 2 * hidden_dim, bias=bias)
        self.c_proj = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x1, x2 = x.chunk(2, dim=-1)

        x = F.silu(x1) * x2
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


def rms_norm(x, weight=None, eps: float = 1e-05):
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return output * weight
    return output


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-05, weight=True, dtype=None, device=None):
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
        self.normalization = 'rms'
        hidden_dim = model_params['embedding_dim']
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





