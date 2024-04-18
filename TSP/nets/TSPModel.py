import torch
import torch.nn as nn
import torch.nn.functional as F
from env.TSPEnv import Reset_State, Step_State
from nets.layers import GTLayer, reshape_by_heads, MLP_swiglu, Activation


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)

    def pre_forward(self, reset_state: Reset_State):
        self.encoded_nodes = self.encoder(reset_state)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state: Step_State, deterministic=False, use_memory=True):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        problem_size = state.ninf_mask.size(-1)

        if state.current_node is None:  # First node
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            if pomo_size > problem_size: # Add more initial nodes between 0 and problem_size-1
                selected = selected % problem_size

            prob = torch.ones(size=(batch_size, pomo_size))
            encoded_first_node = self.encoded_nodes.gather(dim=1, index=selected[:, :, None].expand(batch_size, pomo_size, self.embedding_dim))
            # shape: (batch, pomo, embedding)

            self.decoder.set_q1(encoded_first_node)

        else:
            encoded_last_node = self.encoded_nodes.gather(dim=1, index=state.current_node[:, :, None].expand(batch_size, pomo_size, self.embedding_dim))
            # shape: (batch, pomo, embedding)
            if self.model_params['use_memory']:
                # Gather the mem info given by the current_node
                edge_mem_last_node = state.edge_memory.gather(dim=2, index=state.current_node[:, :, None, None].expand(batch_size, pomo_size, 1, problem_size))

                logits = self.decoder(encoded_last_node, edge_mem_last_node.squeeze(2), ninf_mask=state.ninf_mask, use_memory=use_memory)
            else:
                logits = self.decoder(encoded_last_node, None, ninf_mask=state.ninf_mask, use_memory=False)

            # shape: (batch, pomo, problem)

            if not deterministic:  # Sample
                probs = F.softmax(logits, dim=2)
                # shape: (batch, pomo, problem)

                selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                # shape: (batch, pomo)

                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                # shape: (batch, pomo)

            else:
                selected = logits.argmax(dim=2)
                # shape: (batch, pomo)

                prob = None

        return selected, prob


class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.coord_embedding = nn.Linear(2, embedding_dim, bias=False)
        self.dist_embedding = nn.Linear(1, embedding_dim, bias=False)

        self.layers = nn.ModuleList([GTLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data: Reset_State):
        h = self.coord_embedding(data.coords)
        #h = torch.ones(data.coords.shape[0], data.coords.shape[1], self.model_params['embedding_dim'])
        # shape: (batch, problem, embedding)

        e = self.dist_embedding(data.dist.unsqueeze(-1))
        # shape: (batch, problem, problem, embedding)

        for layer in self.layers:
            h = layer(h, e)

        return h


class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.sqrt_embedding_dim = self.model_params['embedding_dim'] ** (1 / 2)
        self.head_num = self.model_params['head_num']
        qkv_dim = self.embedding_dim // self.head_num

        self.W_q_first = nn.Linear(self.embedding_dim, self.head_num * qkv_dim, bias=False)
        self.W_q_last = nn.Linear(self.embedding_dim, self.head_num * qkv_dim, bias=False)
        self.W_kv = nn.Linear(self.embedding_dim, 2 * self.head_num * qkv_dim, bias=False)

        if self.model_params['use_memory']:
            self.mem_FF = MLP_swiglu(2, 1, self.embedding_dim, **model_params)

        self.multi_head_combine = nn.Linear(self.head_num * qkv_dim, self.embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)

        k, v = self.W_kv(encoded_nodes).split(self.embedding_dim, dim=-1)
        self.k = reshape_by_heads(k, head_num=self.head_num)
        self.v = reshape_by_heads(v, head_num=self.head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        self.q_first = reshape_by_heads(self.W_q_first(encoded_q1), head_num=self.head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, mem_last_node, ninf_mask, use_memory=True):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        batch_size, pomo_size = encoded_last_node.size(0), encoded_last_node.size(1)
        problem_size = self.k.size(2)

        #  Embedding of the last visited node
        q_last = reshape_by_heads(self.W_q_last(encoded_last_node), head_num=self.head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        # Sum the first and last visited.
        q = self.q_first + q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        # Multi-Head Attention
        # q   (batch, head_num, pomo, qkv_dim)
        # k   (batch, head_num, problem, qkv_dim)
        # v   (batch, head_num, problem, qkv_dim)
        # mem (batch, pomo, problem)
        out_concat = torch.nn.functional.scaled_dot_product_attention(q, self.k, self.v, attn_mask=ninf_mask[:, None, :, :].expand(batch_size, self.head_num, pomo_size, problem_size))
        out_concat = out_concat.transpose(1, 2).reshape(batch_size, pomo_size, self.embedding_dim)
        # shape: (batch, pomo, embedding)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        if use_memory:
            score = torch.stack((score, mem_last_node), dim=-1)
            score = self.mem_FF(score).squeeze(-1)

        score_scaled = score / self.sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = self.model_params['logit_clipping'] * torch.tanh(score_scaled)

        logits = score_clipped + ninf_mask

        return logits

