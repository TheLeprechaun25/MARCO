import torch
from torch import nn
from nets.layers import GTLayer, Activation


class Model(nn.Module):
    def __init__(self, **model_params):
        super(Model, self).__init__()
        self.model_params = model_params
        self.tanh_clipping = model_params['tanh_clipping']
        self.hidden_dim = model_params['hidden_dim']
        self.n_heads = model_params['n_heads']

        # Input Node Encoding
        node_dim = 1
        self.sol_embedding = nn.Embedding(2, model_params['hidden_dim'])

        if model_params['use_mem']:
            self.mem_linear = nn.Linear(2, model_params['hidden_dim'], bias=model_params['bias'])
            node_dim += 1

        self.in_mlp = nn.Sequential(
            nn.Linear(node_dim*model_params['hidden_dim'], model_params['hidden_dim'], bias=model_params['bias']),
            Activation(model_params['activation']),
            nn.Dropout(model_params['dropout']),
        )

        # GNN layers
        shared_weights = model_params['shared_weights']
        if shared_weights:
            encoder_l = GTLayer(**model_params)
            self.encoder_layers = nn.ModuleList([encoder_l for _ in range(model_params['n_layers'])])
        else:
            self.encoder_layers = nn.ModuleList([GTLayer(**model_params) for _ in range(model_params['n_layers'])])

        self.edge_embeddings = nn.ModuleList([nn.Embedding(2, 2 * self.n_heads) for _ in range(model_params['n_layers'])])

        self.out_linear = nn.Sequential(
            nn.Linear(model_params['hidden_dim'], 1, bias=model_params['bias'])
        )
        # For testing
        self.e1 = []
        self.e2 = []

    def pre_forward(self, state):
        # Compute edge features just once per episode, since they are fixed.
        _, n = state.ising_solutions.size()

        # Edges
        self.e1 = []
        self.e2 = []
        e = state.adj_matrix.clone()
        for w_e in self.edge_embeddings:
            e1, e2 = w_e(e).split(self.n_heads, dim=-1)
            e1 = e1.transpose(2, 3).transpose(1, 2)  # (B, nh, T, T)
            e2 = e2.transpose(2, 3).transpose(1, 2)  # (B, nh, T, T)
            self.e1.append(e1)
            self.e2.append(e2)

    def forward(self, state, testing=False):
        if testing and self.model_params['mixed_precision'] and state.mem_info is not None:
            mem_info = state.mem_info.half()
        else:
            mem_info = state.mem_info

        # Node Embeddings: solutions, masks, mem_info, eig_vecs
        sol_h = self.sol_embedding(state.binary_solutions.clone())
        h = sol_h
        if self.model_params['use_mem']:
            mem_h = self.mem_linear(mem_info)
            h = torch.cat((h, mem_h), dim=-1)

        h = self.in_mlp(h)

        # GNN layers
        for idx, layer in enumerate(self.encoder_layers):
            if testing:
                h = layer(h, self.e1[idx], self.e2[idx])
            else:
                e = state.adj_matrix.clone()
                e1, e2 = self.edge_embeddings[idx](e).split(self.n_heads, dim=3)
                e1 = e1.transpose(2, 3).transpose(1, 2)  # (B, nh, T, T)
                e2 = e2.transpose(2, 3).transpose(1, 2)  # (B, nh, T, T)
                h = layer(h, e1, e2)

        # Output
        out = self.out_linear(h).squeeze(-1)

        return out
