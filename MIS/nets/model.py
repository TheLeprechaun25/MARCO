import torch
from torch import nn
from nets.layers import GTLayer, Activation


class Model(nn.Module):
    def __init__(self, **model_params):
        super(Model, self).__init__()
        self.model_params = model_params
        self.tanh_clipping = model_params['tanh_clipping']
        bias = model_params['bias']

        # Input Node Encoding
        self.sol_embedding = nn.Embedding(2, model_params['hidden_dim'])
        self.mask_embedding = nn.Embedding(2, model_params['hidden_dim'])
        self.mem_linear = nn.Linear(2, model_params['hidden_dim'], bias=False)
        self.in_mlp = nn.Sequential(
            nn.Linear(3*model_params['hidden_dim'], model_params['hidden_dim'], bias=bias),
            Activation(model_params['activation']),
            nn.Dropout(model_params['dropout']),
        )

        # Input Edge Encoding
        self.init_edge_embed = nn.Embedding(2, model_params['hidden_dim'])

        shared_weights = False
        if shared_weights:
            encoder_l = GTLayer(**model_params)
            self.encoder_layers = nn.ModuleList([encoder_l for _ in range(model_params['n_layers'])])
        else:
            self.encoder_layers = nn.ModuleList([GTLayer(**model_params) for _ in range(model_params['n_layers'])])

        self.decoder = nn.Sequential(
            nn.Linear(model_params['hidden_dim'], model_params['hidden_dim'], bias=model_params['bias']),
            Activation(model_params['activation']),
            nn.Dropout(model_params['dropout']),
            nn.Linear(model_params['hidden_dim'], 1, bias=model_params['bias']),
        )

    def forward(self, state):
        # ENCODER
        sol_h = self.sol_embedding(state.solutions.int())
        mask_h = self.mask_embedding(state.masks.int())
        if state.mem_info is None:
            mem_h = torch.zeros_like(sol_h)
        else:
            mem_h = self.mem_linear(state.mem_info)
        h = torch.cat((sol_h, mask_h, mem_h), dim=-1)

        h = self.in_mlp(h)

        e = self.init_edge_embed(state.adj_matrix.int())

        for layer in self.encoder_layers:
            h = layer(h, e)

        # DECODER
        out = self.decoder(h)

        # mask
        out = out.squeeze(-1)
        out[state.masks] = -torch.inf
        return out
