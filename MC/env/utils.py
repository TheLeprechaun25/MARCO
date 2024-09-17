from dataclasses import dataclass
import torch


@dataclass
class BinaryState:
    adj_matrix: torch.Tensor
    ising_solutions: torch.Tensor
    binary_solutions: torch.Tensor
    mem_info: torch.Tensor


def get_features(adj_matrix, solutions, masks, node_mem, get_edges=True):
    if node_mem is None:
        nodes = torch.cat([solutions, masks], dim=2)
    else:
        nodes = torch.cat([solutions, masks, node_mem], dim=2)

    if get_edges:
        edges = adj_matrix.unsqueeze(dim=3).float()
        return nodes, edges
    else:
        return nodes


def compute_immediate_rewards(adj, solutions, testing):
    """
    Compute the fitness change when flipping each bit
    Args:
        adj (Tensor): Adjacency matrix. Shape: [batch_size, n_nodes, n_nodes].
        solutions (Tensor): Tensor of solutions. Shape: [batch_size, n_nodes].
        testing
    Returns:
        Tensor: Immediate rewards. Shape: [batch_size, n_nodes].
    """
    if testing:  # batch_size = 1
        solutions = solutions.unsqueeze(-1).float()  # Reshape solutions to [batch_size, n_nodes, 1]
        rewards = solutions * adj.float() @ solutions
    else:
        solutions = solutions.unsqueeze(-1).float()  # Reshape solutions to [batch_size, n_nodes, 1]
        rewards = solutions * torch.bmm(adj.float(), solutions)  # Batch matrix multiplication
    return rewards.squeeze(-1)

