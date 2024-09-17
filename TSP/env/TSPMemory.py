import numpy as np
import torch


class Memory:
    def __init__(self, state_dim, memory_aggr, memory_size=10000, n_memories=1, device='cpu'):
        """
        Memory that saves States and returns the average value of the k-nearest neighbours of a state
        """
        self.state_dim = state_dim
        self.n = np.sqrt(state_dim).astype('int')
        self.memory_size = memory_size
        self.memory_aggr = memory_aggr
        self.n_memories = n_memories # number of memories to use. If 1, use a single shared memory for all the states
        self.device = device

        # Initialize memories and index
        self.state_memories = [torch.zeros((0, self.state_dim)) for _ in range(self.n_memories)]

        self.used_memory = 0

    def save_in_memory(self, state):
        batch_size, pomo_size = state.shape[0], state.shape[1]

        state = state.view(batch_size, pomo_size, self.state_dim).float()

        for idx in range(self.n_memories):
            # If memory is full, remove the oldest state
            if len(self.state_memories[idx]) >= self.memory_size:
                self.state_memories[idx] = torch.roll(self.state_memories[idx], -pomo_size, dims=0)
                self.state_memories[idx][-pomo_size:] = state
            else:
                self.state_memories[idx] = torch.vstack([self.state_memories[idx], state[idx]])

        self.used_memory += pomo_size

    def get_knn(self, state, step, k, return_similarity=False):
        if self.used_memory == 0:
            return None, None, None

        norm_factor = float(2*step)

        # Get k nearest states using faiss
        batch_size, pomo_size = state.shape[0], state.shape[1]
        state = state.view(batch_size, pomo_size, self.state_dim).float()

        k = self.used_memory if k > self.used_memory else k

        nearest_states = torch.zeros((batch_size, pomo_size, self.state_dim))

        avg_similarity = torch.zeros((batch_size, pomo_size))
        max_similarity = torch.zeros((batch_size, pomo_size))
        for idx in range(self.n_memories):
            inner_products = torch.mm(state[idx], self.state_memories[idx].t())  # Result dim: (pomo_size, used_memory)
            similarity, indices = torch.topk(inner_products, k, largest=True, sorted=True)
            # similarity.shape = (pomo, k) and indices.shape = (pomo, k)

            if return_similarity:
                avg_similarity[idx] = similarity.mean(1) / norm_factor
                max_similarity[idx] = similarity.max(1)[0] / norm_factor

            nearest_s = self.state_memories[idx][indices]
            # nearest_s.shape = (pomo, k, state_dim)
            # Aggregate among k neighbors: Weighted based on similarity.
            if self.memory_aggr == 'sum':  # No weighting
                nearest_states[idx] = torch.sum(nearest_s, dim=1)
            elif self.memory_aggr == 'linear':  # Linear weighted sum
                # similarity from [0, N] --> [0, 1]
                sim = similarity / norm_factor
                nearest_states[idx] = torch.sum(nearest_s * sim[:, :, None], dim=1)
                # nearest_s = (pomo, k, state_dim) and sim = (pomo, k), nearest_states = (batch, pomo, state_dim)
                # Divide by k
                nearest_states[idx] = nearest_states[idx] / k
        nearest_states = nearest_states.reshape(batch_size, pomo_size, self.n, self.n)
        return nearest_states, avg_similarity, max_similarity


def select_memory(mem_aggr, state_dim, batch_size, device):
    n_memories = batch_size

    memory = Memory(state_dim=state_dim,
                    memory_aggr=mem_aggr,
                    n_memories=n_memories,
                    device=device)
    return memory
