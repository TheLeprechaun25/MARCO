import numpy as np
import torch


class Memory:
    def __init__(self, state_dim, memory_aggr, memory_size=10000, n_memories=1, device='cpu'):
        """
        Memory that saves State-Action pairs and returns the average value of the k-nearest neighbours of a state
        Key: Solution state of the problem
        Value: One-hot encoding of the performed action
        """
        self.state_dim = state_dim
        self.memory_size = memory_size
        self.memory_aggr = memory_aggr
        self.n_memories = n_memories # number of memories to use. If 1, use a single shared memory for all the states
        self.device = device

        # Initialize memories and index
        self.state_memories = [torch.zeros((0, self.state_dim)) for _ in range(self.n_memories)]
        self.action_memories = [torch.zeros((0, self.state_dim, 2)) for _ in range(self.n_memories)]

        self.used_memory = 0

    def save_in_memory(self, state, action):
        batch_size = state.shape[0]
        batch_range = torch.arange(batch_size)

        # Get the current values in solutions
        cur_values = state[batch_range, action]

        # Double the actions so that we can distinguish between 0->1 and 1->0
        double_action = action.clone()
        double_action[cur_values == 1] += self.state_dim

        # Perform one-hot encoding
        one_hot_actions = torch.nn.functional.one_hot(double_action, num_classes=2*self.state_dim).float()
        one_hot_actions = one_hot_actions.view(batch_size, 2, self.state_dim)
        one_hot_actions = one_hot_actions.transpose(1, 2).contiguous().view(batch_size, self.state_dim, 2)

        #one_hot_actions = one_hot_actions.cpu()
        #state = state.cpu()

        for idx in range(self.n_memories):
            # If memory is full, remove the oldest state
            if len(self.state_memories[idx]) >= self.memory_size:
                if self.n_memories == 1:
                    self.state_memories[idx] = torch.roll(self.state_memories[idx], -batch_size, dims=0)
                    self.state_memories[idx][-batch_size:] = state
                    self.action_memories[idx] = torch.roll(self.action_memories[idx], -batch_size, dims=0)
                    self.action_memories[idx][-batch_size:] = one_hot_actions
                else:
                    self.state_memories[idx] = torch.roll(self.state_memories[idx], -1, dims=0)
                    self.state_memories[idx][-1] = state[idx]
                    self.action_memories[idx] = torch.roll(self.action_memories[idx], -1, dims=0)
                    self.action_memories[idx][-1] = one_hot_actions[idx].unsqueeze(0)
            else:
                if self.n_memories == 1:
                    self.state_memories[idx] = torch.vstack([self.state_memories[idx], state])
                    self.action_memories[idx] = torch.vstack([self.action_memories[idx], one_hot_actions])
                else:
                    self.state_memories[idx] = torch.vstack([self.state_memories[idx], state[idx]])
                    self.action_memories[idx] = torch.vstack([self.action_memories[idx], one_hot_actions[idx].unsqueeze(0)])

        if self.n_memories == 1:
            self.used_memory += batch_size
        else:
            self.used_memory += 1

    def get_knn(self, state, k):
        # Get k nearest states
        batch_size = state.shape[0]
        state = state.float()
        k = self.used_memory if k > self.used_memory else k

        nearest_actions = torch.zeros((batch_size, self.state_dim, 2))

        avg_similarity = np.zeros(batch_size)
        max_similarity = np.zeros(batch_size)
        revisited = torch.zeros(batch_size)
        for idx in range(self.n_memories):
            if self.n_memories != 1:  # Multiple memories
                inner_products = torch.mm(state[idx:idx+1], self.state_memories[idx].t())  # Result is pomo_size * used_memory
                similarity, indices = torch.topk(inner_products, k, largest=True, sorted=True)

                #similarity, indices = self.index[idx].search(state[idx:idx+1], k)
                # similarity.shape = (1, k) and indices.shape = (1, k)

                revisited[idx] = (similarity == self.state_dim).sum()
                avg_similarity[idx] = torch.mean(similarity)
                max_similarity[idx] = similarity[:, 0]

                nearest_acts = self.action_memories[idx][indices.flatten(), :].reshape(indices.shape + (self.state_dim, 2))
                # nearest_acts.shape = (1, k, state_dim, 2)
                # Aggregate among k neighbors: Weighted based on similarity.
                if self.memory_aggr == 'sum':  # No weighting
                    nearest_actions[idx] = torch.sum(nearest_acts, dim=1)
                elif self.memory_aggr == 'linear':  # Linear weighted sum
                    # similarity from [-N, N] --> [0, N] --> [0, 1]
                    sim = (similarity + self.state_dim) / (2 * self.state_dim)
                    nearest_actions[idx] = torch.sum(nearest_acts * sim[:, :, None, None], dim=1)
                elif self.memory_aggr == 'exp':  # Exponential weighted sum
                    sim = (similarity + self.state_dim) / (2 * self.state_dim)
                    nearest_actions[idx] = torch.sum(nearest_acts * (torch.exp(torch.log(torch.tensor(2)) * (sim[:, :, None, None])) - 1), dim=1)
                    # nearest_actions.shape = (batch_size, state_dim, 2)
            else:  # Single memory
                inner_products = torch.mm(state, self.state_memories[idx].t())  # Result is pomo_size * used_memory
                similarity, indices = torch.topk(inner_products, k, largest=True, sorted=True)

                #similarity, indices = self.index[idx].search(state, k)
                # similarity.shape = (batch_size, k) and indices.shape = (batch_size, k)

                revisited = (similarity == self.state_dim).sum(axis=1)
                avg_similarity = torch.mean(similarity, dim=1)
                max_similarity = similarity[:, 0]

                nearest_acts = self.action_memories[idx][indices, :].reshape(indices.shape + (self.state_dim, 2))
                # nearest_acts.shape = (batch_size, k, state_dim, 2)
                # Aggregate among k neighbors: Weighted based on similarity.
                if self.memory_aggr == 'sum':  # No weighting
                    nearest_actions = torch.sum(nearest_acts, dim=1)
                elif self.memory_aggr == 'linear':  # Linear weighted sum
                    # similarity from [-N, N] --> [0, N] --> [0, 1]
                    sim = (similarity + self.state_dim) / (2 * self.state_dim)
                    nearest_actions = torch.sum(nearest_acts * sim[:, :, None, None], dim=1)
                elif self.memory_aggr == 'exp':  # Exponential weighted sum
                    sim = (similarity + self.state_dim) / (2 * self.state_dim)
                    nearest_actions = torch.sum(nearest_acts * (torch.exp(torch.log(torch.tensor(2)) * sim[:, :, None, None]) - 1), dim=1)
                    # nearest_actions.shape = (batch_size, state_dim, 2)

        return nearest_actions, revisited, avg_similarity, max_similarity


class OperationMemory:
    def __init__(self, mem_type, state_dim, batch_size, memory_size=10000, device='cpu'):
        self.mem_type = mem_type
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.batch_range = torch.arange(batch_size)
        self.memory_size = memory_size
        self.device = device

        self.last_changed = -1 * torch.ones(batch_size, 2*state_dim)
        self.state_memories = [torch.zeros((0, self.state_dim)) for _ in range(batch_size)]

        self.used_memory = 0

    def save_in_memory(self, state, action):
        batch_size = state.shape[0]
        batch_range = torch.arange(batch_size)

        # Get the current values in solutions
        cur_values = state[batch_range, action]

        # Double the actions so that we can distinguish between 0->1 and 1->0
        double_action = action.clone()
        double_action[cur_values == 1] += self.state_dim

        double_action = double_action.cpu()
        state = state.cpu()

        for idx in range(self.batch_size):
            # If memory is full, remove the oldest state
            if len(self.state_memories[idx]) >= self.memory_size:
                self.state_memories[idx] = torch.roll(self.state_memories[idx], -1, dims=0)
                self.state_memories[idx][-1] = state[idx]
            else:
                self.state_memories[idx] = torch.vstack([self.state_memories[idx], state[idx]])

        # Update last changed
        self.last_changed[self.last_changed != -1] += 1
        self.last_changed[self.batch_range, double_action] = 1

        # Update used memory
        self.used_memory += 1

    def get_knn(self, state, k):
        # K must be used_memory or less
        k = self.used_memory if k > self.used_memory else k

        # Initialize tensors
        revisited = torch.zeros(self.batch_size)
        avg_similarity = np.zeros(self.batch_size)
        max_similarity = np.zeros(self.batch_size)

        # State to float
        state = state.float()

        # For each batch element, get the k nearest neighbors
        for idx in range(self.batch_size):
            # Get similarities and indices of the k nearest neighbors
            inner_products = torch.mm(state[idx:idx + 1], self.state_memories[idx].t())  # Result is pomo_size * used_memory
            similarity, indices = torch.topk(inner_products, k, largest=True, sorted=True)

            # Check how many times the same state has been visited
            revisited[idx] = (similarity == self.state_dim).sum()

            # Compute average and max similarity
            avg_similarity[idx] = torch.mean(similarity)
            max_similarity[idx] = similarity[:, 0]

        mem_info = None
        if self.mem_type == 'op_based':  # Gather information about the last times each operation has been used
            mem_info = self.last_changed.clone().view(self.batch_size, 2, self.state_dim)
            mem_info = mem_info.transpose(1, 2).contiguous().view(self.batch_size, self.state_dim, 2)
            mem_info = mem_info.to(self.device)

        return mem_info, revisited, avg_similarity, max_similarity


def select_memory(memory_type, mem_aggr, state_dim, batch_size, testing, device):
    if memory_type in ['op_based', 'none']:
        memory = OperationMemory(mem_type=memory_type,
                                 state_dim=state_dim,
                                 batch_size=batch_size,
                                 device=device)
    else:
        if memory_type == 'shared' and testing:
            n_memories = 1
        else:  # 'individual' or training
            n_memories = batch_size

        memory = Memory(state_dim=state_dim,
                        memory_aggr=mem_aggr,
                        n_memories=n_memories,
                        device=device)
    return memory
