import torch
import random
from env.memory import select_memory
from env.generators import RandomErdosRenyiGraphGenerator as Generator
from env.utils import BinaryState, compute_immediate_rewards


class Env:
    def __init__(self, env_params, memory_params, device):
        self.env_params = env_params
        self.memory_params = memory_params

        # Env Params
        self.n = env_params['problem_size']
        self.device = device
        self.float_dtype = torch.float32
        self.int_dtype = torch.int32
        self.testing = False
        self.batch_size = None
        self.batch_range = None
        self.patience = None
        self.max_iterations = None
        self.iteration = 0

        # Env Modules
        self.generator = Generator(p_connection=0.15)
        self.memory = None
        self.state = None

        # Env Records
        self.immediate_rewards = None
        self.fitness = None
        self.reward = None
        self.best_fitness_values = None
        self.best_mean_fitness = None
        self.non_improving_steps = None

    def reset(self, batch_size, test_graph=None, solution_seed=None):
        """
        :param batch_size: (int) Num of instances in each batch (for training) or num of initializations (for testing)
        :param test_graph
        :param solution_seed
        """
        self.batch_size = batch_size
        self.batch_range = torch.arange(batch_size)

        # Generate batch of graphs if test graph is None (training)
        if test_graph is None:
            self.testing = False
            adj_matrix = self.generator.generate_graphs(self.n, batch_size)
            adj_matrix = torch.tensor(adj_matrix, dtype=self.int_dtype).to(self.device)
        else:
            self.testing = True
            self.n = test_graph.shape[1]
            adj_matrix = test_graph.clone().int().to(self.device)

        # Initialize memory
        self.memory = select_memory(memory_type=self.memory_params['memory_type'],
                                    mem_aggr=self.memory_params['mem_aggr'],
                                    state_dim=self.n,
                                    batch_size=self.batch_size,
                                    testing=self.testing,
                                    device=self.device)

        # Initialize memory info
        mem_info = torch.zeros(batch_size, self.n, 2, dtype=self.float_dtype).to(self.device) if self.memory_params['memory_type'] != 'none' else None

        # Initialize solutions
        binary_solutions = self.generate_batch_of_solutions(solution_seed)
        ising_solutions = 2 * binary_solutions - 1

        # Initialize immediate rewards
        self.immediate_rewards = compute_immediate_rewards(adj_matrix, ising_solutions, self.testing)

        # Initialize state
        self.state = BinaryState(adj_matrix=adj_matrix, ising_solutions=ising_solutions, binary_solutions=binary_solutions, mem_info=mem_info)

        # Initialize environment records
        self.fitness = self.compute_fitness()
        self.best_fitness_values = self.fitness.clone()
        self.best_mean_fitness = self.fitness.mean().item()
        self.non_improving_steps = torch.zeros(batch_size, dtype=self.int_dtype).to(self.device)
        self.iteration = 0

        return self.state, False

    def step(self, action):
        """
        :param action: (torch.Tensor) Action(s) to be executed. Shape: (batch_size)
        """
        self.iteration += 1

        # Save state(key) and action(value) in memory
        self.memory.save_in_memory(self.state.ising_solutions.clone(), action)

        # Update solutions based on actions
        self.state.binary_solutions[torch.arange(self.batch_size), action] = 1 - self.state.binary_solutions[torch.arange(self.batch_size), action]
        self.state.ising_solutions[torch.arange(self.batch_size), action] = - self.state.ising_solutions[torch.arange(self.batch_size), action]

        # Update fitness with immediate rewards of selected actions
        self.fitness += self.immediate_rewards[self.batch_range, action]

        # Compute immediate rewards for next iteration
        self.immediate_rewards = compute_immediate_rewards(self.state.adj_matrix, self.state.ising_solutions, self.testing)

        # Get k-nearest neighbors of current solutions
        mem_info, revisited, avg_sim, max_sim = self.memory.get_knn(self.state.ising_solutions.clone(), k=self.memory_params['k'])

        self.state.mem_info = mem_info

        # COMPUTE REWARDS (only during training)
        R = {}

        # Main reward: improvement in fitness
        self.reward = (self.fitness - self.best_fitness_values)
        R['Fitness improvement'] = self.reward.mean().item()

        # make zero the negative --> only positive rewards when improving the best found
        # TODO slightly negative rewards when not improving the best found?
        self.reward[self.reward < 0] = 0

        # Punish for visiting the same solution again
        R['Re-Visited'] = revisited.float().mean().item()
        revisited_idx = revisited != 0
        if (self.memory_params['memory_type'] != 'none') and (not self.testing):
            self.reward[revisited_idx] -= self.env_params['revisit_punishment'] * revisited[revisited_idx]

        R['Reward'] = self.reward
        R['Avg similarity'] = avg_sim
        R['Max similarity'] = max_sim

        # Check if episode is done
        done = (self.non_improving_steps.sum() >= (self.patience * self.batch_size)) or (self.iteration >= self.max_iterations)

        # Update fitness records
        self.non_improving_steps += 1
        best_idx = self.fitness > self.best_fitness_values
        self.best_fitness_values[best_idx] = self.fitness[best_idx]
        self.non_improving_steps[best_idx] = 0

        return self.state, R, done

    def compute_fitness(self):
        """
        :return: (torch.Tensor) Fitness of the current solutions. Shape: (batch_size)
        """
        fitness = torch.zeros(self.batch_size)

        for b in range(self.batch_size):  # TODO: Vectorize this
            if self.testing:
                fitness[b] = (1 / 4) * torch.sum(torch.mul(self.state.adj_matrix[0], 1 - torch.outer(self.state.ising_solutions[b], self.state.ising_solutions[b])))
            else:
                fitness[b] = (1 / 4) * torch.sum(torch.mul(self.state.adj_matrix[b], 1 - torch.outer(self.state.ising_solutions[b], self.state.ising_solutions[b])))
        return fitness.float()

    def generate_batch_of_solutions(self, seed=None):
        """
        :param seed: (int) Seed for initializing solutions
        :return: (torch.Tensor) Solutions of the graph. Shape: (batch_size, n)
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        binary_solutions = torch.randint(0, 2, (self.batch_size, self.n), dtype=self.int_dtype).to(self.device)

        return binary_solutions
