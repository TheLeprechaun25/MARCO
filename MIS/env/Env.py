from dataclasses import dataclass
import networkx as nx
import torch
import random
from env.memory import select_memory
from env.generators import RandomErdosRenyiGraphGenerator as Generator


@dataclass
class State:
    adj_matrix: torch.Tensor
    solutions: torch.Tensor
    masks: torch.Tensor
    mem_info: torch.Tensor


class Env:
    def __init__(self, env_params, memory_params, device):
        self.env_params = env_params
        self.memory_params = memory_params

        # Env Params
        self.n = env_params['problem_size']
        self.initialization = env_params['initialization']
        self.device = device
        self.float_dtype = torch.float32
        self.int_dtype = torch.int32
        self.testing = False

        self.batch_size = None
        self.patience = None
        self.patience_count = 0
        self.iteration = 0
        self.max_iterations = None

        # Env Modules
        self.generator = Generator(p_connection=0.15)
        self.memory = None
        self.state = None

        # Env Records
        self.fitness = None
        self.reward = None
        self.best_fitness_values = None
        self.best_mean_fitness = None
        self.non_improving_steps = None

    def reset(self, batch_size, graphs=None, init_solution_seed=None):
        """
        :param batch_size: (int) Num of instances in each batch (for training) or num of initializations (for testing)
        :param graphs: (torch.Tensor) Adjacency matrix of the test graph to be solved (for testing). Shape: (batch_size, n, n)
        :param init_solution_seed: (int) Seed for initializing solutions (for testing)
        """
        self.batch_size = batch_size

        if graphs is None:
            self.testing = False
            adj_matrix = self.generator.generate_graphs(self.n, batch_size)
            adj_matrix = torch.tensor(adj_matrix, dtype=self.float_dtype).to(self.device)
        else:
            self.testing = True
            self.n = graphs.shape[1]
            adj_matrix = graphs.unsqueeze(0).to(self.device)

        # Initialize memory
        self.memory = select_memory(memory_type=self.memory_params['memory_type'],
                                    mem_aggr=self.memory_params['mem_aggr'],
                                    state_dim=self.n,
                                    batch_size=self.batch_size,
                                    testing=self.testing,
                                    device=self.device)

        # Initialize solutions
        solutions = self.generate_batch_of_solutions(adj_matrix, init_solution_seed)

        masks = self.create_action_mask(adj_matrix, solutions)

        # Initialize memory info
        mem_info = torch.zeros(batch_size, self.n, 2, dtype=self.float_dtype).to(self.device) if self.memory_params['memory_type'] != 'none' else None

        self.state = State(adj_matrix=adj_matrix, solutions=solutions, masks=masks, mem_info=mem_info)

        # Initialize environment records
        self.fitness = self.compute_fitness()
        self.best_fitness_values = self.fitness.clone()
        self.best_mean_fitness = self.fitness.mean().item()
        self.non_improving_steps = torch.zeros(batch_size, dtype=self.int_dtype).to(self.device)
        self.iteration = 0

        return self.state, False

    def step(self, action):
        """
        :param action: (torch.Tensor) Action(s) to be executed. Shape: (batch_size,)
        """
        self.iteration += 1
        # Save state(key) and action(value) in memory
        sol = 2 * self.state.solutions - 1
        self.memory.save_in_memory(sol, action)

        # Update solutions based on actions
        self.state.solutions[torch.arange(self.batch_size), action] = 1 - self.state.solutions[torch.arange(self.batch_size), action]

        # Update masks
        self.state.masks = self.create_action_mask(self.state.adj_matrix, self.state.solutions)

        self.fitness = self.compute_fitness()

        # Get k-nearest neighbors of current solutions
        sol = 2 * self.state.solutions - 1
        mem_info, revisited, avg_sim, max_sim = self.memory.get_knn(sol, k=self.memory_params['k'])

        self.state.mem_info = mem_info

        # COMPUTE REWARDS (only during training)
        R = {}
        if not self.testing:
            # Main reward: improvement in fitness
            self.reward = (self.fitness - self.best_fitness_values)
            R['Fitness improvement'] = self.reward.mean().item()
            # make zero the negative --> only positive rewards when improving the best found
            self.reward[self.reward < 0] = 0

            # Normalize reward
            # self.reward /= self.n

            # Add a small positive reward to those who are in local maxima and revisited states
            R['Re-Visited'] = revisited.float().mean().item()

            revisited_idx = revisited != 0

            # Punish for visiting the same solution again
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
        :return: (torch.Tensor) Fitness of the current solutions. Shape: (batch_size,)
        """
        #is_independent_set = self.is_independent_set_vectorized(self.state.solutions, self.state.adj_matrix)
        #assert torch.all(is_independent_set)

        # Calculate fitness as the number of nodes in the set
        fitness = torch.sum(self.state.solutions, dim=1)

        return fitness

    def create_action_mask(self, adj_matrix, solutions):
        """
        :param adj_matrix: (torch.Tensor) Adjacency matrix of the graph. Shape: (batch_size, n, n)
        :param solutions: (torch.Tensor) Solutions of the graph. Shape: (batch_size, n)
        :return: (torch.Tensor) Action mask. Shape: (batch_size, n)
        """
        if self.testing:
            # Reshape and transpose solutions for matrix multiplication
            reshaped_solutions = solutions.unsqueeze(-1).transpose(1, 2)  # Shape: (batch_size, 1, n)

            # Perform matrix multiplication adj_matrix will be automatically broadcasted to match the batch size
            adjacent_mask = torch.matmul(reshaped_solutions, adj_matrix[0]).squeeze(1)  # Shape: (batch_size, n)

            # Nodes that can't be added (any adjacent node is in the set)
            action_mask = (adjacent_mask > 0) & (solutions == 0)

        else:
            # Expand solutions to match the adjacency matrix shape
            expanded_solutions = solutions.unsqueeze(2)  # Shape: (batch_size, n, 1)

            # Use batch matrix multiplication to find if any adjacent node is in the set
            adjacent_mask = torch.bmm(adj_matrix, expanded_solutions).squeeze(2)  # Shape: (batch_size, n)

            # Nodes that can't be added (any adjacent node is in the set)
            action_mask = (adjacent_mask > 0) & (solutions == 0)

        return action_mask.to(self.device)

    def generate_batch_of_solutions(self, adj_matrix, seed=None):
        """
        :param adj_matrix: (torch.Tensor) Adjacency matrix of the graph. Shape: (batch_size, n, n)
        :param seed: (int) Seed for initializing solutions
        :return: (torch.Tensor) Solutions of the graph. Shape: (batch_size, n)
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        solutions = torch.zeros(self.batch_size, self.n, dtype=torch.float32).to(self.device)

        if self.initialization == 'greedy':

            # Precompute the neighbors for each node in each graph
            if self.testing:
                neighbors = [torch.nonzero(adj_matrix[0], as_tuple=False)]
            else:
                neighbors = [torch.nonzero(adj_matrix[b], as_tuple=False) for b in range(self.batch_size)]

            for b in range(self.batch_size):
                d = 0 if self.testing else b

                available_nodes = set(range(self.n))
                node_neighbors = neighbors[d]

                while available_nodes:
                    node = random.sample(list(available_nodes), 1)[0]

                    # Vectorized check for independent set condition
                    if not torch.any((adj_matrix[d, node] == 1) & (solutions[b] == 1)):
                        solutions[b, node] = 1
                        # Remove the node and its neighbors
                        neighbor_nodes = node_neighbors[node_neighbors[:, 0] == node][:, 1]
                        available_nodes -= {node, *neighbor_nodes.tolist()}
                    else:
                        available_nodes.remove(node)

        elif self.initialization == 'greedy2':
            # Precompute the neighbors for each node in each graph
            if self.testing:
                neighbors = [torch.nonzero(adj_matrix[0], as_tuple=False)]
            else:
                neighbors = [torch.nonzero(adj_matrix[b], as_tuple=False) for b in range(self.batch_size)]

            for b in range(self.batch_size):
                d = 0 if self.testing else b

                if b==0:
                    G = nx.from_numpy_array(adj_matrix[d].cpu().numpy())
                    while 1:
                        min_val = 2 * self.n
                        node = None
                        for n in G.nodes():
                            if G.degree(n) < min_val and G.degree(n) != 0:
                                min_val = G.degree(n)
                                node = n
                        if node is None:  # stop if all the nodes are in the independent set
                            break
                        solutions[b, node] = 1
                        G.remove_nodes_from([n for n in G.neighbors(node)])
                else:
                    available_nodes = set(range(self.n))
                    node_neighbors = neighbors[d]
                    while available_nodes:
                        node = random.sample(list(available_nodes), 1)[0]

                        # Vectorized check for independent set condition
                        if not torch.any((adj_matrix[d, node] == 1) & (solutions[b] == 1)):
                            solutions[b, node] = 1
                            # Remove the node and its neighbors
                            neighbor_nodes = node_neighbors[node_neighbors[:, 0] == node][:, 1]
                            available_nodes -= {node, *neighbor_nodes.tolist()}
                        else:
                            available_nodes.remove(node)

        return solutions

    @staticmethod
    def is_independent_set_vectorized(solutions, adj_matrix):
        """
        :param solutions: (torch.Tensor) Solution of the graph. Shape: (batch_size, n)
        :param adj_matrix: (torch.Tensor) Adjacency matrix of the graph. Shape: (batch_size, n, n)
        :return: (torch.Tensor) Whether the solutions are an independent set. Shape: (batch_size,)
        """
        # Expand solutions to match the adjacency matrix shape for batch matrix multiplication
        expanded_solutions = solutions.unsqueeze(2)  # Shape: (batch_size, n, 1)

        # Use batch matrix multiplication to find if any adjacent node is in the set
        adjacent_solution = torch.bmm(adj_matrix, expanded_solutions).squeeze(2)  # Shape: (batch_size, n)

        # Check for each solution if there are no adjacent nodes in the set
        # A node is part of the solution if it's marked as 1 in 'solutions'
        # It's an independent set if none of these nodes have adjacent nodes also in the set
        is_independent_set = torch.all((adjacent_solution <= 1) | (solutions == 0), dim=1)

        return is_independent_set

