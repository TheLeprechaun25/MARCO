import numpy as np
import networkx as nx


class RandomErdosRenyiGraphGenerator:
    def __init__(self, p_connection=0.15):
        self.p_connection = p_connection

    def generate_graphs(self, n_nodes, batch_size, seed=None):
        if seed is not None:
            np.random.seed(seed)

        batch = np.zeros((batch_size, n_nodes, n_nodes), dtype=np.int32)
        for b in range(batch_size):
            g = nx.erdos_renyi_graph(n_nodes, self.p_connection)
            adj = nx.to_numpy_array(g, dtype=np.int32)

            # No self-connections (this modifies adj in-place).
            np.fill_diagonal(adj, 0)

            batch[b] = adj

        return batch

