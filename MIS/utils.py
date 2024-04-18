import random
import string
import numpy as np
import networkx as nx
from pathlib import Path
import pickle
import torch


def load_eval_data(graph_type, n_graphs):
    test_graphs = []
    if graph_type == 'er700800':
        for i in range(n_graphs):
            graph_path = f"../data/er_700_800/ER_700_800_0.15_{i}.gpickle"
            g = pickle.load(open(graph_path, 'rb'))
            test_graphs.append(torch.tensor(nx.to_numpy_array(g), dtype=torch.float32).to("cpu"))

        bl_values = pickle.load(open(f"results/greedy_results/er_700_800.pkl", 'rb'))
        bl_values = list(bl_values['cut'])
        bl_values = bl_values[:n_graphs]

    elif graph_type == 'rb200300':
        nx_rb_graphs = read_rb_dataset(f"../data/rb200-300")
        for g in nx_rb_graphs[:n_graphs]:
            test_graphs.append(torch.tensor(nx.to_numpy_array(g), dtype=torch.float32).to("cpu"))

        bl_values = pickle.load(open(f"results/greedy_results/rb_200_300.pkl", 'rb'))
        bl_values = list(bl_values['cut'])
        bl_values = bl_values[:n_graphs]

    elif graph_type == 'rb8001200':
        nx_rb_graphs = read_rb_dataset(f"../data/rb800-1200")
        for g in nx_rb_graphs[:n_graphs]:
            test_graphs.append(torch.tensor(nx.to_numpy_array(g), dtype=torch.float32).to("cpu"))

        bl_values = pickle.load(open(f"results/greedy_results/rb_800_1200.pkl", 'rb'))
        bl_values = list(bl_values['cut'])
        bl_values = bl_values[:n_graphs]

    else:
        test_size = int(graph_type[2:])
        test_batch_path = f"../data/er_test/ER_N{test_size}_100graphs.pkl"
        test_graphs = torch.tensor(np.array(pickle.load(open(test_batch_path, 'rb'))))[:n_graphs]

        bl_values = None

    return test_graphs, bl_values


def read_graph(graph_path):
    with open(graph_path, 'rb') as f:
        g = pickle.load(f)
    return g


def read_rb_dataset(data_dir, size=None):
    data_dir = Path(data_dir)
    graph_paths = sorted(list(data_dir.rglob("*.gpickle")))
    if size is not None:
        assert size > 0
        graph_paths = graph_paths[:size]

    graphs = []
    for graph_path in graph_paths:
        g = read_graph(graph_path)
        graphs.append(g)

    return graphs


def generate_word(length):
    VOWELS = "aeiou"
    CONSONANTS = "".join(set(string.ascii_lowercase) - set(VOWELS))
    word = ""
    for i in range(length):
        if i % 2 == 0:
            word += random.choice(CONSONANTS)
        else:
            word += random.choice(VOWELS)
    return word
