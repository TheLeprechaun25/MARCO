import random
import string
import numpy as np
from pathlib import Path
import pickle
import networkx as nx
import torch


def load_test_data(test_sizes, test_batch_sizes):
    test_graphs = []
    bl_values = []
    for test_size, test_batch_size in zip(test_sizes, test_batch_sizes):
        test_batch_path = f"../data/er_test/ER_N{test_size}_100graphs.pkl"
        test_graphs.append(torch.tensor(np.array(pickle.load(open(test_batch_path, 'rb'))))[:test_batch_size])

        model_size = test_size if test_size <= 200 else 200
        results = pickle.load(open(f"../data/er_test/results_eco_dqn/results_ER_N{test_size}_100graphs_model{model_size}.pkl", 'rb'))

        bl_values.append(torch.tensor(results['cut'])[:test_batch_size])

    return test_graphs, bl_values


def load_eval_data(graph_type, n_graphs):
    test_graphs = []
    if graph_type == 'er700800':
        for i in range(n_graphs):
            graph_path = f"../data/er_700_800/ER_700_800_0.15_{i}.gpickle"
            g = pickle.load(open(graph_path, 'rb'))
            test_graphs.append(torch.tensor(nx.to_numpy_array(g), dtype=torch.float32).to("cpu"))

        eco_dqn_fitness = pickle.load(open(f"results/eco_dqn_results/results_er700800_model200.pkl", 'rb'))
        eco_dqn_fitness = list(eco_dqn_fitness['cut'])
        bl_values = eco_dqn_fitness[:n_graphs]

    elif graph_type == 'rb200300':
        nx_rb_graphs = read_rb_dataset(f"../data/rb200-300")
        for g in nx_rb_graphs[:n_graphs]:
            test_graphs.append(torch.tensor(nx.to_numpy_array(g), dtype=torch.float32).to("cpu"))

        eco_dqn_fitness = pickle.load(open(f"results/eco_dqn_results/results_rb200300_model200.pkl", 'rb'))
        eco_dqn_fitness = list(eco_dqn_fitness['cut'])
        bl_values = eco_dqn_fitness[:n_graphs]

    elif graph_type == 'rb8001200':
        nx_rb_graphs = read_rb_dataset(f"../data/rb800-1200")
        for g in nx_rb_graphs[:n_graphs]:
            test_graphs.append(torch.tensor(nx.to_numpy_array(g), dtype=torch.float32).to("cpu"))

        eco_dqn_fitness = pickle.load(open(f"results/eco_dqn_results/results_rb8001200_model200.pkl", 'rb'))
        eco_dqn_fitness = list(eco_dqn_fitness['cut'])
        bl_values = eco_dqn_fitness[:n_graphs]

    else:
        test_size = int(graph_type[2:])
        test_batch_path = f"../data/er_test/ER_N{test_size}_100graphs.pkl"
        test_graphs = torch.tensor(np.array(pickle.load(open(test_batch_path, 'rb'))))[:n_graphs]
        model_size = test_size if test_size <= 200 else 200
        results = pickle.load(open(f"../data/er_test/results_eco_dqn/results_ER_N{test_size}_100graphs_model{model_size}.pkl", 'rb'))

        bl_values = torch.tensor(results['cut'])[:n_graphs]

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
