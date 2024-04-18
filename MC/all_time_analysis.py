import pickle
import time
import torch
from nets.model import Model
from env.Env import Env
from options.eval_options import get_options
from utils import load_eval_data


class Evaluator:
    def __init__(self, _env_params, _memory_params, _model_params, _eval_params):
        self.env_params = _env_params
        self.memory_params = _memory_params
        self.model_params = _model_params
        self.eval_params = _eval_params

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.device = device
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)

        self.env = Env(self.env_params, self.memory_params, device)
        self.env.max_iterations = self.eval_params['max_steps']
        self.env.patience = self.eval_params['patience']

        self.model = Model(**self.model_params).to(device)

        # Restore
        self.model.load_state_dict(torch.load(self.eval_params['model_load_path'])['model_state_dict'])

        if self.eval_params['verbose']:
            print(f'Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

        # Mixed precision
        if self.model_params['mixed_precision']:
            self.model = self.model.half()

        # Compile
        if self.eval_params['compile']:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)

        # Load test data
        self.eval_graphs, self.bl_values = load_eval_data(self.eval_params['eval_graph_type'], self.eval_params['n_graphs'])

    def evaluate(self):
        eval_results = {}
        best_fitness, avg_fitness, best_gap, avg_gap, times, all_time_fitness = self.run_evaluation(self.eval_graphs, self.bl_values, self.eval_params['save_all_time'])
        eval_results['best_fitness'] = best_fitness
        eval_results['avg_fitness'] = avg_fitness
        eval_results['best_gap'] = best_gap
        eval_results['avg_gap'] = avg_gap
        eval_results['times'] = times
        if self.eval_params['save_all_time']:
            eval_results['all_time_fitness'] = all_time_fitness

        if self.eval_params['save_results']:
            params = f'_maxSteps{self.eval_params["max_steps"]}_nSolutions{self.eval_params["n_solutions"]}_topK{self.eval_params["topk"]}'
            pickle.dump(eval_results, open(f'results/eval_results_{self.eval_params["eval_graph_type"]}_{params}.pkl', 'wb'))

    @torch.no_grad()
    def run_evaluation(self, eval_graphs, ref_best_fitness, save_all=False):
        ref_best_fitness = torch.tensor(ref_best_fitness)
        batch_size = len(eval_graphs)
        eval_graph_type = self.eval_params['eval_graph_type']
        n_solutions = self.eval_params['n_solutions']

        if self.eval_params['verbose']:
            print(
                f"\nEvaluating with {batch_size} graphs {eval_graph_type}. {n_solutions} initializations per graph. {self.eval_params['max_steps']}*N max steps. Patience {self.env.patience}.")

        batch_best_fitness = []
        batch_avg_fitness = []
        all_fitness_values = {}
        self.model.eval()
        all_times = []
        all_fitness = []
        all_memory = []
        for b in range(batch_size):
            print(f'Graph {b}')
            start_time = time.time()
            n = len(eval_graphs[b])
            self.env.max_iterations = self.eval_params['max_steps'] * n
            state, done = self.env.reset(n_solutions, eval_graphs[b].unsqueeze(0).to(self.device), solution_seed=42)
            self.model.pre_forward(state)
            step = 0
            times = []
            fitness = []
            memory = []
            while not done:
                logits = self.model(state, testing=True).squeeze(-1)

                topk_actions = torch.topk(logits, self.eval_params['topk'], dim=1).indices  # (n_solutions, topk)
                for k in range(self.eval_params['topk']):
                    actions = topk_actions[:, k]

                    # Perform the action
                    state, R, done = self.env.step(actions)
                    step += 1

                cur_best_fitness = self.env.best_fitness_values.max().item()
                cur_time = time.time() - start_time
                mem = (self.env.memory.state_memories[0].element_size() * self.env.memory.state_memories[0].nelement()) / 1024 / 1024
                times.append(cur_time)
                fitness.append(cur_best_fitness)
                memory.append(mem)

            all_times.append(times)
            all_fitness.append(fitness)
            all_memory.append(memory)

        # Save fitness, time and memory
        pickle.dump(all_fitness, open(f'results/eval_fitness_{self.eval_params["eval_graph_type"]}.pkl', 'wb'))
        pickle.dump(all_times, open(f'results/eval_times_{self.eval_params["eval_graph_type"]}.pkl', 'wb'))
        pickle.dump(all_memory, open(f'results/eval_memory_{self.eval_params["eval_graph_type"]}.pkl', 'wb'))

        # Plot
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np
        import seaborn as sns
        import pandas as pd

        # Numpy
        fitness = np.array(all_fitness)
        times = np.array(all_times)
        memory = np.array(all_memory)

        # Pandas
        fitness_df = pd.DataFrame(fitness)
        times_df = pd.DataFrame(times)
        memory_df = pd.DataFrame(memory)

        # Gather
        all_df = pd.DataFrame()
        all_df['fitness'] = fitness_df.mean()
        all_df['time'] = times_df.mean()
        all_df['memory'] = memory_df.mean()
        all_df['fitness_std'] = fitness_df.std()
        all_df['memory_std'] = memory_df.std()


        plt.figure(figsize=(12, 7))  # Adjust figure size for larger font
        plt.rcParams.update({'font.size': 19})  # Increasing font size
        ax = sns.lineplot(data=all_df['fitness'], color='blue', label='Fitness', marker='o', markevery=3)
        ax.set_ylabel('Fitness', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        if self.env.n == 500:
            opt = 10912.33
        elif self.env.n == 200:
            opt = 1903.33
        elif self.env.n == 700:
            opt = 20877.66
        else:
            raise NotImplementedError
        line_opt_fitness = plt.axhline(y=opt, color='black', linestyle='--')

        ax2 = ax.twinx()
        sns.lineplot(data=all_df['memory'], color='red', ax=ax2, label='Memory (MB)', marker='^', markevery=3)
        ax2.set_ylabel('Memory (MB)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

        ax.set_xlabel('Step')

        #plt.title('Fitness and Memory Usage Over Time')
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2 + [line_opt_fitness], labels + labels2 + ['Opt. fitness'], loc='upper left')

        # remove x legend
        ax.get_legend().remove()
        plt.tight_layout()  # Adjust layout to prevent clipping of larger text
        plt.show()

        return batch_best_fitness, batch_avg_fitness, None, None, times, all_fitness_values,


def main():
    # Get options
    env_params, memory_params, model_params, eval_params = get_options()

    model_params['use_mem'] = memory_params['memory_type'] != 'none'

    evaluator = Evaluator(env_params, memory_params, model_params, eval_params)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
