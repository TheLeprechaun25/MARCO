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
            print(f"\nEvaluating with {batch_size} graphs {eval_graph_type}. {n_solutions} initializations per graph. {self.eval_params['max_steps']}*N max steps. Patience {self.env.patience}.")

        batch_best_fitness = []
        batch_avg_fitness = []
        all_fitness_values = {}
        self.model.eval()
        times = []
        for b in range(batch_size):
            n = len(eval_graphs[b])
            self.env.max_iterations = self.eval_params['max_steps'] * n
            state, done = self.env.reset(n_solutions, eval_graphs[b].unsqueeze(0).to(self.device), solution_seed=42)
            self.model.pre_forward(state)
            step = 0
            start_time = time.time()
            while not done:
                logits = self.model(state, testing=True).squeeze(-1)

                topk_actions = torch.topk(logits, self.eval_params['topk'], dim=1).indices # (n_solutions, topk)
                for k in range(self.eval_params['topk']):
                    actions = topk_actions[:, k]

                    # Perform the action
                    state, R, done = self.env.step(actions)
                    step += 1

                    if save_all:
                        cur_best_fitness = self.env.best_fitness_values.max().item()
                        cur_time = time.time() - start_time
                        if b not in all_fitness_values:
                            all_fitness_values[b] = [[cur_best_fitness], [cur_time]]
                        else:
                            all_fitness_values[b][0].append(cur_best_fitness)
                            all_fitness_values[b][1].append(cur_time)

            times.append(time.time() - start_time)
            best_fitness = self.env.best_fitness_values
            batch_best_fitness.append(best_fitness.max().item())
            batch_avg_fitness.append(best_fitness.mean().item())
            if self.eval_params['verbose']:
                if ref_best_fitness[b] != 0:
                    best_gap = (100 * (ref_best_fitness[b] - batch_best_fitness[b]) / ref_best_fitness[b])
                    avg_gap = (100 * (ref_best_fitness[b] - batch_avg_fitness[b]) / ref_best_fitness[b])
                else:
                    best_gap = -1
                    avg_gap = -1
                print(f"Instance {b + 1}/{batch_size}) {time.time()-start_time:.2f}s. Best fitness: {batch_best_fitness[b]}, Best gap: {best_gap:.2f}%, Average gap: {avg_gap:.2f}%. Final step: {step}")

        # End of evaluation
        tensor_best = torch.tensor(batch_best_fitness)
        tensor_avg = torch.tensor(batch_avg_fitness)
        best_gap = (100 * (ref_best_fitness - tensor_best) / ref_best_fitness)
        avg_gap = (100 * (ref_best_fitness - tensor_avg) / ref_best_fitness)
        if self.eval_params['verbose']:
            total_time = sum(times)
            print(f"Best fitness: {tensor_best.mean().item():.3f} (Gap {best_gap.mean().item():.2f}%). Average fitness: {tensor_avg.mean().item():.3f} (Gap: {avg_gap.mean().item():.2f}%). Tot time: {total_time:.2f}s ({total_time / batch_size:.2f}s/instance).")

        return batch_best_fitness, batch_avg_fitness, best_gap, avg_gap, times, all_fitness_values,


def main():
    # Get options
    env_params, memory_params, model_params, eval_params = get_options()

    model_params['use_mem'] = memory_params['memory_type'] != 'none'

    evaluator = Evaluator(env_params, memory_params, model_params, eval_params)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
