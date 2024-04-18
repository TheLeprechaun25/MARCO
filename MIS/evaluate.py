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
        self.env.testing = True

        self.model = Model(**self.model_params).to(device)

        if self.eval_params['verbose']:
            print(f'Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

        # Restore
        self.model.load_state_dict(torch.load(self.eval_params['model_load_path'])['model_state_dict'])

        # Compile
        if self.eval_params['compile']:
            self.model = torch.compile(self.model)

        # Load eval data
        self.eval_graphs, self.bl_values = load_eval_data(self.env_params['graph_type'], self.eval_params['n_instances'])

    def evaluate(self):
        eval_results = {}

        eval_batch_size = len(self.eval_graphs)
        best_fitness, avg_fitness, times, all_time_fitness = self.run_evaluation(self.eval_graphs, self.bl_values, eval_batch_size, self.eval_params['save_all_time'])

        eval_results['times'] = times
        eval_results['best_fitness'] = best_fitness
        eval_results['avg_fitness'] = avg_fitness
        if self.eval_params['save_all_time']:
            eval_results['all_time_fitness'] = all_time_fitness
        if self.eval_params['save_results']:
            model_name = self.eval_params['model_load_path'].split('/')[-1].split('.')[0]
            params = f'_init{self.env_params["initialization"]}_maxSteps{self.eval_params["max_steps"]}_nSolutions{self.eval_params["n_solutions"]}_topK{self.eval_params["topk"]}_memory{self.memory_params["memory_type"]}'
            pickle.dump(eval_results, open(f'results/eval_results_{self.env_params["graph_type"]}_{model_name}{params}.pkl', 'wb'))

    @torch.no_grad()
    def run_evaluation(self, eval_graphs, ref_best_fitness, eval_batch_size, save_all_fitness=False):
        n_solutions = self.eval_params['n_solutions']
        self.env.max_iterations = self.eval_params['max_steps']
        if self.eval_params['verbose']:
            print(f"\nEvaluating with {eval_batch_size} graphs ({self.env_params['graph_type']}). {n_solutions} initializations per graph. {self.eval_params['max_steps']}*N max steps. Patience {self.env.patience}.")

        batch_best_fitness = []
        batch_avg_fitness = []
        all_fitness_values = {}
        self.model.eval()
        times = []
        for b in range(eval_batch_size):
            n = len(eval_graphs[b])
            self.env.max_iterations = self.eval_params['max_steps'] * n

            state, done = self.env.reset(n_solutions, eval_graphs[b].to(self.device), init_solution_seed=42)

            step = 0
            start_time = time.time()
            while not done:
                logits = self.model(state).squeeze(-1)

                topk_actions = torch.topk(logits, self.eval_params['topk'], dim=1).indices # (n_solutions, topk)
                for k in range(self.eval_params['topk']):
                    actions = topk_actions[:, k]

                    if k > 0:
                        # Check whether all actions are valid or masked by state.masks
                        exit_loop = False
                        for s in range(n_solutions):
                            if state.masks[s, actions[s]] == 1:
                                exit_loop = True
                        if exit_loop:
                            break

                    # Perform the action
                    state, R, done = self.env.step(actions)
                    step += 1

                    if save_all_fitness:
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
                if ref_best_fitness is not None:
                    best_gap = (100 * (ref_best_fitness[b] - batch_best_fitness[b]) / ref_best_fitness[b]).item()
                    avg_gap = (100 * (ref_best_fitness[b] - batch_avg_fitness[b]) / ref_best_fitness[b]).item()
                    print(f"Instance {b + 1}/{eval_batch_size}) {time.time()-start_time:.2f}s. Best fitness: {batch_best_fitness[b]}, Best gap: {best_gap:.2f}%, Average gap: {avg_gap:.2f}%. Final step: {step}")
                else:
                    print(f"Instance {b + 1}/{eval_batch_size}) {time.time()-start_time:.2f}s. Best fitness: {batch_best_fitness[b]}. Final step: {step}")

        # End of evaluation
        tensor_best = torch.tensor(batch_best_fitness)
        tensor_avg = torch.tensor(batch_avg_fitness)

        if self.eval_params['verbose']:
            total_time = sum(times)
            if ref_best_fitness is not None:
                best_gap = (100 * (ref_best_fitness - tensor_best) / ref_best_fitness)
                avg_gap = (100 * (ref_best_fitness - tensor_avg) / ref_best_fitness)
                print(f"Best fitness: {tensor_best.mean().item():.3f} (Gap {best_gap.mean().item():.2f}%). Average fitness: {tensor_avg.mean().item():.3f} (Gap: {avg_gap.mean().item():.2f}%). Tot time: {total_time:.2f}s ({total_time / eval_batch_size:.2f}s/instance).")
            else:
                print(f"Best fitness: {tensor_best.mean().item():.3f}. Average fitness: {tensor_avg.mean().item():.3f}. Tot time: {total_time:.2f}s ({total_time / eval_batch_size:.2f}s/instance).")

        return batch_best_fitness, batch_avg_fitness, times, all_fitness_values


def main():
    # Get options
    env_params, memory_params, model_params, eval_params = get_options()

    # Set node and edge dimensions
    # Node dim
    if memory_params['memory_type'] == 'none':
        model_params['node_dim'] = 2
    else:
        model_params['node_dim'] = 4

    # Edge dim
    model_params['edge_dim'] = 1

    evaluator = Evaluator(env_params, memory_params, model_params, eval_params)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
