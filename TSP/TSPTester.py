import torch
import pickle
from logging import getLogger
from env.TSPEnv import TSPEnv as Env
from nets.TSPModel import TSPModel as Model
from utils import *


class TSPTester:
    def __init__(self, env_params, model_params, tester_params):

        # Save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.marco = self.env_params['memory_type'] != 'none'

        # Result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # Device
        device = torch.device('cuda') if (tester_params['use_cuda'] and torch.cuda.is_available()) else torch.device('cpu')
        self.device = device
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Print number of model params
        self.logger.info('Number of Model Parameters: {}'.format(sum([p.numel() for p in self.model.parameters()])))

        # Restore model weights
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Restore marco weights
        if self.marco:
            marco_fullname = './result/marco/checkpoint-marco-11.pt'
            checkpoint = torch.load(marco_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Compile
        if self.tester_params['compile']:
            self.compiled_model = torch.compile(self.model)

        # Load test data
        self.test_graphs = {}
        self.opt_test_fitness = {}
        for test_size, test_batch_size in zip(self.tester_params['eval_sizes'], self.tester_params['eval_n_instances']):
            path = f"../data/tsp/tsp{test_size}_test_concorde.pkl"
            graphs = pickle.load(open(path, 'rb'))
            coord_m = np.array(graphs[:test_batch_size])
            self.test_graphs[test_size] = torch.from_numpy(coord_m).to(device).float()
            path = f"../data/tsp/tsp{test_size}_test_concorde_costs.pkl"
            opt_score = pickle.load(open(path, 'rb'))
            self.opt_test_fitness[test_size] = torch.tensor(opt_score[:test_batch_size])

        # Time Estimator
        self.time_estimator = TimeEstimator()

    def run(self):
        for test_size, test_batch_size, pomo_size in zip(self.tester_params['eval_sizes'], self.tester_params['eval_n_instances'], self.tester_params['pomo_sizes']):
            self.logger.info('\n=================================================================')
            self.logger.info("Evaluating model on {} instances of size: {}. Pomo: {}".format(test_batch_size, test_size, pomo_size))
            eval_results = self._test_one_size(test_size, test_batch_size, pomo_size)
            avg_score, avg_gap, aug_score, aug_gap, tot_time, time_per_instance = eval_results
            self.logger.info("Results model on {} instances of size: {}. Pomo: {}".format(test_batch_size, test_size, pomo_size))
            avg_opt_score = self.opt_test_fitness[test_size].mean().item()
            self.logger.info("No aug Score: {:.3f} ({:.2f})%, Aug Score: {:.3f} ({:.2f})%, Time: {} ({} per instance). Opt score: {:.3f}".format(avg_score, avg_gap, aug_score, aug_gap, tot_time, time_per_instance, avg_opt_score))

    @torch.no_grad()
    def _test_one_size(self, size, n_instances, pomo_size):
        self.time_estimator.reset()

        self.env = Env(**self.env_params)
        self.env.problem_size = size
        self.env.pomo_size = pomo_size

        coords = self.test_graphs[size]
        opt_scores = self.opt_test_fitness[size]
        self.model.eval()

        n_samples = self.tester_params['n_samples']
        aug_scores = torch.zeros((n_instances, n_samples))
        no_aug_scores = torch.zeros((n_instances, n_samples))
        aug_gaps = torch.zeros((n_instances, n_samples))
        no_aug_gaps = torch.zeros((n_instances, n_samples))

        for i in range(n_instances):
            # Load Problems
            self.env.load_problems(1, coords[i:i+1], self.tester_params['aug_factor'])
            self.env.initialize_memory(testing=True, device=self.device)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            for s in range(n_samples):
                # POMO Rollout
                deterministic = self.marco or (s == 0)
                use_memory = self.marco and (s > 0)
                # TODO: make always deterministic but with memory
                R = self.rollout(deterministic, use_memory)
                reward = R['reward']
                #punishment = R['punishment']

                # Obtain results
                aug_reward = reward.reshape(self.tester_params['aug_factor'], self.env.pomo_size)
                max_pomo_reward, _ = aug_reward.max(dim=1)  # get best results from pomo
                no_aug_score = -max_pomo_reward[0].float().mean()  # negative sign to make positive value
                max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
                aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value
                no_aug_gap = 100 * (no_aug_score - opt_scores[i]) / opt_scores[i]
                aug_gap = 100 * (aug_score - opt_scores[i]) / opt_scores[i]

                # Update Scores
                no_aug_scores[i, s] = no_aug_score.item()
                aug_scores[i, s] = aug_score.item()
                no_aug_gaps[i, s] = no_aug_gap.item()
                aug_gaps[i, s] = aug_gap.item()

                """analyse_solutions = False
                if analyse_solutions:
                    # Check how many repeated solutions we got
                    all_solutions = self.env.step_state.edge_solution.reshape(-1, size, size)
                    all_solutions = all_solutions + all_solutions.transpose(1, 2) # make symmetric
                    all_solutions = all_solutions.reshape(-1, size * size).cpu().numpy()
                    unique_solutions, counts = np.unique(all_solutions, axis=0, return_counts=True)
                    repeated_solutions = unique_solutions[counts > 1]
                    repeated_counts = counts[counts > 1]
                    total_repeated_solutions = sum(repeated_counts) - len(repeated_counts)"""

                # Logs
                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(i*n_samples + s + 1, n_instances*n_samples)
                self.logger.info("I:{}/{} S:{}/{}, Elapsed[{}], Remain[{}], Score:{:.3f} ({:.2f})%, aug:{:.3f} ({:.2f})%".format(
                    i+1, n_instances, s+1, n_samples, elapsed_time_str, remain_time_str, no_aug_score, no_aug_gap, aug_score, aug_gap))

                # Reset Env
                self.env.reset()

        total_time, time_per_instance = self.time_estimator.get_total_time(n_instances*n_samples)

        avg_score = no_aug_scores.mean().item()
        avg_gap = no_aug_gaps.mean().item()
        best_score = aug_scores.min(1).values.mean().item()
        best_gap = aug_gaps.min(1).values.mean().item()
        return avg_score, avg_gap, best_score, best_gap, total_time, time_per_instance


    def rollout(self, deterministic, use_memory):
        state, R, done = self.env.pre_step()
        while not done:
            # Get actions
            selected, _ = self.model(state, deterministic, use_memory)

            # Step the environment
            state, R, done = self.env.step(selected)

        return R
