import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW as Optimizer
from nets.model import Model
from env.Env import Env
from utils import load_test_data, generate_word
from options.train_options import get_options


class Trainer:
    def __init__(self, _env_params, _memory_params, _model_params, _optimizer_params, _trainer_params):
        self.env_params = _env_params
        self.memory_params = _memory_params
        self.model_params = _model_params
        self.optimizer_params = _optimizer_params
        self.trainer_params = _trainer_params

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.device = device
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)

        # Training environment
        self.env = Env(self.env_params, self.memory_params, device)
        self.env.patience = self.trainer_params['train_patience']
        self.env.max_iterations = self.trainer_params['max_train_steps']

        # Testing environment
        self.test_env = Env(self.env_params, self.memory_params, device)
        self.test_env.patience = self.trainer_params['test_patience']
        self.test_env.max_iterations = self.trainer_params['max_test_steps']

        # Model
        self.model = Model(**self.model_params).to(device)
        if self.trainer_params['verbose']:
            print(f'Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

        # Optimizer
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])

        # Restore weights
        if self.trainer_params['model_load']['enable']:
            path = self.trainer_params['model_load']['path']
            self.model.load_state_dict(torch.load(path)['model_state_dict'])
            self.optimizer.load_state_dict(torch.load(path)['optimizer_state_dict'])
            # update the learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.optimizer_params['optimizer']['lr']

        # Compile
        self.compile = False
        if self.compile:
            self.compiled_model = torch.compile(self.model)

        # Load test data
        test_sizes = [20, 60, 100, 200]
        test_batch_sizes = [32, 16, 8, 4]
        self.test_graphs, self.bl_values = load_test_data(test_sizes, test_batch_sizes)

    def train(self):
        # initialize fitness moving average
        best_fitness_ma = 0
        steps_until_last_save = 0
        start_time = time.time()
        for epoch in range(1, self.trainer_params['epochs'] + 1):
            if self.trainer_params['verbose']:
                print(f"\n\nEpoch {epoch}/{self.trainer_params['epochs']})")

            self.model.train()

            for episode in range(self.trainer_params['n_episodes']):
                if self.trainer_params['rand_train_sizes']:
                    new_size = np.random.randint(50, 200)

                    new_env_params = self.env_params.copy()
                    new_env_params['problem_size'] = new_size
                    self.env = Env(new_env_params, self.memory_params, self.device)
                    self.env.patience = self.trainer_params['train_patience']
                    self.env.max_iterations = self.trainer_params['max_train_steps']
                    if self.trainer_params['verbose']:
                        print(f"Using Random Curriculum Learning. Current graph size: {self.env.n}. Batch size: {self.trainer_params['batch_size']}")

                # Generate new batch
                state, done = self.env.reset(self.trainer_params['batch_size'])
                self.model.pre_forward(state)

                # Some lists to store results
                fitness_improve = []
                revisited = []
                avg_similarity = []
                max_similarity = []
                tot_reward = []
                total_steps = 0
                avg_total_reward = 0
                loss_values = []
                while not done:
                    log_probs = []
                    rewards = []
                    for step in range(self.trainer_params['episode_length']):
                        # Get action probabilities from model
                        if self.compile:
                            logits = self.compiled_model(state)
                        else:
                            logits = self.model(state)

                        log_p = F.log_softmax(logits.squeeze(-1), dim=-1)

                        probs = log_p.exp()
                        actions = probs.multinomial(1).squeeze(1)
                        log_p = log_p.gather(1, actions.unsqueeze(-1)).squeeze(-1)
                        log_probs.append(log_p)

                        # Perform the action
                        state, R_dict, done = self.env.step(actions)

                        # Store results
                        rewards.append(R_dict['Reward'])
                        fitness_improve.append(R_dict['Fitness improvement'])
                        tot_reward.append(R_dict['Reward'].mean().item())
                        revisited.append(R_dict['Re-Visited']/self.env.batch_size)
                        avg_similarity.append(R_dict['Avg similarity']/self.env.batch_size)
                        max_similarity.append(R_dict['Max similarity']/self.env.batch_size)

                        total_steps += 1

                    # Calculate discounting rewards
                    t_steps = torch.arange(len(rewards))
                    discounts = self.trainer_params['gamma'] ** t_steps
                    r = [r_i * d_i for r_i, d_i in zip(rewards, discounts)]
                    r = r[::-1]
                    b = torch.cumsum(torch.stack(r), dim=0)
                    c = [b[k, :] for k in reversed(range(b.shape[0]))]
                    R = [c_i / d_i for c_i, d_i in zip(c, discounts)]

                    # Check reward and save if best
                    if done:
                        steps_until_last_save += 1
                        avg_total_reward = np.sum(tot_reward)
                        if avg_total_reward > best_fitness_ma:
                            best_fitness_ma = avg_total_reward
                            if self.trainer_params['verbose']:
                                print(f"New best reward: {avg_total_reward:.4f}!")

                    # Compute loss and back-propagate
                    self.optimizer.zero_grad()
                    policy_loss = []
                    for log_prob_i, r_i in zip(log_probs, R):
                        policy_loss.append(-log_prob_i * r_i)
                    loss = torch.cat(policy_loss).mean()
                    loss.backward()

                    # clip grad norms
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_params['max_grad_norm'])

                    self.optimizer.step()
                    loss_values.append(loss.item())

                best_fitness = self.env.best_fitness_values.mean().item()

                # Print results
                if self.trainer_params['verbose']:
                    elapsed_seconds = time.time() - start_time
                    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))

                    print(f"Ep {episode+1}/{self.trainer_params['n_episodes']}) Best Fit: {best_fitness:.2f}. "
                          f"Loss: {np.mean(loss_values):.4f}. Reward: {avg_total_reward:.3f}. Revisit: {np.mean(revisited):.3f}. "
                          f"Avg/Max Sim: {np.mean(avg_similarity):.2f}/{np.mean(max_similarity):.2f}. {total_steps} steps. Elap time: {elapsed_time}")

            # Test model in different sizes.
            test_results, path = self.run_tests(epoch)

            if self.trainer_params['save_model'] and (epoch % self.trainer_params['save_model_freq'] == 0):
                self.save_model(path)

    def save_model(self, path):
        if isinstance(self.model, torch.nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def run_tests(self, epoch, episode=""):
        test_results = {}

        path = f"saved_models/{self.memory_params['memory_type']}_{self.trainer_params['execution_name']}_epoch{epoch}{episode}_n{self.env.n}_"

        for i in range(len(self.test_graphs)):
            er_graphs = self.test_graphs[i]
            bl_values = self.bl_values[i]
            size = er_graphs.size(2)
            best_fitness, best_gap, avg_fitness, avg_gap = self.run_test(er_graphs, bl_values)
            test_results[f'best_fitness_{size}'] = best_fitness
            test_results[f'best_gap_{size}'] = best_gap
            test_results[f'avg_fitness_{size}'] = avg_fitness
            test_results[f'avg_gap_{size}'] = avg_gap
            path += f"_{size}_{best_fitness:.2f}__"

        path += '.pth'
        return test_results, path

    @torch.no_grad()
    def run_test(self, test_graphs, ref_best_fitness):
        test_batch_size, n, _ = test_graphs.size()
        n_solutions = self.trainer_params['n_solutions']
        self.test_env.max_iterations = 2*n  # self.trainer_params['max_test_steps']
        if self.trainer_params['verbose']:
            print(f"\nTesting with {test_batch_size} graphs of size {n}. {n_solutions} initializations per graph. {self.test_env.max_iterations} steps.")

        all_best_fitness = torch.zeros(test_batch_size)
        all_avg_fitness = torch.zeros(test_batch_size)

        self.model.eval()
        start_time = time.time()
        for b in range(test_batch_size):
            state, done = self.test_env.reset(n_solutions, test_graphs[b].unsqueeze(0).to(self.device), solution_seed=42)
            self.model.pre_forward(state)
            step = 0
            while not done:
                step += 1
                if self.compile:
                    logits = self.compiled_model(state, testing=True).squeeze(-1)
                else:
                    logits = self.model(state, testing=True).squeeze(-1)

                topk_actions = torch.topk(logits, self.trainer_params['topk'], dim=1).indices
                for k in range(self.trainer_params['topk']):
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
                    state, R, done = self.test_env.step(actions)

            best_fitness = self.test_env.best_fitness_values
            all_best_fitness[b] = best_fitness.max()
            all_avg_fitness[b] = best_fitness.mean()

        # End of testing
        best_gap = (100 * (ref_best_fitness - all_best_fitness) / ref_best_fitness).mean().item()
        avg_gap = (100 * (ref_best_fitness - all_avg_fitness) / ref_best_fitness).mean().item()
        all_avg_fitness = all_avg_fitness.mean().item()
        all_best_fitness = all_best_fitness.mean().item()

        if self.trainer_params['verbose']:
            total_time = time.time() - start_time
            print(f"Best fitness: {all_best_fitness:.3f} (Gap {best_gap:.2f}%). Average fitness: {all_avg_fitness:.3f} (Gap: {avg_gap:.2f}%). "
                  f"Tot time: {total_time:.2f}s ({total_time / test_batch_size:.2f}s/instance).")

        return all_best_fitness, best_gap, all_avg_fitness, avg_gap


def main():
    # Get options
    env_params, memory_params, model_params, optimizer_params, trainer_params = get_options()

    model_params['use_mem'] = memory_params['memory_type'] != 'none'
    execution_name = generate_word(6)
    trainer_params['execution_name'] = execution_name
    print(f'Execution name: {execution_name}')
    print(env_params)
    print(memory_params)
    print(model_params)
    print(trainer_params)

    trainer = Trainer(env_params, memory_params, model_params, optimizer_params, trainer_params)
    trainer.train()


if __name__ == "__main__":
    main()
