import torch
from logging import getLogger
import pickle
from env.TSPEnv import TSPEnv as Env
from nets.TSPModel import TSPModel as Model
from torch.optim import AdamW as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from utils import *

torch.set_float32_matmul_precision('high')


class TSPTrainer:
    def __init__(self, env_params, model_params, optimizer_params, trainer_params):
        # Save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.marco = self.env_params['memory_type'] != 'none'
        if self.marco:
            assert trainer_params['finetune']

        # Result folder, logger
        self.logger = getLogger(name='trainer')

        # Device
        device = torch.device('cuda') if (trainer_params['use_cuda'] and torch.cuda.is_available()) else torch.device('cpu')
        self.device = device
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # print number of params in model
        self.logger.info('Number of Model Parameters: {}'.format(sum([p.numel() for p in self.model.parameters()])))

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            path = model_load['path']
            epoch = model_load['epoch']
            checkpoint_fullname = f'{path}/checkpoint-{epoch}.pt'
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.logger.info('Saved Model Loaded !!')

        if trainer_params['finetune']:
            # Freeze all parameters in the model
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze the parameters of self.model.decoder.mem_linear
            for param in self.model.decoder.mem_FF.parameters():
                param.requires_grad = True

            # print the params that are being trained
            self.logger.info('Number of Trainable Parameters: {}'.format(sum([p.numel() for p in self.model.parameters() if p.requires_grad])))

        # Load test data
        self.test_graphs = {}
        self.opt_test_fitness = {}
        for test_size, test_batch_size in zip(self.trainer_params['test_sizes'], self.trainer_params['test_batch_sizes']):
            path = f"../data/tsp/tsp{test_size}_test_concorde.pkl"
            graphs = pickle.load(open(path, 'rb'))
            coord_m = np.array(graphs[:test_batch_size])
            self.test_graphs[test_size] = torch.from_numpy(coord_m).to(device).float()
            path = f"../data/tsp/tsp{test_size}_test_concorde_costs.pkl"
            opt_score = pickle.load(open(path, 'rb'))
            self.opt_test_fitness[test_size] = torch.tensor(opt_score[:test_batch_size])

        # utility
        self.time_estimator = TimeEstimator()
        self.result_folder = get_result_folder()
        self.result_log = LogData()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            # LR Decay
            self.scheduler.step()

            # Logs & Checkpoint
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            if all_done or ((epoch % self.trainer_params['model_save_interval']) == 0):
                # Test the model
                self.logger.info("Testing trained_model")
                test_scores, test_gaps = self.test_model(self.trainer_params['test_sizes'], self.trainer_params['test_batch_sizes'], self.trainer_params['test_pomo_sizes'])

                self.logger.info(f"Epoch {epoch:3d}.")
                results = {}
                for key in test_scores.keys():
                    self.logger.info(f"N: {key}. Score: {test_scores[key]:.4f}. Gap: {test_gaps[key]:.4f}%")
                    results[f"test_score_{key}"] = test_scores[key]
                    results[f"test_gap_{key}"] = test_gaps[key]

                # Save the model
                self.logger.info("Saving trained_model")
                if self.trainer_params['finetune']:
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.model.decoder.mem_FF.state_dict(),
                        'result_log': self.result_log.get_raw_data()
                    }
                    torch.save(checkpoint_dict, '{}/checkpoint-marco-{}.pt'.format(self.result_folder, epoch))
                else:
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'result_log': self.result_log.get_raw_data()
                    }
                    torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        distance_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            self.model.train()

            if self.trainer_params['rand_train_sizes']:
                # Get a random problem size from 20 to problem size
                assert self.env_params['problem_size'] >= 100, "problem size should be larger than 100"
                self.env.problem_size = torch.randint(100, self.env_params['problem_size'] + 1, (1,)).item()
                self.env.pomo_size = min(self.env.problem_size, 100)

            self.env.load_problems(batch_size)
            self.env.initialize_memory(testing=False, device=self.device)

            if self.marco:
                avg_score, avg_loss, avg_distance = self._train_one_batch_marco(batch_size)
            else:  # MARCO
                avg_score, avg_loss, avg_distance = self._train_one_batch(batch_size)

            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            distance_AM.update(avg_distance, batch_size)

            episode += batch_size

        # Log for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}, Distance: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg, distance_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        # Sampling
        R, prob_list = self.rollout(batch_size, deterministic=False)
        if self.env_params['normalize']:
            reward = R['reward'] / self.env.problem_size
        else:
            reward = R['reward']

        # Loss
        advantage = reward - reward.float().mean(dim=1, keepdims=True)

        log_prob = prob_list.log()
        advantage = advantage[:, :, None].expand(-1, -1, log_prob.shape[2])
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD

        loss_mean = loss.mean()

        # Score
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        self.model.zero_grad()
        loss_mean.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_params['max_grad_norm'])
        self.optimizer.step()
        return score_mean.item(), loss_mean.item(), 0

    def _train_one_batch_marco(self, batch_size):
        with torch.no_grad():
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            # Deterministic Rollout
            bl_R, _ = self.rollout(batch_size, deterministic=True)
            bl_reward = bl_R['reward'] / self.env.problem_size

        # Sampling
        avg_scores = []
        avg_losses = []
        avg_distance = []
        for _ in range(4):
            R, prob_list = self.rollout(batch_size, deterministic=False)

            reward = R['reward'] / self.env.problem_size
            #punishment = R['punishment'] / self.env.problem_size
            # Score
            max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
            score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

            avg_dist = 1 - R['avg_sim']
            # avg_dist[avg_dist > 0.15] = 0.15
            min_dist = 1 - R['max_sim']
            # Objective = minimize cost, minimize similarity, improve previous cost
            reward = reward - bl_reward
            worse_idx = (reward < bl_reward)
            if self.env_params['punish_type'] == 'avg':
                advantage = reward + self.env_params['repeat_punishment'] * avg_dist
            elif self.env_params['punish_type'] == 'max':
                advantage = reward + self.env_params['repeat_punishment'] * min_dist
            else:
                raise NotImplementedError

            #advantage[worse_idx] = 0

            # Loss
            log_prob = prob_list.log()
            advantage = advantage[:, :, None].expand(-1, -1, log_prob.shape[2])
            loss = -advantage * log_prob  # Minus Sign: To Increase REWARD

            loss_mean = loss.mean()

            # Step & Return
            self.model.zero_grad()
            loss_mean.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_params['max_grad_norm'])
            self.optimizer.step()

            avg_scores.append(score_mean.item())
            avg_losses.append(loss_mean.item())
            avg_distance.append(avg_dist.mean().item())
        return np.mean(avg_scores), np.mean(avg_losses), np.mean(avg_distance)

    def rollout(self, batch_size, deterministic):
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        state, reward, done = self.env.pre_step()
        while not done:
            # Get actions
            selected, prob = self.model(state, deterministic)

            # Step the environment
            state, reward, done = self.env.step(selected)

            if not deterministic:
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        return reward, prob_list

    def test_model(self, test_sizes, test_batch_sizes, pomo_sizes):
        test_scores = {}
        test_gaps = {}
        for test_size, test_batch_size, pomo_size in zip(test_sizes, test_batch_sizes, pomo_sizes):
            if self.marco:
                test_score, test_gap = self._test_one_batch_marco(test_size, test_batch_size, pomo_size)
            else:
                test_score, test_gap = self._test_one_batch(test_size, test_batch_size, pomo_size)
            test_scores[test_size] = test_score
            test_gaps[test_size] = test_gap
        return test_scores, test_gaps

    @torch.no_grad()
    def _test_one_batch(self, test_size, test_batch_size, pomo_size):
        env_params = self.env_params.copy()
        env_params['problem_size'] = test_size
        env_params['pomo_size'] = pomo_size
        coords = self.test_graphs[test_size]
        opt_scores = self.opt_test_fitness[test_size]

        test_env = Env(**env_params)

        self.model.eval()
        test_env.load_problems(test_batch_size, coords=coords)
        test_env.initialize_memory(testing=True, device=self.device)

        reset_state, _, _ = test_env.reset()
        self.model.pre_forward(reset_state)
        # POMO Rollout
        ###############################################
        state, R, done = test_env.pre_step()
        while not done:
            selected, _ = self.model(state, deterministic=True, use_memory=False)

            # shape: (batch, pomo)
            state, R, done = test_env.step(selected)

        # return score
        reward = R['reward']
        max_pomo_reward, _ = reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()
        gap = 100 * (-max_pomo_reward - opt_scores).float() / opt_scores
        return score_mean.item(), gap.mean().item()

    @torch.no_grad()
    def _test_one_batch_marco(self, test_size, test_batch_size, pomo_size):
        env_params = self.env_params.copy()
        env_params['problem_size'] = test_size
        env_params['pomo_size'] = pomo_size
        env_params['retrieve_freq'] = 10
        coords = self.test_graphs[test_size]
        opt_scores = self.opt_test_fitness[test_size]

        test_env = Env(**env_params)

        self.model.eval()
        test_env.load_problems(test_batch_size, coords=coords)
        test_env.initialize_memory(testing=True, device=self.device)

        reset_state, _, _ = test_env.reset()
        self.model.pre_forward(reset_state)

        # Deterministic POMO Rollout
        state, R, done = test_env.pre_step()
        while not done:
            selected, _ = self.model(state, deterministic=True, use_memory=False)
            # shape: (batch, pomo)
            state, R, done = test_env.step(selected)
        best_rewards = R['reward'].max(dim=1)[0]

        score_mean = -best_rewards.float().mean()
        gap = 100 * (-best_rewards - opt_scores).float() / opt_scores
        self.logger.info(f"Greedy, N: {test_size}. Score: {score_mean:.4f}. Gap: {gap.mean():.4f}%")

        # additional rollouts (optionally stochastic)
        for i in range(10):
            state, R, done = test_env.pre_step()
            while not done:
                selected, _ = self.model(state, deterministic=True, use_memory=True)
                # shape: (batch, pomo)
                state, R, done = test_env.step(selected)

            cur_rewards = R['reward'].max(dim=1)[0]
            best_rewards = torch.max(best_rewards, cur_rewards)

            best_score_mean = -best_rewards.float().mean()
            cur_score_mean = -cur_rewards.float().mean()
            best_gap = 100 * (-best_rewards - opt_scores).float() / opt_scores
            cur_gap = 100 * (-cur_rewards - opt_scores).float() / opt_scores
            self.logger.info(f"Rollout {i}, N: {test_size}. Best Score: {best_score_mean:.2f} ({best_gap.mean():.4f}%). Cur Score: {cur_score_mean:.2f} ({cur_gap.mean():.4f}%).")

        # return score
        score_mean = -best_rewards.float().mean()
        gap = 100 * (-best_rewards - opt_scores).float() / opt_scores
        return score_mean.item(), gap.mean().item()
