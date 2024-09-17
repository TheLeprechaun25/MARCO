from dataclasses import dataclass
import torch
from env.TSPMemory import select_memory
from env.utils import get_random_problems, tsp_coordinate_augmentations, get_distances_from_coords


@dataclass
class Reset_State:
    coords: torch.Tensor
    # shape: (batch, node, 2)
    dist: torch.Tensor
    # shape: (batch, node, node)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)
    edge_solution: torch.Tensor = None
    # shape: (batch, pomo, node, node)
    edge_memory: torch.Tensor = None
    # shape: (batch, pomo, node, node)


class TSPEnv:
    def __init__(self, **env_params):
        # Params
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        self.retrieve_freq = self.env_params['retrieve_freq']

        # Static
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.coords = None  # shape: (batch, node, 2)
        self.dist = None  # shape: (batch, node, node)

        # Dynamic
        self.step_state = None
        self.selected_count = None
        self.current_node = None  # shape: (batch, pomo)
        self.selected_node_list = None  # shape: (batch, pomo, 0~problem)
        self.punishments = []

        # Memory
        self.memory = None

    def load_problems(self, batch_size, coords=None, aug_factor=1):
        self.batch_size = batch_size

        # Generate or load instances
        if coords is None:
            self.coords = get_random_problems(batch_size, self.problem_size)
        else:
            self.coords = coords  # Use loaded instances
        # coords.shape: (batch, problem, 2)

        # Compute distances
        self.dist = get_distances_from_coords(self.coords)
        # dist.shape: (batch, problem, problem)

        # Augment data
        if aug_factor > 1:
            self.batch_size = self.batch_size * aug_factor
            #self.coords = augment_xy_data_by_8_fold(self.coords)
            self.coords = tsp_coordinate_augmentations(self.coords, aug_factor)
            # shape: (aug_factor*batch, problem, 2)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def initialize_memory(self, testing, device):
        # Initialize memory
        self.memory = select_memory(mem_aggr=self.env_params['mem_aggr'],
                                    state_dim=self.problem_size**2,
                                    batch_size=self.batch_size,
                                    device=device)
        if (not testing) and self.env_params['memory_type'] == 'none':
            assert self.env_params['repeat_punishment'] == 0, "repeat_punishment is not 0 but memory_type is none"

    def reset(self):
        # Create reset state
        reset_state = Reset_State(coords=self.coords, dist=self.dist)
        self.punishments = [torch.zeros(self.batch_size, self.pomo_size)]
        reward = None
        done = False
        return reset_state, reward, done


    def pre_step(self):
        # Create dynamic variables
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # Create step state
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)
        self.step_state.edge_solution = torch.zeros((self.batch_size, self.pomo_size, self.problem_size, self.problem_size), dtype=torch.long)
        # shape: (batch, pomo, problem, problem)
        if self.env_params['memory_type'] != 'none':
            self.step_state.edge_memory = torch.zeros((self.batch_size, self.pomo_size, self.problem_size, self.problem_size), dtype=torch.float32)
            # shape: (batch, pomo, problem, problem)

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)

        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # Update edge-based solutions
        if self.selected_count > 1:
            prev_selected = self.selected_node_list[:, :, -2]
            self.step_state.edge_solution[self.BATCH_IDX, self.POMO_IDX, prev_selected, self.current_node] = 1
            self.step_state.edge_solution[self.BATCH_IDX, self.POMO_IDX, self.current_node, prev_selected] = 1

        # Update step state
        self.step_state.current_node = self.current_node  # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')  # shape: (batch, pomo, node)

        # Retrieve from memory
        avg_sim, max_sim = None, None
        if self.selected_count > 0 and (self.selected_count % self.retrieve_freq == 0):
            edge_mem, avg_sim, max_sim = self.memory.get_knn(self.step_state.edge_solution, self.selected_count, self.env_params['k'], return_similarity=True)
            if (self.env_params['memory_type'] != 'none') and (edge_mem is not None):
                self.step_state.edge_memory = edge_mem

        # Returning values
        done = (self.selected_count == self.problem_size)
        R = {'reward': None, 'punishment': None}
        if done:
            # add final edge
            self.step_state.edge_solution[self.BATCH_IDX, self.POMO_IDX, self.selected_node_list[:, :, -1], self.selected_node_list[:, :, 0]] = 1
            self.step_state.edge_solution[self.BATCH_IDX, self.POMO_IDX, self.selected_node_list[:, :, 0], self.selected_node_list[:, :, -1]] = 1
            reward = -self._get_travel_distance()  # note the minus sign!
            if self.env_params['memory_type'] != 'none':
                # Save final solution in memory
                self.memory.save_in_memory(self.step_state.edge_solution)

                if avg_sim is not None:
                    R['avg_sim'] = avg_sim
                    R['max_sim'] = max_sim

            R['reward'] = reward

        return self.step_state, R, done

    def _get_travel_distance(self):
        travel_distances = self.step_state.edge_solution * self.dist.unsqueeze(1).expand(self.batch_size, self.pomo_size, self.problem_size, self.problem_size) / 2
        # sum over last and second last dimensions
        return travel_distances.sum(dim=[2, 3])
