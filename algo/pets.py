import os
import numpy as np
from scipy.io import savemat
from components.cem import CEMOptimizer
from components.dynamics import NormalDynamicModel
from utils.radam import RAdam
import torch as th


class PETS:
    def __init__(
            self,
            d_state, d_action,
            device,
            action_ub, action_lb,
            plan_horizon,
            n_particles,
            exploitation_task,
            clip_value,
            args
    ):
        super(PETS, self).__init__()
        self.d_state, self.d_action = d_state, d_action
        self.plan_hor = plan_horizon
        self.n_part = n_particles
        self.task = exploitation_task
        # Action bounds for gymmb envs [-1, 1]
        self.act_ub, self.act_lb = action_ub, action_lb

        self.device = device
        self.has_been_trained = False
        self.cur_states = None
        self.clip_value = clip_value
        self.actions_buffer = np.array([]).reshape(0, self.d_action)
        self.prev_sol = np.tile((self.act_lb + self.act_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.act_ub - self.act_lb) / 16, [self.plan_hor])

        self.model = NormalDynamicModel(
            d_state=self.d_state, d_action=self.d_action, n_units=args.model_n_units, n_layers=args.model_n_layers,
            ensemble_size=args.model_ensemble_size, activation=args.model_activation,
            device=self.device
        )
        self.model_optimizer = RAdam(self.model.parameters(),
                                     lr=args.model_lr, weight_decay=args.model_weight_decay)

        self.cem = CEMOptimizer(
            solution_dim=self.plan_hor * self.d_action,
            lower_bound=np.tile(self.act_lb, [self.plan_hor]),
            upper_bound=np.tile(self.act_ub, [self.plan_hor]),
            cost_function=self._compile_cost,
            max_iter=args.cem_max_iter, population_size=args.population_size, num_elites=args.num_elites
        )

    def reset(self):
        self.prev_sol = np.tile((self.act_lb + self.act_ub) / 2, [self.plan_hor])
        self.init_var = np.tile((self.act_ub - self.act_lb) / 16, [self.plan_hor])

    def get_action(self, states, deterministic=False):
        if not self.has_been_trained:
            return th.rand(size=(states.shape[0], self.d_action), device=self.device) * - 1
        if self.actions_buffer.shape[0] > 0:
            action, self.actions_buffer = self.actions_buffer[0], self.actions_buffer[1:]
            action = th.from_numpy(action).unsqueeze(0)
            return action

        self.cur_states = states

        solution = self.cem.obtain_solution(self.prev_sol, self.init_var)
        self.prev_sol = np.concatenate([np.copy(solution)[self.d_action:], np.zeros(self.d_action, dtype=np.float32)])
        self.actions_buffer = solution[: self.d_action].reshape(-1, self.d_action)

        return self.get_action(states)

    def update(self, states, actions, state_deltas):
        self.has_been_trained = True
        self.model_optimizer.zero_grad()

        loss = self.model.loss(states, actions, state_deltas)
        loss.backward()
        th.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)

        self.model_optimizer.step()

        return loss.item()

    @th.no_grad()
    def _compile_cost(self, actions_sequences):
        """
        Arguments:
            actions_sequences: (np.ndarray) [population_size, plan_horizon * d_action]
        """
        n_opt = actions_sequences.shape[0]

        actions_sequences = th.from_numpy(actions_sequences).float().to(self.device)
        # shape = [population_size, plan_horizon, d_action]
        actions_sequences = actions_sequences.view(-1, self.plan_hor, self.d_action)

        transposed = actions_sequences.transpose(0, 1)

        expanded = transposed[:, :, None]

        tiled = expanded.expand(-1, -1, self.n_part, -1)
        # shape = [plan_horizon, population_size * n_particles, d_action]
        actions_sequences = tiled.contiguous().view(self.plan_hor, -1, self.d_action)
        # shape = [1, d_state] --> [population_size * n_particles, d_state]
        cur_states = self.cur_states.to(self.device)
        cur_states = cur_states.expand(n_opt * self.n_part, -1)

        costs = th.zeros(n_opt, self.n_part, device=self.device)

        for t in range(self.plan_hor):
            cur_acs = actions_sequences[t]
            next_states = self.model.sample(cur_states, cur_acs)

            step_rewards = self.task(cur_states, cur_acs, next_states)
            step_rewards = step_rewards.reshape([-1, self.n_part])

            costs += step_rewards
            cur_states = next_states

        return costs.mean(dim=1).detach().cpu().numpy()

    def save(self, filename):
        param_dict = {
            "model": self.model.state_dict()
        }
        th.save(param_dict, filename)

    def load(self, filename):
        param_dict = th.load(filename)
        self.model.load_state_dict(param_dict["model"])

