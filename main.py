import numpy as np
import torch
import os
import gym
from dotmap import DotMap
from pathlib import Path
from tensorboardX import SummaryWriter

import envs
import envs.gymmb
from components.env_loop import EnvLoop
from components.buffer import Buffer
from components.normalizer import TransitionNormalizer
from components.arguments import common_args, pets_args, dynamics_model
from algo.pets import PETS
from utils.wrappers import BoundedActionsEnv, IsDoneEnv, MuJoCoCloseFixWrapper, RecordedEnv
from utils.misc import to_np, EpisodeStats


def get_random_agent(d_action, device):
    class RandomAgent:
        @staticmethod
        def get_action(states, deterministic=False):
            return torch.rand(size=(states.shape[0], d_action), device=device) * - 1
    return RandomAgent()


def get_deterministic_agent(agent):
    class DeterministicAgent:
        @staticmethod
        def get_action(states):
            return agent.get_action(states, deterministic=True)
    return DeterministicAgent()


def get_env(env_name, record=False):
    env = gym.make(env_name)
    env = BoundedActionsEnv(env)

    env = IsDoneEnv(env)
    env = MuJoCoCloseFixWrapper(env)
    if record:
        env = RecordedEnv(env)

    env.seed(np.random.randint(np.iinfo(np.uint32).max))
    if hasattr(env.action_space, 'seed'):  # Only for more recent gym
        env.action_space.seed(np.random.randint(np.iinfo(np.uint32).max))
    if hasattr(env.observation_space, 'seed'):  # Only for more recent gym
        env.observation_space.seed(np.random.randint(np.iinfo(np.uint32).max))

    return env


class MainLoopTraining:
    def __init__(self, logger, args):
        self.step_i = 0
        # env_config
        tmp_env = gym.make(args.env_name)
        # Cheetah, Pusher, Swimmer, Pendulum --> is_done always return false
        self.is_done = tmp_env.unwrapped.is_done
        # eval task: default standard
        self.eval_tasks = {args.task_name: tmp_env.tasks()[args.task_name]}
        # Exploitation_task: reward computation
        self.exploitation_task = tmp_env.tasks()[args.task_name]
        self.d_state = tmp_env.observation_space.shape[0]
        self.d_action = tmp_env.action_space.shape[0]
        # Action bounded as [-1, 1]
        self.act_lb = np.clip(tmp_env.action_space.low, -1., 1.)
        self.act_ub = np.clip(tmp_env.action_space.high, -1., 1.)
        self.max_episode_steps = tmp_env.spec.max_episode_steps
        del tmp_env
        if args.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.logger = logger
        # Wrapped Env
        self.env_loop = EnvLoop(get_env, env_name=args.env_name, render=args.render)

        # Buffer and Normalizer
        self.buffer = Buffer(self.d_state, self.d_action, args.n_total_steps)
        if args.normalize_data:
            self.buffer.setup_normalizer(TransitionNormalizer(self.d_state, self.d_action, self.device))

        # Agent
        self.agent = PETS(
            d_state=self.d_state, d_action=self.d_action, device=self.device, exploitation_task=self.exploitation_task,
            action_lb=self.act_lb, action_ub=self.act_ub, n_particles=args.n_particles, plan_horizon=args.plan_horizon,
            clip_value=args.grad_clip, args=args
        )

        self.stats = EpisodeStats(self.eval_tasks)
        self.last_avg_eval_score = None
        self.random_agent = get_random_agent(self.d_action, self.device)

        self.args = args

    def train(self):
        self.step_i += 1

        behavior_agent = self.random_agent if self.step_i <= self.args.n_warm_up_steps else self.agent
        # behavior_agent = self.agent
        with torch.no_grad():
            action = behavior_agent.get_action(self.env_loop.state, deterministic=False).to('cpu')

        state, next_state, done = self.env_loop.step(to_np(action))
        reward = self.exploitation_task(state, action, next_state).item()
        self.buffer.add(state, action, next_state, torch.from_numpy(np.array([[reward]], dtype=np.float32)))
        self.stats.add(state, action, next_state, done)

        # Reset Cross Entropy Mean and Variance
        if done:
            self.agent.reset()
            for task_name in self.eval_tasks:
                last_ep_return = self.stats.ep_returns[task_name][-1]
                last_ep_length = self.stats.ep_lengths[task_name][-1]
                print(f"MainLoopStep {self.step_i} | train | EpReturns {last_ep_return: 5.2f} | EpLength {last_ep_length: 5.2f}")
                self.logger.add_scalar("TrainingEpReturn", last_ep_return, self.step_i)

        # Training Dynamics Model
        # -------------------------------
        if (self.args.model_training_freq is not None and self.args.model_training_n_batches > 0
                and self.step_i % self.args.model_training_freq == 0):
            loss = np.nan
            batch_i = 0
            while batch_i < self.args.model_training_n_batches:
                losses = []
                for states, actions, state_deltas in self.buffer.train_batches(self.args.model_ensemble_size,
                                                                               self.args.model_batch_size):
                    train_loss = self.agent.update(states, actions, state_deltas)
                    losses.append(train_loss)
                batch_i += len(losses)
                loss = np.mean(losses)
            self.logger.add_scalar("TrainingModelLoss", loss, self.step_i)

        # Print TaskName StepReward
        # ------------------------------
        for task_name in self.eval_tasks:
            step_reward = self.stats.get_recent_reward(task_name)
            # print(f"Step {self.step_i}\tReward: {step_reward}")
            self.logger.add_scalar("StepReward", step_reward, self.step_i)

        # Save agent parameters
        # ------------------------------
        if self.step_i >= self.args.n_warm_up_steps and self.step_i % self.args.save_freq == 0:
            os.makedirs(str(args.run_dir / 'incremental'), exist_ok=True)
            self.agent.save(str(args.run_dir / 'incremental' / ('model_step%i.pt' % self.step_i)))
            self.agent.save(str(self.args.run_dir / 'model.pt'))

        experiment_finished = self.step_i >= self.args.n_total_steps
        return DotMap(
            done=experiment_finished,
            step_i=self.step_i
        )

    def stop(self):
        self.env_loop.close()


if __name__ == "__main__":
    args = common_args()
    args = pets_args(args)
    args = dynamics_model(args)

    # Save Directory
    model_dir = Path('./models') / args.env_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir()
                         if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    args.run_dir = model_dir / curr_run
    args.log_dir = args.run_dir / 'logs'
    os.makedirs(str(args.log_dir))
    logger = SummaryWriter(str(args.log_dir))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not args.use_cuda and args.n_training_threads is not None:
        torch.set_num_threads(args.n_training_threads)

    training = MainLoopTraining(args)
    # MainLoop
    res = DotMap(done=False)
    while not res.done:
        res = training.train()

    training.stop()
