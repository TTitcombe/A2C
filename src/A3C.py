"""ASynchronous Advantage Actor Critic (A3C) algorithm"""
import gym
import torch
import torch.multiprocessing as mp

from .A2C import A2C
from .actor_critic import ActorCritic, ActorCriticSeparate


class A3C(A2C):
    def __init__(
        self,
        n_inputs: int,
        n_actions: int,
        env,
        learning_rate=1e-3,
        separate_models=True,
    ):
        self.env_name = env
        self.separate_models = separate_models
        self.n_inputs = n_inputs
        self.n_actions = n_actions

        self.model = self._get_model()

        self.T = mp.Queue()
        self.episode_rewards = mp.Queue()

        self.optimiser = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        self.lr = learning_rate

    def _get_env(self):
        return gym.make(self.env_name)

    def _get_model(self):
        if self.separate_models:
            return ActorCriticSeparate(self.n_inputs, self.n_actions)
        else:
            return ActorCritic(self.n_inputs, self.n_actions)

    def train(
        self,
        T_max: int,
        t_max: int,
        num_processes=2,
        gamma=0.99,
        plot=True,
        max_episodes=2000,
        render=False,
        verbose=False,
    ) -> list:
        self.model.share_memory()

        self.T.put(0)
        self.T_max = T_max

        self.verbose = verbose

        procs = []

        for i in range(num_processes):
            proc = mp.Process(
                target=self.train_process, args=(self.model, t_max, gamma, i)
            )
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()

    def train_process(self, model, t_max, gamma, process_id):
        print("Starting process: {}".format(process_id))

        T = self.T.get()
        self.T.put(T + 1)

        episode_reward = 0

        env = self._get_env()
        state = env.reset()

        optim = torch.optim.RMSprop(model.parameters(), lr=self.lr)

        while T < self.T_max:
            states = []
            actions = []
            rewards = []
            terminals = []

            # Get the latest model
            model.load_state_dict(self.model.state_dict())

            for t in range(t_max):
                action = self._select_action(state, model=model)
                next_state, reward, is_done, _ = env.step(action)
                episode_reward += reward

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                terminals.append(is_done)

                if is_done:
                    state = env.reset()

                    self.episode_rewards.put(episode_reward)
                    episode_reward = 0
                else:
                    state = next_state

                T = self.T.get()
                self.T.put(T + 1)

            # Update the model
            self._update(
                states, actions, rewards, terminals, gamma, model=model, optim=optim
            )

    def play(self) -> None:
        score = 0.0
        is_done = False
        env = self._get_env()
        state = env.reset()
        while not is_done:
            action = self._select_action(state, best=True)
            state, reward, is_done, _ = env.step(action)
            score += reward
        return score
