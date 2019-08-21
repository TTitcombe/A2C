"""Synchronous Advantage Actor Critic (A2C) algorithm"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from .actor_critic import ActorCritic, ActorCriticSeparate
from utils import moving_average


class A2C:
    def __init__(
        self,
        n_inputs: int,
        n_actions: int,
        env,
        learning_rate=1e-3,
        separate_models=True,
    ):
        if separate_models:
            self.model = ActorCriticSeparate(n_inputs, n_actions)
        else:
            self.model = ActorCritic(n_inputs, n_actions)

        self.env = env

        self.optimiser = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)

    def train(
        self,
        T_max: int,
        t_max: int,
        max_episodes=2000,
        render=False,
        gamma=0.99,
        plot=True,
        verbose=False,
    ) -> list:

        self.verbose = verbose
        T = 0

        state = self.env.reset()

        episode_number = 1
        episode_reward = 0
        ep_rewards = []

        while T < T_max:
            states = []
            actions = []
            rewards = []
            terminals = []
            for t in range(t_max):
                action = self._select_action(state)
                next_state, reward, is_done, _ = self.env.step(action)
                episode_reward += reward

                if render:
                    self.env.render()

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                terminals.append(is_done)

                if is_done:
                    if self.verbose:
                        print(
                            "Reward for episode {}: {}".format(
                                episode_number, episode_reward
                            )
                        )

                    state = self.env.reset()

                    ep_rewards.append(episode_reward)
                    episode_reward = 0
                    episode_number += 1
                    if episode_number > max_episodes:
                        T = T_max + 1
                        break
                else:
                    state = next_state

                T += 1

            # Update the model
            self._update(states, actions, rewards, terminals, gamma)

        if plot:
            self.plot_scores(ep_rewards)

        return ep_rewards

    def _select_action(self, state: np.ndarray, best=False) -> int:
        state = torch.from_numpy(state).float().unsqueeze(0)
        if best:
            return self.model.action_head(state).max(1)[1].item()
        else:
            return self.model.action_head(state).multinomial(1).item()

    def _update(
        self,
        states: list,
        actions: list,
        action_rewards: list,
        terminals: list,
        gamma: float,
    ) -> None:
        # Get rewards
        total_rewards = self._calc_total_rewards(
            states, action_rewards, terminals, gamma
        )

        # Run states through the model to get values AND actions
        states = torch.FloatTensor(states)
        action_probs, q_values = self.model.evaluate(states)

        # Calculate value loss
        advantage = total_rewards - q_values
        value_loss = advantage.pow(2).mean()

        # Calculate action loss
        actions = torch.LongTensor(actions).view(-1, 1)
        log_action_probs = action_probs.log().gather(1, actions)
        action_loss = (log_action_probs * advantage).mean()

        # Loss
        loss = value_loss - action_loss
        if self.verbose:
            print("Loss: {:.3f}".format(loss.item()))

        # Update parameters
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def _calc_total_rewards(
        self, states: list, action_rewards: list, terminals: list, gamma: float
    ) -> torch.Tensor:
        if terminals[-1]:
            total_rewards = [0]
        else:
            state = torch.from_numpy(states[-1]).float().unsqueeze(0)
            total_rewards = [self.model.value_head(state).item()]

        for i in range(2, len(action_rewards) + 1):
            if not terminals[-i]:
                reward = action_rewards[-i] + gamma * total_rewards[-1]
            else:
                reward = 0.0
            total_rewards.append(reward)

        total_rewards.reverse()
        return torch.FloatTensor(total_rewards).unsqueeze(1)

    def test(self) -> None:
        score = 0.0
        is_done = False
        state = self.env.reset()
        while not is_done:
            action = self._select_action(state, best=True)
            state, reward, is_done, _ = self.env.step(action)
            score += reward
        return score

    def plot_scores(self, scores: list, window=50) -> None:
        if window:
            scores = moving_average(scores, window_size=window)
        plt.plot(range(len(scores)), scores)
        plt.xlabel("Episode")
        plt.ylabel("Episode reward (smoothed {})".format(window))
        plt.show()
