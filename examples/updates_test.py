"""Test A2C with different update frequencies"""
import gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src import A2C
from utils import moving_average

# HYP -------------------------------------------------------------
FRAMES = 50_000  # number of training frames at which training will stop
MAX_EPISODES = (
    1e6
)  # number of episodes at which training will stop, if FRAMES hasn't already been reached

env = gym.make("CartPole-v1")
N_INPUTS = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n

UPDATES = [5, 20, 50, 100, 200, 1000, 5000]
all_scores = {}
all_rewards = {}

for update in tqdm(UPDATES):
    # TRAIN -----------------------------------------------------------
    agent = A2C(N_INPUTS, N_ACTIONS, env)
    rewards = agent.train(FRAMES, update, max_episodes=MAX_EPISODES, plot=False)
    all_rewards[update] = rewards

    # SCORE -----------------------------------------------------------
    scores = []
    for _ in range(100):
        scores.append(agent.play())
    all_scores[update] = np.mean(scores)

print(all_scores)

for update, rewards in all_rewards.items():
    rewards = moving_average(rewards, 10)
    plt.plot(range(len(rewards)), rewards, label=str(update))
    plt.xlabel("Episode")
    plt.ylabel("Episode reward (smoothed 10)")
    plt.legend()
plt.show()
