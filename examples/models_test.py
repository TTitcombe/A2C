"""Test A2C with a shared Actor/Critic model and completely separate weights"""
import pickle
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from A2C import A2C
from utils import moving_average

# HYP -------------------------------------------------------------
FRAMES = 50_000  # number of training frames at which training will stop
MAX_EPISODES = (
    1e6
)  # number of episodes at which training will stop, if FRAMES hasn't already been reached
UPDATE = 20

env = gym.make("CartPole-v1")
N_INPUTS = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n

all_scores = {}
all_rewards = {}

for separate in tqdm([True, False]):
    # TRAIN -----------------------------------------------------------
    agent = A2C(N_INPUTS, N_ACTIONS, env, separate_models=separate)
    rewards = agent.train(FRAMES, UPDATE, max_episodes=MAX_EPISODES, plot=False)
    all_rewards[separate] = rewards

    # SCORE -----------------------------------------------------------
    scores = []
    for _ in range(100):
        scores.append(agent.test())
    all_scores[separate] = np.mean(scores)

print(all_scores)
path = os.path.join("results", "A2C_model_test_results.pickle")
with open(path, "wb") as handle:
    pickle.dump(all_scores, handle)

for separate, rewards in all_rewards.items():
    rewards = moving_average(rewards, 50)
    if separate:
        label = "Separate models"
    else:
        label = "Shared weights"
    plt.plot(range(len(rewards)), rewards, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Episode reward (smoothed 50)")
    plt.legend()
plt.show()
