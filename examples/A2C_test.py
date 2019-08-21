"""Test A2C on the simple CartPole env"""
import time

import gym
import numpy as np

from src import A2C

# HYP -------------------------------------------------------------
FRAMES = 50_000  # number of training frames at which training will stop
UPDATE_EVERY = 20  # number of frames before an episode is stopped
MAX_EPISODES = (
    40_000
)  # number of episodes at which training will stop, if FRAMES hasn't already been reached

env = gym.make("CartPole-v1")
N_INPUTS = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n

# TRAIN -----------------------------------------------------------
agent = A2C(N_INPUTS, N_ACTIONS, env)
start = time.time()
agent.train(FRAMES, UPDATE_EVERY, max_episodes=MAX_EPISODES)
end = time.time()

print("It took {:.3f}s to train".format(end - start))

# SCORE -----------------------------------------------------------
scores = []
for _ in range(100):
    scores.append(agent.play())
print("Average score: {}".format(np.mean(scores)))
