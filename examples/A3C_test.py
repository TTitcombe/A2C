"""Test A3C on the simple CartPole env"""
import time

import numpy as np

from src import A3C


def main(env, frames, update_every, n_inputs, n_actions, n_processes):
    # TRAIN -----------------------------------------------------------
    agent = A3C(n_inputs, n_actions, env)
    start = time.time()
    agent.train(frames, update_every, num_processes=n_processes)
    end = time.time()

    print("It took {:.3f}s to train".format(end - start))

    # SCORE -----------------------------------------------------------
    scores = []
    for _ in range(100):
        scores.append(agent.play())
    print("Average score: {}".format(np.mean(scores)))


if __name__ == "__main__":
    # This is necessary for running on Windows.
    # See https://pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection
    # for more information

    FRAMES = 50_000  # number of training frames at which training will stop
    UPDATE_EVERY = 20  # number of frames before an episode is stopped

    env = "CartPole-v1"
    N_INPUTS = 4
    N_ACTIONS = 2
    N_PROCESSES = 2

    main(env, FRAMES, UPDATE_EVERY, N_INPUTS, N_ACTIONS, N_PROCESSES)
