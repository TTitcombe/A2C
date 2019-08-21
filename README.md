# A2C

This repo contains a synchronous version of the (Asynchronous) Advantage Actor Critic (A3C) 
presented by [Mnih et al.](https://arxiv.org/pdf/1602.01783.pdf), written in PyTorch.

The [OpenAI baseline](https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py) was used as a reference.

## What is A2C?
A2C is a *policy gradient, Actor-critic* algorithm.

The **Critic** model estimates the value of a state.\
The **Actor** model enacts the policy, i.e. which actions to take.

In A2C, we estimate the **Advantage function** to reduce the variance of the policy gradient 
(taking fewer "bad" steps when we update the parameters).\
The Advantage function is the difference between our value-action estimates and value estimates.

Unlike DQN, A2C doesn't need expensive sweeps over a replay memory to train.

## To Run
There is a test script for running the algorithm on `CartPole-v1` OpenAI gym environment. 
When in the top level of this repository, run:
```bash
python -m examples.A2C_test 
```
This will train A2C for ~4'000 episodes, achieving an average score of around ~400 in ~30seconds on an average CPU.

### Results
Currently A2C has only been tested on `CartPole-v1`
![cartpole_results](results/A2C_CartPole.png)
As you can see, the agent improves well over the first few episodes, only becoming unstable when it gets an average 
episode reward of 120.