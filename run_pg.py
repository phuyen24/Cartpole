#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import numpy as np
import gym
from PG import PGN

env = gym.make("CartPole-v1")

net = PGN(env.observation_space.shape[0], env.action_space.n)
net.load_state_dict(torch.load("cartpole-pg.pth"))
net.eval()

obs = env.reset()
total_reward = 0.0

while True:
    obs_v = torch.tensor([obs], dtype=torch.float32)
    with torch.no_grad():
        probs = F.softmax(net(obs_v), dim=1).numpy()[0]
    action = int(np.argmax(probs))

    env.render()
    obs, reward, done, _ = env.step(action)
    total_reward += reward

    if done:
        print(f"Total reward: {total_reward:.1f}")
        break

env.close()
