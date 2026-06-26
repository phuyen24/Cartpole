#!/usr/bin/env python3
import sys
import math
import gym
import ptan
import numpy as np
import typing as tt
import pygame
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.001
EPISODES_TO_TRAIN = 4
ENTROPY_BETA = 0.01
SOLVED_MEAN_REWARD = 475
RENDER = False

# ── Palette ──────────────────────────────────────────────────────────────────
BG      = (247, 250, 252)
CART_C  = ( 27,  43,  91)
POLE_C  = (255, 127,  80)
TRACK_C = ( 24,  28,  30)
PIVOT_C = (235, 238, 240)
PANEL   = ( 36,  46,  63)
TXT_DIM = (189, 199, 220)
TXT_WHT = (255, 255, 255)
GREEN   = ( 16, 185, 129)
BORDER  = (197, 198, 208)

W, H     = 900, 560
TRACK_Y  = H - 100
CART_W   = 96
CART_H   = 48
POLE_LEN = 180
POLE_W   = 5
PIVOT_R  = 8


class PGN(nn.Module):
    def __init__(self, input_size: int, n_actions: int):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def calc_qvals(rewards: tt.List[float]) -> tt.List[float]:
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    res = list(reversed(res))
    mean_q = np.mean(res)
    std_q = np.std(res) + 1e-8
    return [(q - mean_q) / std_q for q in res]


def draw(screen, obs, step, episode, reward, mean_100, fonts):
    f_hdr, f_lbl, f_dat = fonts
    screen.fill(BG)

    # Track
    pygame.draw.line(screen, TRACK_C, (0, TRACK_Y),     (W, TRACK_Y),     1)
    pygame.draw.line(screen, TRACK_C, (0, TRACK_Y + 8), (W, TRACK_Y + 8), 1)

    # Cart
    scale = (W // 2 - CART_W) / 2.4
    cx    = int(W // 2 + obs[0] * scale)
    top_y = TRACK_Y - CART_H
    pygame.draw.rect(screen, CART_C,
                     pygame.Rect(cx - CART_W // 2, top_y, CART_W, CART_H),
                     border_radius=4)

    # Pole
    piv_x = cx
    piv_y = top_y + CART_H // 2
    angle  = obs[2]
    end_x  = int(piv_x + POLE_LEN * math.sin(angle))
    end_y  = int(piv_y - POLE_LEN * math.cos(angle))
    pygame.draw.line(screen, POLE_C, (piv_x, piv_y), (end_x, end_y), POLE_W)

    # Pivot
    pygame.draw.circle(screen, PIVOT_C, (piv_x, piv_y), PIVOT_R)

    # Stats panel
    PW, PH, PX, PY = 240, 180, 24, 24
    surf = pygame.Surface((PW, PH), pygame.SRCALPHA)
    surf.fill((*PANEL, 210))
    screen.blit(surf, (PX, PY))
    pygame.draw.rect(screen, (*BORDER, 50), (PX, PY, PW, PH), 1, border_radius=8)

    y = PY + 14
    screen.blit(f_hdr.render("SIMULATION STATS", True, TXT_DIM), (PX + 16, y))
    y += 28

    def row(label, value, color=TXT_WHT):
        lbl = f_lbl.render(label, True, TXT_DIM)
        val = f_dat.render(value, True, color)
        screen.blit(lbl, (PX + 16, y + 2))
        screen.blit(val, (PX + PW - val.get_width() - 16, y))

    row("Step",    str(step));                y += 28
    row("Episode", str(episode));             y += 28
    pygame.draw.line(screen, BORDER,
                     (PX + 16, y), (PX + PW - 16, y), 1)
    y += 12
    row("Reward",   f"{reward:.1f}",  GREEN); y += 28
    row("Mean_100", f"{mean_100:.1f}", GREEN)

    pygame.display.flip()


if __name__ == "__main__":
    if RENDER:
        pygame.init()
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("CartPole — Policy Gradient Training")
        clock = pygame.time.Clock()
        f_hdr = pygame.font.SysFont("Arial", 11, bold=True)
        f_lbl = pygame.font.SysFont("Arial", 12)
        f_dat = pygame.font.SysFont("Arial", 18, bold=True)
        fonts = (f_hdr, f_lbl, f_dat)

    env = gym.make("CartPole-v1")
    writer = SummaryWriter(comment="-cartpole-reinforce-baseline")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,
                                   apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_idx = 0
    done_episodes = 0
    last_reward = 0.0
    mean_rewards = 0.0

    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_states, cur_actions, cur_rewards = [], [], []

    for step_idx, exp in enumerate(exp_source):
        if RENDER:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    writer.close()
                    pygame.quit()
                    sys.exit()
            draw(screen, exp.state, step_idx, done_episodes,
                 last_reward, mean_rewards, fonts)
            clock.tick(60)

        cur_states.append(exp.state)
        cur_actions.append(int(exp.action))
        cur_rewards.append(exp.reward)

        if exp.last_state is None:
            batch_states.extend(cur_states)
            batch_actions.extend(cur_actions)
            batch_qvals.extend(calc_qvals(cur_rewards))
            cur_states.clear()
            cur_actions.clear()
            cur_rewards.clear()
            batch_episodes += 1

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            last_reward = new_rewards[0]
            total_rewards.append(last_reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, last_reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", last_reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards >= SOLVED_MEAN_REWARD:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                torch.save(net.state_dict(), "cartpole-pg.pth")
                break

        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        states_v = torch.as_tensor(np.asarray(batch_states), dtype=torch.float32)
        batch_actions_t = torch.as_tensor(batch_actions, dtype=torch.int64)
        batch_qvals_v = torch.as_tensor(batch_qvals, dtype=torch.float32)

        optimizer.zero_grad()
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        loss_policy = -log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss = -ENTROPY_BETA * entropy_v
        loss_v = loss_policy + entropy_loss

        loss_v.backward()
        optimizer.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

    writer.close()
    env.close()
    if RENDER:
        pygame.quit()
