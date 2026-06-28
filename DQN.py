#!/usr/bin/env python3
import argparse
import csv
import math
import os
import random
import sys
import time

import gym
import numpy as np
import ptan
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

# ── CLI args ──────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--seed", type=int, default=None,
                     help="Seed value (any non-negative integer). Omit to use next unfinished seed from SEEDS.")
_parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                     help="List of seeds to cycle through (default: 0 1 2 3 4).")
_args, _ = _parser.parse_known_args()

# ── Experiment config ─────────────────────────────────────────────────────────
SEEDS              = _args.seeds
MAX_STEPS          = 1_500_000
ALGO               = "DQN"
SOLVED_MEAN_REWARD = 475

# ── Hyper-parameters ──────────────────────────────────────────────────────────
GAMMA         = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE    = 32

EPSILON_START = 1.0
EPSILON_STOP  = 0.02
EPSILON_STEPS = 5000

REPLAY_BUFFER = 50000
TARGET_SYNC   = 1000
RENDER        = True

# ── Palette ───────────────────────────────────────────────────────────────────
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


class DQN(nn.Module):
    def __init__(self, input_size: int, n_actions: int):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def calc_target(target_net: DQN, local_reward: float, next_state: np.ndarray) -> float:
    if next_state is None:
        return local_reward

    state_v = torch.tensor([next_state], dtype=torch.float32)

    with torch.no_grad():
        next_q_v = target_net(state_v)
        best_q = next_q_v.max(dim=1)[0].item()

    return local_reward + GAMMA * best_q


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def draw(screen, obs, step, episode, reward, mean_100, epsilon, fonts):
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
    PW, PH, PX, PY = 240, 210, 24, 24
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

    row("Step",    str(step));                 y += 28
    row("Episode", str(episode));              y += 28
    row("Epsilon", f"{epsilon:.2f}");          y += 28
    pygame.draw.line(screen, BORDER,
                     (PX + 16, y), (PX + PW - 16, y), 1)
    y += 12
    row("Reward",   f"{reward:.1f}",  GREEN);  y += 28
    row("Mean_100", f"{mean_100:.1f}", GREEN)

    pygame.display.flip()


def run_seed(seed: int) -> dict:
    set_seed(seed)

    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    try:
        env.reset(seed=seed)
    except TypeError:
        env.seed(seed)

    tb_writer = SummaryWriter(comment=f"-cartpole-dqn-seed{seed}")

    net = DQN(env.observation_space.shape[0], env.action_space.n)
    target_net = DQN(env.observation_space.shape[0], env.action_space.n)
    target_net.load_state_dict(net.state_dict())
    print(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    replay_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_BUFFER)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    mse_loss = nn.MSELoss()

    total_rewards = []
    step_idx = 0
    done_episodes = 0
    last_reward = 0.0
    mean_rewards = 0.0
    solved = False
    history = []

    t_start = time.time()

    while True:
        if RENDER:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    tb_writer.close()
                    pygame.quit()
                    sys.exit()

        step_idx += 1

        if step_idx >= MAX_STEPS:
            solved = False
            break

        selector.epsilon = max(EPSILON_STOP, EPSILON_START - step_idx / EPSILON_STEPS)
        replay_buffer.populate(1)

        if RENDER and replay_buffer.buffer:
            obs = replay_buffer.buffer[-1].state
            draw(screen, obs, step_idx, done_episodes,
                 last_reward, mean_rewards, selector.epsilon, fonts)
            clock.tick(60)

        if len(replay_buffer) < BATCH_SIZE:
            continue

        # sample batch
        batch = replay_buffer.sample(BATCH_SIZE)
        batch_states  = [exp.state  for exp in batch]
        batch_actions = [exp.action for exp in batch]
        batch_targets = [calc_target(target_net, exp.reward, exp.last_state)
                         for exp in batch]
        # train
        optimizer.zero_grad()
        states_v   = torch.as_tensor(np.asarray(batch_states), dtype=torch.float32)
        net_q_v    = net(states_v)
        target_q   = net_q_v.detach().numpy().copy()
        target_q[range(BATCH_SIZE), batch_actions] = batch_targets
        target_q_v = torch.as_tensor(target_q, dtype=torch.float32)
        loss_v     = mse_loss(net_q_v, target_q_v)
        loss_v.backward()
        optimizer.step()
        if step_idx % TARGET_SYNC == 0:
            target_net.load_state_dict(net.state_dict())

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            last_reward    = new_rewards[0]
            total_rewards.append(last_reward)
            mean_rewards   = float(np.mean(total_rewards[-100:]))
            history.append({
                "episode":  done_episodes,
                "step":     step_idx,
                "reward":   last_reward,
                "mean_100": round(mean_rewards, 2),
            })
            print("%d: reward: %6.2f, mean_100: %6.2f, epsilon: %.2f, episodes: %d" % (
                step_idx, last_reward, mean_rewards, selector.epsilon, done_episodes))
            tb_writer.add_scalar("reward",     last_reward,      step_idx)
            tb_writer.add_scalar("reward_100", mean_rewards,     step_idx)
            tb_writer.add_scalar("epsilon",    selector.epsilon, step_idx)
            tb_writer.add_scalar("episodes",   done_episodes,    step_idx)
            if mean_rewards >= SOLVED_MEAN_REWARD:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                solved = True
                break

    wall_time = round(time.time() - t_start, 1)

    model_dir = os.path.join("models", ALGO)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(model_dir, f"cartpole-{ALGO.lower()}-seed{seed}.pth"))

    hist_dir = os.path.join("history", ALGO)
    os.makedirs(hist_dir, exist_ok=True)
    csv_path = os.path.join(hist_dir, f"history_{ALGO}_seed{seed}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "step", "reward", "mean_100"])
        w.writeheader()
        w.writerows(history)
    print(f"History saved → {csv_path}")

    tb_writer.close()
    env.close()

    return {
        "algo":     ALGO,
        "seed":     seed,
        "episodes": done_episodes,
        "steps":    step_idx,
        "time_s":   wall_time,
        "mean100":  round(mean_rewards, 2),
        "std":      round(float(np.std(total_rewards[-100:])), 2),
        "solved":   solved,
    }


if __name__ == "__main__":
    if RENDER:
        pygame.init()
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("CartPole — DQN Training")
        clock = pygame.time.Clock()
        f_hdr = pygame.font.SysFont("Arial", 11, bold=True)
        f_lbl = pygame.font.SysFont("Arial", 12)
        f_dat = pygame.font.SysFont("Arial", 18, bold=True)
        fonts = (f_hdr, f_lbl, f_dat)

    results_file = "results.csv"
    fieldnames   = ["algo", "seed", "episodes", "steps", "time_s", "mean100", "std", "solved"]

    if _args.seed is not None:
        next_seed = _args.seed
    else:
        done_seeds = set()
        if os.path.exists(results_file):
            with open(results_file, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    if row.get("algo", "").strip() == ALGO:
                        done_seeds.add(int(row["seed"]))

        next_seed = next((s for s in SEEDS if s not in done_seeds), None)
        if next_seed is None:
            next_seed = max(done_seeds) + 1 if done_seeds else 0

    print(f"\n{'='*40}\nRunning {ALGO} seed {next_seed}\n{'='*40}")
    result = run_seed(next_seed)

    file_exists = os.path.exists(results_file)
    with open(results_file, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(result)
    print(f"Result for seed {next_seed} appended to {results_file}")

    if RENDER:
        pygame.quit()
