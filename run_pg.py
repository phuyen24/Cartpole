#!/usr/bin/env python3
import sys
import math
import torch
import torch.nn.functional as F
import numpy as np
import gym
import pygame
from PG import PGN, SOLVED_MEAN_REWARD

# ── Palette 
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

W, H    = 900, 560
TRACK_Y = H - 100
CART_W  = 96
CART_H  = 48
POLE_LEN = 180
POLE_W   = 5
PIVOT_R  = 8


def draw(screen, obs, step, episode, reward, mean_100, fonts):
    f_hdr, f_lbl, f_dat = fonts
    screen.fill(BG)

    # ── Track ─────────────────────────────────────────────────────────────────
    pygame.draw.line(screen, TRACK_C, (0, TRACK_Y),     (W, TRACK_Y),     1)
    pygame.draw.line(screen, TRACK_C, (0, TRACK_Y + 8), (W, TRACK_Y + 8), 1)

    # ── Cart ──────────────────────────────────────────────────────────────────
    scale = (W // 2 - CART_W) / 2.4
    cx    = int(W // 2 + obs[0] * scale)
    top_y = TRACK_Y - CART_H
    pygame.draw.rect(screen, CART_C,
                     pygame.Rect(cx - CART_W // 2, top_y, CART_W, CART_H),
                     border_radius=4)

    # ── Pole ──────────────────────────────────────────────────────────────────
    piv_x = cx
    piv_y = top_y + CART_H // 2
    angle  = obs[2]
    end_x  = int(piv_x + POLE_LEN * math.sin(angle))
    end_y  = int(piv_y - POLE_LEN * math.cos(angle))
    pygame.draw.line(screen, POLE_C, (piv_x, piv_y), (end_x, end_y), POLE_W)

    # ── Pivot circle ──────────────────────────────────────────────────────────
    pygame.draw.circle(screen, PIVOT_C, (piv_x, piv_y), PIVOT_R)

    # ── Stats panel ───────────────────────────────────────────────────────────
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

    if mean_100 > SOLVED_MEAN_REWARD:
        txt = f_dat.render("SOLVED!", True, GREEN)
        screen.blit(txt, (W // 2 - txt.get_width() // 2, H // 2 - 20))

    pygame.display.flip()


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("CartPole — Policy Gradient")
    clock = pygame.time.Clock()

    f_hdr = pygame.font.SysFont("Arial", 11, bold=True)
    f_lbl = pygame.font.SysFont("Arial", 12)
    f_dat = pygame.font.SysFont("Arial", 18, bold=True)
    fonts = (f_hdr, f_lbl, f_dat)

    env = gym.make("CartPole-v1")
    net = PGN(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(torch.load("cartpole-pg.pth"))
    net.eval()

    total_rewards = []
    episode = 0

    while True:
        obs       = env.reset()
        episode  += 1
        step      = 0
        ep_reward = 0.0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    sys.exit()

            obs_v = torch.tensor([obs], dtype=torch.float32)
            with torch.no_grad():
                probs = F.softmax(net(obs_v), dim=1).numpy()[0]
            action = int(np.argmax(probs))

            obs, reward, done, _ = env.step(action)
            step      += 1
            ep_reward += reward

            mean_100 = float(np.mean(total_rewards[-100:])) if total_rewards else ep_reward
            draw(screen, obs, step, episode, ep_reward, mean_100, fonts)
            clock.tick(60)

            if done:
                total_rewards.append(ep_reward)
                break

        mean_100 = float(np.mean(total_rewards[-100:])) if total_rewards else 0.0
        if mean_100 > SOLVED_MEAN_REWARD:
            print("Solved! Mean_100: %.1f after %d episodes" % (mean_100, episode))
            pygame.time.wait(2000)
            break

    env.close()
    pygame.quit()
