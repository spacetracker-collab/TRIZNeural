import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# -------------------------
# Dataset (Engineering Benchmark)
# -------------------------
class EngineeringDataset:
    def sample(self, batch=64):
        # features: cost, efficiency, energy, durability, complexity
        return torch.randn(batch, 5)

# -------------------------
# Abstract Encoder
# -------------------------
class Encoder(nn.Module):
    def __init__(self, inp=5, hidden=64):
        super().__init__()
        self.fc = nn.Linear(inp, hidden)

    def forward(self, x):
        return torch.tanh(self.fc(x))

# -------------------------
# 40 TRIZ Principles
# -------------------------
class TRIZLayer(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(40)])

    def forward(self, x):
        return torch.stack([F.relu(l(x)) for l in self.layers], dim=1)

# -------------------------
# RL Policy
# -------------------------
class Policy(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.fc = nn.Linear(dim, 40)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)

# -------------------------
# Decoder
# -------------------------
class Decoder(nn.Module):
    def __init__(self, dim=64, out=5):
        super().__init__()
        self.fc = nn.Linear(dim, out)

    def forward(self, x):
        return self.fc(x)

# -------------------------
# Full Model
# -------------------------
class TRIZRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.triz = TRIZLayer()
        self.policy = Policy()
        self.dec = Decoder()

    def forward(self, x):
        z = self.enc(x)
        h = self.triz(z)
        probs = self.policy(z)

        idx = torch.multinomial(probs, 1).squeeze()
        selected = h[torch.arange(h.size(0)), idx]

        out = self.dec(selected)
        return out, probs, h

# -------------------------
# TRIZ Laws (15 Loss Terms)
# -------------------------
def loss_ifr(out):
    return (out**2).mean() / (torch.abs(out).mean() + 1e-6)

def loss_opposites(h):
    return torch.mean((h - h.flip(1))**2)

def loss_efficiency(out):
    return torch.mean(out**2)

def loss_dynamization(h):
    return torch.var(h)

def loss_ideality(out):
    return torch.mean(torch.abs(out - 1))

def loss_interpenetration(h):
    return torch.mean(h.std(dim=1))

def loss_coordination(h):
    return torch.mean((h.mean(dim=1) - h)**2)

def loss_simplicity(out):
    return torch.mean(torch.abs(out))

def loss_energy(out):
    return torch.mean(out**2)

def loss_stability(h):
    return torch.var(h.mean(dim=1))

def loss_diversity(h):
    return -torch.var(h)

def loss_nonuniformity(h):
    return torch.mean(torch.abs(h - h.mean()))

def loss_transition(h):
    return torch.mean(torch.diff(h, dim=1)**2)

def loss_continuity(h):
    return torch.mean(h**2)

def loss_hierarchy(h):
    return torch.mean(h.mean(dim=1))

# -------------------------
# Reward
# -------------------------
def reward(out):
    return torch.mean(torch.abs(out)) - torch.mean(out**2)

# -------------------------
# Evaluation Metrics
# -------------------------
def evaluate(out, h):
    ideality = torch.mean(torch.abs(out))
    diversity = torch.var(h)
    innovation = diversity * ideality
    return ideality.item(), diversity.item(), innovation.item()

# -------------------------
# Patent Generator
# -------------------------
def patent(out):
    return f"""
TITLE: TRIZ-Based Adaptive Engineering System

ABSTRACT:
A neural system that generates optimized engineering solutions using inventive principles.

CLAIM:
Dynamic selection of principles via reinforcement learning improves ideality.

RESULT:
{out}
"""

# -------------------------
# Training
# -------------------------
def train():
    model = TRIZRL()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = EngineeringDataset()

    for epoch in range(100):
        x = data.sample()

        out, probs, h = model(x)
        R = reward(out).detach()

        logp = torch.log(probs + 1e-8)

        loss = (
            -torch.mean(logp * R)
            + loss_ifr(out)
            + loss_opposites(h)
            + loss_efficiency(out)
            + loss_dynamization(h)
            + loss_ideality(out)
            + loss_interpenetration(h)
            + loss_coordination(h)
            + loss_simplicity(out)
            + loss_energy(out)
            + loss_stability(h)
            + loss_diversity(h)
            + loss_nonuniformity(h)
            + loss_transition(h)
            + loss_continuity(h)
            + loss_hierarchy(h)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 20 == 0:
            i, d, inv = evaluate(out, h)
            print(f"Epoch {epoch} | Ideality {i:.3f} | Innovation {inv:.3f}")

    print(patent(out[0].detach().numpy()))

if __name__ == "__main__":
    train()
