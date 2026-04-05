import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Dataset
# -------------------------
class EngineeringDataset:
    def sample(self, batch=64):
        return torch.randn(batch, 5)

# -------------------------
# Encoder
# -------------------------
class Encoder(nn.Module):
    def __init__(self, inp=5, hidden=64):
        super().__init__()
        self.fc = nn.Linear(inp, hidden)

    def forward(self, x):
        return torch.tanh(self.fc(x))

# -------------------------
# TRIZ Layer (40 principles)
# -------------------------
class TRIZLayer(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(40)])

    def forward(self, x):
        return torch.stack([F.relu(l(x)) for l in self.layers], dim=1)
        # shape: (B, 40, D)

# -------------------------
# Policy
# -------------------------
class Policy(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.fc = nn.Linear(dim, 40)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)  # (B, 40)

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
        z = self.enc(x)                  # (B, D)
        h = self.triz(z)                 # (B, 40, D)
        probs = self.policy(z)           # (B, 40)

        # Sample actions per batch element
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()          # (B,)

        # Select corresponding TRIZ outputs
        selected = h[torch.arange(h.size(0)), actions]  # (B, D)

        out = self.dec(selected)         # (B, 5)

        return out, probs, actions, dist, h

# -------------------------
# Reward
# -------------------------
def reward_fn(out):
    return torch.mean(torch.abs(out), dim=1) - torch.mean(out**2, dim=1)
    # shape: (B,)

# -------------------------
# Loss (TRIZ Laws simplified)
# -------------------------
def triz_losses(out, h):
    return (
        torch.mean(out**2) +                     # efficiency
        torch.mean((h - h.flip(1))**2) -         # opposites
        torch.var(h)                             # diversity
    )

# -------------------------
# Evaluation
# -------------------------
def evaluate(out, h):
    ideality = torch.mean(torch.abs(out)).item()
    diversity = torch.var(h).item()
    innovation = ideality * diversity
    return ideality, diversity, innovation

# -------------------------
# Patent Generator
# -------------------------
def patent(out):
    return f"""
TITLE: TRIZ-Based Adaptive Engineering System

ABSTRACT:
A neural system using TRIZ principles and reinforcement learning
to generate optimized engineering solutions.

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

        out, probs, actions, dist, h = model(x)

        rewards = reward_fn(out).detach()  # (B,)

        # Correct policy gradient loss
        log_probs = dist.log_prob(actions)  # (B,)
        policy_loss = -torch.mean(log_probs * rewards)

        # TRIZ structural losses
        structural_loss = triz_losses(out, h)

        loss = policy_loss + structural_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 20 == 0:
            i, d, inv = evaluate(out, h)
            print(f"Epoch {epoch} | Ideality {i:.3f} | Innovation {inv:.3f}")

    print(patent(out[0].detach().numpy()))

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    train()
