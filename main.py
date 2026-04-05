import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
# TRIZ Layer
# -------------------------
class TRIZLayer(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(40)])

    def forward(self, x):
        return torch.stack([F.relu(l(x)) for l in self.layers], dim=1)

# -------------------------
# Policy
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

        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()

        selected = h[torch.arange(h.size(0)), actions]
        out = self.dec(selected)

        return out, dist, h

# -------------------------
# Metrics
# -------------------------
def compute_metrics(out, h):
    ideality = torch.mean(torch.abs(out))
    diversity = torch.var(h)
    innovation = ideality * diversity
    return ideality, diversity, innovation

# -------------------------
# Training
# -------------------------
def train():
    model = TRIZRL()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    data = EngineeringDataset()

    ideality_hist = []
    diversity_hist = []
    innovation_hist = []

    baseline = 0

    for epoch in range(200):
        x = data.sample()

        out, dist, h = model(x)

        ideality, diversity, innovation = compute_metrics(out, h)

        # Strong reward
        reward = (ideality + diversity + innovation).detach()

        # Baseline (moving average)
        baseline = 0.9 * baseline + 0.1 * reward.item()
        advantage = reward - baseline

        # Policy loss
        log_probs = dist.log_prob(dist.sample())
        policy_loss = -torch.mean(log_probs * advantage)

        # Entropy bonus (exploration)
        entropy = dist.entropy().mean()

        # Structural loss (TRIZ constraints)
        structural_loss = (
            torch.mean(out**2)
            + torch.mean((h - h.flip(1))**2)
        )

        loss = policy_loss + structural_loss - 0.01 * entropy

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store metrics
        ideality_hist.append(ideality.item())
        diversity_hist.append(diversity.item())
        innovation_hist.append(innovation.item())

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Ideality {ideality:.3f} | Innovation {innovation:.3f}")

    # -------------------------
    # Plot (NO COLORS SPECIFIED)
    # -------------------------
    plt.figure()

    plt.subplot(3,1,1)
    plt.plot(ideality_hist)
    plt.title("Ideality")

    plt.subplot(3,1,2)
    plt.plot(diversity_hist)
    plt.title("Diversity")

    plt.subplot(3,1,3)
    plt.plot(innovation_hist)
    plt.title("Innovation")

    plt.tight_layout()
    plt.show()

    print("\nFINAL SAMPLE OUTPUT:\n", out[0].detach().numpy())

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    train()
