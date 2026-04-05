import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -------------------------
# Dataset (structured target)
# -------------------------
class EngineeringDataset:
    def sample(self, batch=64):
        x = torch.randn(batch, 5)

        # Define a meaningful target (non-trivial)
        target = torch.stack([
            x[:,0] * x[:,1],
            torch.sin(x[:,2]),
            x[:,3]**2,
            x[:,4] + x[:,0],
            x[:,1] - x[:,2]
        ], dim=1)

        return x, target

# -------------------------
# Model Components
# -------------------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 64)

    def forward(self, x):
        return torch.tanh(self.fc(x))

class TRIZLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(40)])

    def forward(self, x):
        return torch.stack([F.relu(l(x)) for l in self.layers], dim=1)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 40)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 5)

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

        return out, dist, actions, h

# -------------------------
# Metrics
# -------------------------
def compute_metrics(out, target, h):
    mse = F.mse_loss(out, target)
    ideality = 1 / (1 + mse)   # bounded, meaningful
    diversity = torch.var(h)
    innovation = ideality * diversity
    return mse, ideality, diversity, innovation

# -------------------------
# Training
# -------------------------
def train():
    model = TRIZRL()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = EngineeringDataset()

    ideality_hist = []
    diversity_hist = []
    innovation_hist = []

    baseline = 0

    for epoch in range(200):
        x, target = data.sample()

        out, dist, actions, h = model(x)

        mse, ideality, diversity, innovation = compute_metrics(out, target, h)

        # -------------------------
        # REWARD (aligned with task)
        # -------------------------
        reward = (ideality + 0.5 * diversity).detach()

        baseline = 0.9 * baseline + 0.1 * reward.item()
        advantage = reward - baseline

        # -------------------------
        # POLICY LOSS (correct)
        # -------------------------
        log_probs = dist.log_prob(actions)
        policy_loss = -torch.mean(log_probs * advantage)

        # -------------------------
        # SUPERVISED LOSS (anchor!)
        # -------------------------
        supervised_loss = mse

        # -------------------------
        # STRUCTURAL TRIZ LOSS
        # -------------------------
        structural_loss = torch.mean((h - h.flip(1))**2)

        # -------------------------
        # ENTROPY (avoid collapse)
        # -------------------------
        entropy = dist.entropy().mean()

        # -------------------------
        # FINAL LOSS
        # -------------------------
        loss = (
            supervised_loss
            + 0.5 * policy_loss
            + 0.1 * structural_loss
            - 0.01 * entropy
        )

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
    # FIXED PLOTS
    # -------------------------
    fig, axs = plt.subplots(3, 1)

    axs[0].plot(ideality_hist)
    axs[0].set_title("Ideality")

    axs[1].plot(diversity_hist)
    axs[1].set_title("Diversity")

    axs[2].plot(innovation_hist)
    axs[2].set_title("Innovation")

    plt.tight_layout()
    plt.show()

    print("\nFINAL OUTPUT:\n", out[0].detach().numpy())
    print("\nTARGET:\n", target[0].detach().numpy())

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    train()
