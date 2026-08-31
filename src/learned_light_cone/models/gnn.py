import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiMeshGNN1d(nn.Module):
    """A minimal 1D Graph Neural Network demonstrating spatial leakage.

    This constructs a 'multi-mesh' topology by hardcoding message-passing edges
    at multiple spatial scales: local (distance 1), mid (distance N/8), and
    global (distance N/4). This mimics the multi-resolution icosahedral mesh
    hierarchy used in models like GraphCast, allowing us to empirically
    prove the Graph Light Cone hypothesis.
    """

    def __init__(self, in_channels=1, hidden_channels=32, out_channels=1, n_layers=4, N=128):
        super().__init__()
        self.n_layers = n_layers
        self.N = N

        self.in_proj = nn.Linear(in_channels, hidden_channels)
        self.out_proj = nn.Linear(hidden_channels, out_channels)

        # Each node aggregates its own state and the sum of its neighbors at each scale
        self.convs = nn.ModuleList(
            [
                nn.Linear(hidden_channels * 4, hidden_channels)  # self, local, mid, global
                for _ in range(n_layers)
            ]
        )

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        # Handle shape (B, N) -> (B, 1, N)
        is_1d = False
        if x.ndim == 2:
            is_1d = True
            x = x.unsqueeze(1)

        x = x.transpose(1, 2)  # (B, N, C)
        h = self.in_proj(x)

        for i in range(self.n_layers):
            # Gather messages from neighbors at different scales
            # Local: dist = 1
            h_left = torch.roll(h, shifts=1, dims=1)
            h_right = torch.roll(h, shifts=-1, dims=1)
            m_local = (h_left + h_right) / 2

            # Mid: dist = N // 8
            h_l8 = torch.roll(h, shifts=self.N // 8, dims=1)
            h_r8 = torch.roll(h, shifts=-self.N // 8, dims=1)
            m_mid = (h_l8 + h_r8) / 2

            # Global: dist = N // 4
            h_l4 = torch.roll(h, shifts=self.N // 4, dims=1)
            h_r4 = torch.roll(h, shifts=-self.N // 4, dims=1)
            m_global = (h_l4 + h_r4) / 2

            # Aggregate and activate
            concat = torch.cat([h, m_local, m_mid, m_global], dim=-1)
            h = F.relu(self.convs[i](concat))

        out = self.out_proj(h)  # (B, N, C)
        out = out.transpose(1, 2)  # (B, C, N)

        if is_1d:
            out = out.squeeze(1)

        return out
