# sparrow/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import SparrowConfig

class Expert(nn.Module):
    """یک متخصص (Expert) که یک شبکه عصبی کوچک است."""
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.net(x)

class TwoTowerRouter(nn.Module):
    """مسیریاب دو برجی که شباهت ورودی را با تخصص هر متخصص می‌سنجد."""
    def __init__(self, input_size, num_experts, expert_embedding_dim):
        super().__init__()
        self.input_tower = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, expert_embedding_dim)
        )
        self.expert_tower = nn.Embedding(num_experts, expert_embedding_dim)

    def forward(self, x):
        input_embedding = self.input_tower(x)
        expert_embeddings = self.expert_tower.weight
        router_logits = input_embedding @ expert_embeddings.T
        return router_logits

class MoELayer(nn.Module):
    """لایه MoE که از مسیریاب Two-Tower و جریمه‌های هوشمند استفاده می‌کند."""
    def __init__(self, config: SparrowConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts = nn.ModuleList([
            Expert(config.hidden_dim, config.hidden_dim, config.expert_hidden_size) 
            for _ in range(self.num_experts)
        ])
        self.router = TwoTowerRouter(config.hidden_dim, self.num_experts, config.expert_embedding_dim)

    def forward(self, x):
        router_logits = self.router(x)
        gates = F.gumbel_softmax(router_logits, hard=True, dim=-1)
        router_probs = F.softmax(router_logits, dim=-1)

        # محاسبه زیان متعادل‌سازی (Load Balancing)
        fraction_of_tokens = gates.float().mean(dim=0)
        mean_router_prob = router_probs.mean(dim=0)
        load_balancing_loss = self.num_experts * torch.sum(fraction_of_tokens * mean_router_prob)

        # محاسبه انتروپی برای تشویق به اکتشاف
        entropy = -torch.mean(torch.sum(router_probs * torch.log(router_probs + 1e-9), dim=-1))
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gated_output = torch.bmm(gates.unsqueeze(1), expert_outputs).squeeze(1)
        
        return gated_output, load_balancing_loss, entropy