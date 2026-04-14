import torch
import torch.nn as nn
import math
import numpy as np

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        half_dim = self.embed_dim // 2
        device = t.device
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1) * emb  # [B, L, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # [B, L, embed_dim]

class FlowTransformer(nn.Module):
    def __init__(self,
                 argus,
                 state_dim,
                 action_dim,
                 rtg_dim=1,
                 embed_dim=256,
                 seq_len=20,
                 n_layers=6,
                 n_heads=8,
                 dropout=0.1):
        super().__init__()
        self.argus = argus
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        self.embed_rtg = nn.Linear(rtg_dim, embed_dim)
        self.embed_state = nn.Linear(state_dim, embed_dim)
        self.embed_action = nn.Linear(action_dim, embed_dim)

        self.time_embedding = SinusoidalTimeEmbedding(embed_dim)
        self.time_embed_proj = nn.Linear(embed_dim, embed_dim)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len * 4, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.predict_action = nn.Linear(embed_dim, action_dim)

    def forward(self, rtg, state, action, t):
        """
        rtg: [B, L, 1]
        state: [B, L, state_dim]
        action: [B, L, action_dim]
        t: [B, L] or [B, L, 1]
        """
        B, L, _ = state.shape

        rtg_emb = self.embed_rtg(rtg)
        state_emb = self.embed_state(state)
        action_emb = self.embed_action(action)

        t_emb = self.time_embedding(t)
        if len(t_emb.shape) == 4:
            t_emb = t_emb.squeeze(dim=-2)
        t_emb = self.time_embed_proj(t_emb)

        # Stack in (rtg, state, action, t) per time step, then flatten sequence
        stacked = torch.stack((rtg_emb, state_emb, action_emb, t_emb), dim=2)
        stacked = stacked.view(B, L * 4, self.embed_dim)

        x = stacked + self.pos_emb[:, :L * 4, :]
        x = x.permute(1, 0, 2)  # [L*4, B, D]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [B, L*4, D]

        # Select state tokens for action prediction: index 1, 5, 9, ... = 4 * t + 1
        idx = torch.arange(L, device=state.device) * 4 + 1
        x_state_tokens = x[:, idx, :]  # [B, L, D]
        pred_action = self.predict_action(x_state_tokens)  # [B, L, action_dim]
        return pred_action

    def gen_action(self, states, rtg=0.9, time_start=0.0, time_end=1.0, steps=10,
                   x_t_clip_value=None, step_anneal=False, anneal_rate=0.98, num_candidate=16, **kwargs):
        num_samples, L, _ = states.shape
        rtg = torch.full((num_samples, L, 1), rtg, dtype=torch.float32, device=self.argus.device)
        x = torch.randn(num_samples, L, self.argus.action_dim, device=self.argus.device)
        if x_t_clip_value is not None:
            x = torch.clamp(x, -x_t_clip_value, x_t_clip_value)
        dt = 1.0 / steps
        # for current_generation_time in range(self.argus.multi_stage_genration):
        step_index = 0
        for t in np.linspace(time_start, time_end, steps):  # 逐步演化
            # noise = torch.randn(num_samples, 2, device=self.argus.device) * 0.05
            t_tensor = torch.full((num_samples, L, 1), t, dtype=torch.float32, device=self.argus.device)
            v = self.forward(rtg=rtg, state=states, action=x, t=t_tensor)  # * self.energy_model(x) * scale
            if step_anneal:
                dt *= anneal_rate
            x = x + v * dt * self.argus.flow_step_scale
            if x_t_clip_value is not None:
                x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
            step_index += 1

        return x[:, -1, :]


class QFlowTransformer(nn.Module):
    def __init__(self,
                 argus,
                 state_dim,
                 action_dim,
                 embed_dim=256,
                 seq_len=20,
                 n_layers=6,
                 n_heads=8,
                 dropout=0.1):
        super().__init__()
        self.argus = argus
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.token_num = 3

        self.embed_state = nn.Linear(state_dim, embed_dim)
        self.embed_action = nn.Linear(action_dim, embed_dim)

        self.time_embedding = SinusoidalTimeEmbedding(embed_dim)
        self.time_embed_proj = nn.Linear(embed_dim, embed_dim)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len * self.token_num, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * self.token_num,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.predict_action = nn.Linear(embed_dim, action_dim)

    def forward(self, state, action, t):
        """
        rtg: [B, L, 1]
        state: [B, L, state_dim]
        action: [B, L, action_dim]
        t: [B, L] or [B, L, 1]
        """
        B, L, _ = state.shape

        state_emb = self.embed_state(state)
        action_emb = self.embed_action(action)

        t_emb = self.time_embedding(t)
        if len(t_emb.shape) == 4:
            t_emb = t_emb.squeeze(dim=-2)
        t_emb = self.time_embed_proj(t_emb)

        # Stack in (rtg, state, action, t) per time step, then flatten sequence
        stacked = torch.stack((state_emb, action_emb, t_emb), dim=2)
        stacked = stacked.view(B, L * self.token_num, self.embed_dim)

        x = stacked + self.pos_emb[:, :L * self.token_num, :]
        x = x.permute(1, 0, 2)  # [L*4, B, D]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [B, L*4, D]

        # Select state tokens for action prediction: index 1, 5, 9, ... = 4 * t + 1
        idx = torch.arange(L, device=state.device) * self.token_num + 1
        x_state_tokens = x[:, idx, :]  # [B, L, D]
        pred_action = self.predict_action(x_state_tokens)  # [B, L, action_dim]
        return pred_action

    def gen_action(self, states, rtg=0.9, time_start=0.0, time_end=1.0, steps=10,
                   x_t_clip_value=None, step_anneal=False, anneal_rate=0.98, num_candidate=16, **kwargs):
        num_samples, L, _ = states.shape
        rtg = torch.full((num_samples, L, 1), rtg, dtype=torch.float32, device=self.argus.device)
        x = torch.randn(num_samples, L, self.argus.action_dim, device=self.argus.device)
        if x_t_clip_value is not None:
            x = torch.clamp(x, -x_t_clip_value, x_t_clip_value)
        dt = 1.0 / steps
        # for current_generation_time in range(self.argus.multi_stage_genration):
        step_index = 0
        for t in np.linspace(time_start, time_end, steps):  # 逐步演化
            # noise = torch.randn(num_samples, 2, device=self.argus.device) * 0.05
            t_tensor = torch.full((num_samples, L, 1), t, dtype=torch.float32, device=self.argus.device)
            v = self.forward(state=states, action=x, t=t_tensor)  # * self.energy_model(x) * scale
            if step_anneal:
                dt *= anneal_rate
            x = x + v * dt * self.argus.flow_step_scale
            if x_t_clip_value is not None:
                x = x.clamp(-self.argus.x_t_clip_value, self.argus.x_t_clip_value)
            step_index += 1

        return x[:, -1, :]
