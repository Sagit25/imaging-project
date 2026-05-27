import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(emb_dim, channels)
        
    def forward(self, x, t_emb):
        h = F.silu(self.conv1(x))
        t_proj = self.time_emb_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_proj
        h = self.conv2(F.silu(h))
        return x + h

class FlowUNet(nn.Module):
    def __init__(self, in_channels=3, cond_channels=4, base_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, base_dim * 4), nn.SiLU(), nn.Linear(base_dim * 4, base_dim * 4)
        )
        
        # Down
        self.init_conv = nn.Conv2d(in_channels + cond_channels, base_dim, 3, padding=1)
        self.down1 = ResBlock(base_dim, base_dim * 4)
        self.down2 = nn.Conv2d(base_dim, base_dim * 2, 4, stride=2, padding=1)
        self.down3 = ResBlock(base_dim * 2, base_dim * 4)
        
        # Mid
        self.mid1 = ResBlock(base_dim * 2, base_dim * 4)
        
        # Up
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, 4, stride=2, padding=1)
        self.up2 = ResBlock(base_dim * 2, base_dim * 4) # Skip connection concat consideration
        self.final_conv = nn.Conv2d(base_dim, in_channels, 3, padding=1)

    def forward(self, x_t, t, condition):
        t_emb = self.time_mlp(t.view(-1, 1))
        
        x = torch.cat([x_t, condition], dim=1)
        x = self.init_conv(x)
        
        d1 = self.down1(x, t_emb)
        d2 = self.down2(d1)
        d3 = self.down3(d2, t_emb)
        
        m = self.mid1(d3, t_emb)
        
        u1 = self.up1(m)
        u1_concat = torch.cat([u1, d1], dim=1)
        u2 = self.up2(u1_concat, t_emb)
        
        return self.final_conv(u2)

class CFGFlowMatcher:
    def __init__(self, model, cfg_drop_rate=0.1):
        self.model = model
        self.cfg_drop_rate = cfg_drop_rate

    def compute_loss(self, x_1, condition):
        B = x_1.shape[0]
        device = x_1.device

        # Flow Matching Optimal Transport Path
        t = torch.rand(B, 1, 1, 1, device=device)
        x_0 = torch.randn_like(x_1)
        x_t = t * x_1 + (1 - t) * x_0
        target_v = x_1 - x_0

        # CFG Condition Dropout
        if self.cfg_drop_rate > 0 and self.model.training:
            drop_mask = (torch.rand(B, 1, 1, 1, device=device) > self.cfg_drop_rate).float()
            condition = condition * drop_mask

        pred_v = self.model(x_t, t.squeeze(), condition)
        return F.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def sample(self, condition, num_steps=50, cfg_scale=3.0):
        B, C, H, W = condition.shape
        device = condition.device
        
        x_t = torch.randn(B, 3, H, W, device=device)
        null_condition = torch.zeros_like(condition)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            
            v_cond = self.model(x_t, t, condition)
            v_uncond = self.model(x_t, t, null_condition)
            v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
            
            # Euler Integration
            x_t = x_t + v_cfg * dt
            
        return x_t