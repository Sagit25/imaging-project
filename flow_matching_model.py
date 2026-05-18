import torch
import torch.nn as nn
import torch.nn.functional as F


# Basic unet과 이를 토대로 cfg flow matcher 구현 (diffusion x), 추후 실제 모델에 맞게 수정 필요!

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, cond_channels=4, base_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, base_dim * 4), nn.SiLU(), nn.Linear(base_dim * 4, base_dim * 4)
        )
        
        # Down
        self.down1 = nn.Conv2d(in_channels + cond_channels, base_dim, 3, padding=1)
        self.down2 = nn.Conv2d(base_dim, base_dim * 2, 4, stride=2, padding=1)
        
        # Mid
        self.time_emb_proj = nn.Linear(base_dim * 4, base_dim * 2)
        self.mid1 = nn.Conv2d(base_dim * 2, base_dim * 2, 3, padding=1)
        self.mid2 = nn.Conv2d(base_dim * 2, base_dim * 2, 3, padding=1)
        
        # Up
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, 4, stride=2, padding=1)
        self.up2 = nn.Conv2d(base_dim * 2, in_channels, 3, padding=1)

    def forward(self, x_t, t, condition):
        # Time embedding
        t_emb = self.time_mlp(t.view(-1, 1))
        
        # Concat condition
        x = torch.cat([x_t, condition], dim=1)
        
        # Downpass
        d1 = F.silu(self.down1(x))
        d2 = F.silu(self.down2(d1))
        
        # Midpass with Time Injection
        t_proj = self.time_emb_proj(t_emb).view(-1, d2.shape[1], 1, 1)
        m = d2 + t_proj
        m = F.silu(self.mid1(m))
        m = F.silu(self.mid2(m))
        
        # Uppass
        u1 = F.silu(self.up1(m))
        # Skip connection
        u1 = torch.cat([u1, d1], dim=1) 
        out = self.up2(u1)
        
        return out


class CFGFlowMatcher:
    def __init__(self, model, cfg_drop_rate=0.1):
        self.model = model
        self.cfg_drop_rate = cfg_drop_rate

    def compute_loss(self, x_1, condition):
        B = x_1.shape[0]
        device = x_1.device

        # OT Path
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
            
            x_t = x_t + v_cfg * dt
            
        return x_t