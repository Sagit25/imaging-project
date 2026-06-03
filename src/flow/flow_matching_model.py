import torch
import torch.nn.functional as F

class CFGFlowMatcher:
    def __init__(self, model, cfg_drop_rate=0.1):
        self.model = model
        self.cfg_drop_rate = cfg_drop_rate

    def compute_loss(self, x_1, condition):
        B = x_1.shape[0]
        device = x_1.device

        t = torch.rand(B, device=device)
        t_expand = t.view(B, 1, 1, 1)
        
        x_0 = torch.randn_like(x_1)
        x_t = t_expand * x_1 + (1 - t_expand) * x_0
        target_v = x_1 - x_0

        # CFG Dropout
        if self.cfg_drop_rate > 0 and self.model.training:
            drop_mask = (torch.rand(B, 1, 1, 1, device=device) > self.cfg_drop_rate).float()
            condition_masked = condition.clone()
            condition_masked[:, :4] = condition_masked[:, :4] * drop_mask
        else:
            condition_masked = condition

        unet_input = torch.cat([x_t, condition_masked], dim=1)
        
        pred_v = self.model(unet_input, t * 1000.0)
        
        return F.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def sample(self, condition, num_steps=50, cfg_scale=4.0):
        B, C, H, W = condition.shape
        device = condition.device
        
        x_t = torch.randn(B, 3, H, W, device=device)
        
        null_condition = torch.zeros_like(condition)
        null_condition[:, -2:] = condition[:, -2:]
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=device)
            
            # Conditional 
            unet_input_cond = torch.cat([x_t, condition], dim=1)
            v_cond = self.model(unet_input_cond, t * 1000.0)
            
            # Unconditional 
            unet_input_uncond = torch.cat([x_t, null_condition], dim=1)
            v_uncond = self.model(unet_input_uncond, t * 1000.0)
            
            v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
        
            x_t = x_t + v_cfg * dt
            
        return x_t