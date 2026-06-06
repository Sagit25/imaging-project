import torch
import torch.nn.functional as F

class CFGFlowMatcher:
    def __init__(self, model, cfg_drop_rate=0.1, clean_loss_weight=0.5, lowfreq_loss_weight=0.2):
        self.model = model
        self.cfg_drop_rate = cfg_drop_rate
        self.clean_loss_weight = clean_loss_weight
        self.lowfreq_loss_weight = lowfreq_loss_weight

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

        velocity_loss = F.mse_loss(pred_v, target_v)
        pred_x1 = x_t + (1.0 - t_expand) * pred_v
        clean_loss = F.l1_loss(pred_x1, x_1)
        lowfreq_loss = F.l1_loss(
            F.avg_pool2d(pred_x1, kernel_size=8, stride=8),
            F.avg_pool2d(x_1, kernel_size=8, stride=8),
        )

        return (
            velocity_loss
            + self.clean_loss_weight * clean_loss
            + self.lowfreq_loss_weight * lowfreq_loss
        )

    def predict_velocity(self, x_t, condition, t, cfg_scale=1.0):
        unet_input_cond = torch.cat([x_t, condition], dim=1)
        v_cond = self.model(unet_input_cond, t * 1000.0)

        if cfg_scale == 1.0:
            return v_cond

        null_condition = torch.zeros_like(condition)
        null_condition[:, -2:] = condition[:, -2:]
        unet_input_uncond = torch.cat([x_t, null_condition], dim=1)
        v_uncond = self.model(unet_input_uncond, t * 1000.0)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    @torch.no_grad()
    def sample(self, condition, num_steps=50, cfg_scale=2.0, sample_seed=None):
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")

        B, _, H, W = condition.shape
        device = condition.device

        rng_state = None
        cuda_rng_state = None
        if sample_seed is not None:
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(sample_seed)
        x_t = torch.randn(B, 3, H, W, device=device)
        if rng_state is not None:
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=device)

            v_cfg = self.predict_velocity(x_t, condition, t, cfg_scale=cfg_scale)
            x_t = x_t + v_cfg * dt
            x_t = x_t.clamp(-1.5, 1.5)

        return x_t.clamp(-1, 1)
