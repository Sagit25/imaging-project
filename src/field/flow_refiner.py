import torch


class CFGFlowRefiner:
    def __init__(self, model, cfg_drop_rate=0.1, grid_channels=2):
        self.model = model
        self.cfg_drop_rate = cfg_drop_rate
        self.grid_channels = grid_channels

    def compute_loss(self, target, condition):
        B = target.shape[0]
        device = target.device

        t = torch.rand(B, device=device)
        t_expand = t.view(B, 1, 1, 1)

        x_0 = torch.randn_like(target)
        x_t = t_expand * target + (1.0 - t_expand) * x_0
        target_v = target - x_0

        condition_masked = self._drop_condition(condition)
        unet_input = torch.cat([x_t, condition_masked], dim=1)
        pred_v = self.model(unet_input, t * 1000.0)

        return ((pred_v - target_v) ** 2).mean()

    def _drop_condition(self, condition):
        if self.cfg_drop_rate <= 0 or not self.model.training:
            return condition

        B = condition.shape[0]
        device = condition.device
        keep_mask = (torch.rand(B, 1, 1, 1, device=device) > self.cfg_drop_rate).float()
        condition_masked = condition.clone()
        non_grid_channels = condition.shape[1] - self.grid_channels
        condition_masked[:, :non_grid_channels] = condition_masked[:, :non_grid_channels] * keep_mask
        return condition_masked

    @torch.no_grad()
    def sample(self, condition, num_steps=50, cfg_scale=3.0):
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")

        B, _, H, W = condition.shape
        device = condition.device

        x_t = torch.randn(B, 3, H, W, device=device)
        null_condition = torch.zeros_like(condition)
        null_condition[:, -self.grid_channels:] = condition[:, -self.grid_channels:]

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=device)

            unet_input_cond = torch.cat([x_t, condition], dim=1)
            v_cond = self.model(unet_input_cond, t * 1000.0)

            unet_input_uncond = torch.cat([x_t, null_condition], dim=1)
            v_uncond = self.model(unet_input_uncond, t * 1000.0)

            v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
            x_t = x_t + v_cfg * dt

        return x_t.clamp(-1, 1)


def build_refiner_condition(model_input, dist_tensor, warp_outputs):
    return torch.cat(
        [
            warp_outputs["warped_rgb"],
            dist_tensor,
            model_input[:, -2:],
        ],
        dim=1,
    )
