import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialWarpingModule(nn.Module):
    def __init__(self, model, smoothness_weight=0.05):
        super().__init__()
        self.model = model
        self.smoothness_weight = smoothness_weight

    def make_sampling_grid(self, flow_field):
        B, _, H, W = flow_field.shape
        device = flow_field.device

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        identity_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        sampling_grid = identity_grid + flow_field
        return sampling_grid.permute(0, 2, 3, 1)

    def forward_warp(self, input_tensor, flow_field, mode='bilinear', padding_mode='border'):
        """
        input_tensor: [B, C, H, W]
        flow_field: [B, 2, H, W]
        """
        sampling_grid = self.make_sampling_grid(flow_field)

        return F.grid_sample(
            input_tensor,
            sampling_grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=True
        )

    def flow_smoothness_loss(self, flow_field):
        dy = flow_field[:, :, 1:, :] - flow_field[:, :, :-1, :]
        dx = flow_field[:, :, :, 1:] - flow_field[:, :, :, :-1]
        return torch.mean(dy**2) + torch.mean(dx**2)

    def build_warp_outputs(self, model_input, dist_tensor):
        B = model_input.shape[0]
        device = model_input.device

        flow_field = self.model(model_input, torch.zeros(B, device=device))
        warped_rgb = self.forward_warp(dist_tensor, flow_field, padding_mode='border')

        return {
            "flow": flow_field,
            "warped_rgb": warped_rgb,
        }

    def compute_loss(self, model_input, dist_tensor, gt_tensor):
        """
        model_input: [B, 5, H, W]
        dist_tensor: [B, 3, H, W]
        gt_tensor: [B, 3, H, W]
        """
        outputs = self.build_warp_outputs(model_input, dist_tensor)

        recon_loss = F.mse_loss(outputs["warped_rgb"], gt_tensor)
        smooth_loss = self.flow_smoothness_loss(outputs["flow"])
        total_loss = recon_loss + self.smoothness_weight * smooth_loss

        return total_loss, outputs["warped_rgb"], outputs["flow"], outputs
