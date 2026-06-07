import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialWarpingModule(nn.Module):
    def __init__(self, model, smoothness_weight=0.05):
        super().__init__()
        self.model = model
        self.smoothness_weight = smoothness_weight

    def forward_warp(self, dist_tensor, flow_field):
        """
        dist_tensor: [B, 3, H, W]
        flow_field: [B, 2, H, W]
        """
        B, _, H, W = dist_tensor.shape
        device = dist_tensor.device
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        identity_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1) # [B, 2, H, W]
        
        sampling_grid = identity_grid + flow_field # [B, 2, H, W]
        sampling_grid = sampling_grid.permute(0, 2, 3, 1)
        
        unwarped_pred = F.grid_sample(
            dist_tensor, 
            sampling_grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )
        return unwarped_pred

    def compute_loss_components(self, model_input, dist_tensor, gt_tensor):
        """
        model_input: [B, 5, H, W]
        dist_tensor: [B, 3, H, W]
        gt_tensor: [B, 3, H, W]
        """
        B = model_input.shape[0]
        device = model_input.device
        
        flow_field = self.model(model_input, torch.zeros(B, device=device))
        unwarped_pred = self.forward_warp(dist_tensor, flow_field)
        
        recon_loss = F.mse_loss(unwarped_pred, gt_tensor)
        
        dy = flow_field[:, :, 1:, :] - flow_field[:, :, :-1, :]
        dx = flow_field[:, :, :, 1:] - flow_field[:, :, :, :-1]
        smooth_loss = torch.mean(dy**2) + torch.mean(dx**2) # TV
        
        weighted_smooth_loss = self.smoothness_weight * smooth_loss
        total_loss = recon_loss + weighted_smooth_loss
        
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "smooth_loss": smooth_loss,
            "weighted_smooth_loss": weighted_smooth_loss,
            "unwarped_pred": unwarped_pred,
            "flow_field": flow_field,
        }

    def compute_loss(self, model_input, dist_tensor, gt_tensor):
        losses = self.compute_loss_components(model_input, dist_tensor, gt_tensor)
        total_loss = losses["total_loss"]
        unwarped_pred = losses["unwarped_pred"]
        flow_field = losses["flow_field"]
        return total_loss, unwarped_pred, flow_field
