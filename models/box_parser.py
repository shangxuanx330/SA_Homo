from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class get_k_pts_BoxParser(nn.Module):
    def __init__(self, stride: int) -> None:
        super(get_k_pts_BoxParser, self).__init__()
        self.stride = stride

    @torch.no_grad()
    def forward(self, score_map: torch.Tensor, offset_map: torch.Tensor) -> torch.Tensor:
        bs, num_pts, ho, wo = score_map.shape
        flat_size = ho * wo

        # Flatten score_map to [bs, num_pts, flat_size]
        score_flat = score_map.flatten(start_dim=2)

        # Get indices of max scores [bs, num_pts]
        indices = score_flat.argmax(dim=2)

        # Reshape offset_map to [bs, num_pts, 2, ho, wo] then flatten to [bs, num_pts, 2, flat_size]
        offset_flat = offset_map.view(bs, num_pts, 2, ho, wo).flatten(start_dim=3)

        # Prepare indices for gathering: [bs, num_pts, 2, 1]
        indices_exp = indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, 2, 1)

        # Gather offsets [bs, num_pts, 2, 1]
        offset_gathered = offset_flat.gather(dim=3, index=indices_exp)

        # Squeeze to [bs, num_pts, 2]
        offset = offset_gathered.squeeze(3)

        # Compute coarse coordinates
        y = torch.div(indices, wo, rounding_mode="floor")
        x = indices - y * wo
        tl_downscaled_coarse = torch.stack((y, x), dim=2).float()

        # Refined downscaled
        tl_downscaled = tl_downscaled_coarse + offset

        # Upscale
        tl = tl_downscaled * self.stride

        return tl


class get_k_corase_pts_BoxParser(nn.Module):
    def __init__(self, stride: int) -> None:
        super(get_k_corase_pts_BoxParser, self).__init__()
        self.stride = stride

    @torch.no_grad()
    def forward(self, score_map: torch.Tensor, offset_map: torch.Tensor) -> torch.Tensor:
        bs, num_pts, ho, wo = score_map.shape
        flat_size = ho * wo

        # Flatten score_map to [bs, num_pts, flat_size]
        score_flat = score_map.flatten(start_dim=2)

        # Get indices of max scores [bs, num_pts]
        indices = score_flat.argmax(dim=2)

        # Reshape offset_map to [bs, num_pts, 2, ho, wo] then flatten to [bs, num_pts, 2, flat_size]
        offset_flat = offset_map.view(bs, num_pts, 2, ho, wo).flatten(start_dim=3)

        # Prepare indices for gathering: [bs, num_pts, 2, 1]
        indices_exp = indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, 2, 1)

        # Gather offsets [bs, num_pts, 2, 1]
        offset_gathered = offset_flat.gather(dim=3, index=indices_exp)

        # Squeeze to [bs, num_pts, 2]
        offset = offset_gathered.squeeze(3)

        # Compute coarse coordinates
        y = torch.div(indices, wo, rounding_mode="floor")
        x = indices - y * wo

        tl_downscaled_coarse = torch.stack((y, x), dim=2).float()
        


        return tl_downscaled_coarse





class Spatial_BoxParser(nn.Module):
    def __init__(self, stride: int) -> None:
        super(Spatial_BoxParser, self).__init__()
        self.stride = stride

    def forward(self, score_map: torch.Tensor, offset_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            score_map: Tensor of shape (B, K, H, W)
            offset_map: Tensor of shape (B, 2K, H, W)
        Returns:
            x1: Tensor of shape (B, K, 2)
        """
        B, K, H, W = score_map.shape

        # 1. 生成坐标网格
        device = score_map.device
        dtype = score_map.dtype

        y_coords = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, K, H, W)
        x_coords = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, K, H, W)
        
        # 2. 应用偏移
        # 将 offset_map 重新调整形状
        offset_map = offset_map.view(B, K, 2, H, W)
        offset_y = offset_map[:, :, 0, :, :]  # 形状：(B, K, H, W)
        offset_x = offset_map[:, :, 1, :, :]  # 形状：(B, K, H, W)

        y_coords = y_coords + offset_y
        x_coords = x_coords + offset_x

        # 3. 计算加权和
        # 校正后的坐标
        corrected_coords = torch.stack((y_coords, x_coords), dim=-1)  # 形状：(B, K, H, W, 2)
        
        # 将 score_map 扩展以匹配坐标的维度
        scores = score_map.unsqueeze(-1)  # 形状：(B, K, H, W, 1)

        # 对校正后的坐标进行加权
        weighted_coords = corrected_coords * scores  # 形状：(B, K, H, W, 2)
        
        # 在空间维度上求和
        sum_weighted_coords = weighted_coords.view(B, K, -1, 2).sum(dim=2)  # 形状：(B, K, 2)

        # 计算权重之和（即 score_map 的总和）
        sum_scores = scores.view(B, K, -1).sum(dim=2).unsqueeze(-1)  # 形状：(B, K, 1)
        # 4. 计算加权平均值（避免除以零）
        sum_scores = sum_scores + 1e-10  # 防止除以零
        x1 = sum_weighted_coords / sum_scores  # 形状：(B, K, 2)

        # 5. 考虑步长（stride）
        x1 = x1 * self.stride

        return x1  # 返回形状：(B, K, 2)




class DifferentiableBoxParser(nn.Module):
    def __init__(self, stride: int, temperature: float = 0.1, sharpness: float = 10, smoothness: float = 0.1) -> None:
        super(DifferentiableBoxParser, self).__init__()
        self.stride = stride
        self.temperature = temperature
        self.sharpness = sharpness
        self.smoothness = smoothness

    def soft_ceil(self, x):
        return x + (1 - torch.sigmoid(self.sharpness * (x - torch.floor(x))))

    def smooth_clamp(self, x, min_val, max_val):
        x = torch.where(x < min_val, min_val + self.smoothness * torch.tanh((x - min_val) / self.smoothness), x)
        x = torch.where(x > max_val, max_val - self.smoothness * torch.tanh((max_val - x) / self.smoothness), x)
        return x

    def forward(self, score_map: torch.Tensor, offset_map: torch.Tensor) -> torch.Tensor:
        bs, k, ho, wo = score_map.shape
        
        # Compute differentiable argmax for each keypoint
        coords = differentiable_argmax_2d(score_map, temperature=self.temperature)
        
        # Apply soft ceil to get "almost integer" coordinates
        coords_ceil = self.soft_ceil(coords)
        
        # Smooth clamp coordinates to ensure they're within bounds
        coords_clamped = torch.stack([
            self.smooth_clamp(coords_ceil[..., 0], 0, ho - 1),
            self.smooth_clamp(coords_ceil[..., 1], 0, wo - 1)
        ], dim=-1)
        # Get offset of the center using the computed coordinates
        batch_indices = torch.arange(bs, device=coords.device).reshape(bs, 1, 1).expand(bs, k, 1)
        kpt_indices = torch.arange(k, device=coords.device).reshape(1, k, 1).expand(bs, k, 1)
        y_indices = coords_clamped[..., 0].unsqueeze(dim=-1).long()
        x_indices = coords_clamped[..., 1].unsqueeze(dim=-1).long()
        
        offset_y = offset_map[batch_indices, 2*kpt_indices, y_indices, x_indices].squeeze(-1)
        offset_x = offset_map[batch_indices, 2*kpt_indices+1, y_indices, x_indices].squeeze(-1)
        offset = torch.stack([offset_y, offset_x], dim=-1)

        # Compute refined downscaled coordinates
        pts_downscaled = coords + offset
        
        # Compute top-left coordinate in input scale
        pts = pts_downscaled * self.stride

        return pts
    

def differentiable_argmax_2d(logits, temperature=1.0):
    """
    找出[bs,i,:,:]中的值最大的元素的坐标作为[bs,i,2]的返回值， 先纵坐标再横坐标。 确保整个过程是可导的
    输入:
    logits: shape [bs, k, ho, wo]
        bs: batch size
        k:  预测角点的个数 
        ho: 高度 (height)
        wo: 宽度 (width)
    temperature: 浮点数，控制 softmax 的平滑程度

    输出:
    坐标张量: shape [bs, k, 2]
        每行包含 [y, x] 坐标，其中 y 是纵坐标(行), x 是横坐标（列）
    """
    bs, k, ho, wo = logits.shape
    
    # 将 logits 展平: [bs, k, ho, wo] -> [bs*k, ho*wo]
    logits_flat = logits.reshape(bs*k, -1)
    
    # 计算 softmax: [bs*k, ho*wo]
    probs = F.softmax(logits_flat / temperature, dim=-1)
    
    # 创建 y 坐标网格 (纵坐标，对应行): [1, ho, wo] -> [bs*k, ho*wo]
    y_positions = torch.arange(ho, dtype=logits.dtype, device=logits.device).reshape(1, ho, 1).expand(bs*k, ho, wo)
    y_positions_flat = y_positions.reshape(bs*k, -1)
    
    # 创建 x 坐标网格 (横坐标，对应列): [1, ho, wo] -> [bs*k, ho*wo]
    x_positions = torch.arange(wo, dtype=logits.dtype, device=logits.device).reshape(1, 1, wo).expand(bs*k, ho, wo)
    x_positions_flat = x_positions.reshape(bs*k, -1)
    
    # 计算加权和得到 y 坐标 (纵坐标): [bs*k]
    y = torch.sum(probs * y_positions_flat, dim=-1)
    
    # 计算加权和得到 x 坐标 (横坐标): [bs*k]
    x = torch.sum(probs * x_positions_flat, dim=-1)
    
    # 将 y 和 x 坐标堆叠并重塑: [bs, k, 2]
    coords = torch.stack([y, x], dim=-1).reshape(bs, k, 2)
    
    return coords