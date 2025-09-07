import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveLightFieldTuner(nn.Module):
    """自适应光场调谐器(ALFT)
    通过深度信息动态调整相位信息，优化成像效果。"""

    def __init__(self, init_alpha: float = 0.5):
        super().__init__()
        # 可学习参数，初始值0.5，能够通过训练自动调整
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

        # 将深度映射到权重系数，使用1x1卷积
        self.depth_proj = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, phase: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """根据相位图和深度图进行调谐得到新相位"""
        # 将深度归一化到[0,1]范围
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth_gain = torch.sigmoid(self.depth_proj(depth_norm))  # (B,1,H,W)
        tuned_phase = phase + self.alpha * (depth_gain - 0.5)  # 调整相位
        # 将相位包裹到[-pi,pi]范围
        tuned_phase = torch.remainder(tuned_phase + torch.pi, 2 * torch.pi) - torch.pi
        return tuned_phase