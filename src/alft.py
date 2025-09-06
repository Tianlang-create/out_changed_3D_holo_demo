import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveLightFieldTuner(nn.Module):
    """简化版自适应光场调谐模块(ALFT)。
    通过场景深度特征动态调整全息相位，提升衍射效率。"""

    def __init__(self, init_alpha: float = 0.5):
        super().__init__()
        # 单一可学习增益，初值 0.5，可根据训练自动调整
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

        # 深度通道映射到增益系数，使用轻量 1×1 卷积
        self.depth_proj = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, phase: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """输入相位与深度图，输出调谐后的相位。"""
        # depth 归一化到 [0,1]
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth_gain = torch.sigmoid(self.depth_proj(depth_norm))  # (B,1,H,W)
        tuned_phase = phase + self.alpha * (depth_gain - 0.5)  # 调制相位
        # wrap to [-pi,pi]
        tuned_phase = torch.remainder(tuned_phase + torch.pi, 2 * torch.pi) - torch.pi
        return tuned_phase