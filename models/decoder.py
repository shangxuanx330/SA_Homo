import sys
sys.path.append('/data/xieshangxuan/master/img_matching/match_with_transformer')
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common  import ResidualConv2dBNAct



class Decoder_adapative(nn.Module):
    """
    解码器模块，具有以下特点：
    1. 至少包含 4 次降采样。如果输入尺寸不足以支持 4 次降采样，则剩余的层不进行 MaxPooling。
    2. 如果输入尺寸支持超过 4 次降采样，则选择使输出尺寸满足 2 <= h, w < 4 的层数。
    3. 最后一层固定为 nn.AdaptiveAvgPool2d((n, n))，用于将输出调整为指定大小。
    """
    def __init__(self, input_dim=256, input_size=(16, 20), target_size=(2, 2)):
        """
        Args:
            input_dim (int): 输入的通道数（feature maps）。
            input_size (tuple): 输入图像的尺寸 (height, width)。
            target_size (tuple): 最终输出的目标尺寸 (n, n)。
        """
        super(Decoder_adapative, self).__init__()
        self.layers = nn.ModuleList()  # 存储所有中间层
        self.input_dim = input_dim

        # 输入尺寸
        height, width = input_size

        # 计算满足 2 <= h, w < 4 的下采样次数
        max_downsamples = 0
        curr_height, curr_width = height, width
        while curr_height >= 4 or curr_width >= 4:  # 继续下采样直到 h 或 w < 4
            max_downsamples += 1
            curr_height = curr_height // 2
            curr_width = curr_width // 2
            # 检查是否满足 2 <= h, w < 4
            if 2 <= curr_height < 4 and 2 <= curr_width < 4:
                break

        # 至少需要 4 层
        total_layers = max(max_downsamples, 4)

        # 函数用于计算合适的 num_groups
        def get_num_groups(input_dim, desired_divisor=8):
            for i in range(desired_divisor, 0, -1):
                if input_dim % i == 0:
                    return i
            return 1

        num_groups = get_num_groups(input_dim)

        # 创建前 max_downsamples 层（带 MaxPooling）
        for _ in range(min(max_downsamples, total_layers)):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, stride=1),
                    nn.GroupNorm(num_groups=num_groups, num_channels=input_dim),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )

        # 创建剩余的层（不带 MaxPooling）
        for _ in range(total_layers - min(max_downsamples, total_layers)):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, stride=1),
                    nn.GroupNorm(num_groups=num_groups, num_channels=input_dim),
                    nn.ReLU(inplace=True)
                )
            )

        self.final_conv = nn.Conv2d(input_dim, 2, 1)
        # 添加最终的自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d(target_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        x = self.adaptive_pool(x)
        return x






if __name__ == "__main__":
    output = main()