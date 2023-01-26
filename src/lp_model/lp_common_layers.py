import torch.nn as nn
from lp_config.lp_model_config import config

# it must be used always inside a Sequential()
class ConvBlockBase(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation=True):
        if(activation):
            super(ConvBlockBase, self).__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        else:
            super(ConvBlockBase, self).__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(out_channels),
            )

class ConvMobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvMobileBlock, self).__init__()

        self.useResidual = in_channels == out_channels and stride == 1

        midChannels = in_channels+out_channels // 2
        
        self.s = nn.Sequential(
            ConvBlockBase(in_channels, midChannels, 1, activation=False),
            ConvBlockBase(midChannels, midChannels, kernel_size, stride, activation=False),
            ConvBlockBase(midChannels, out_channels, 1)
        )

    def forward(self, x):
        return self.s(x) + x if self.useResidual else self.s(x) 

class ConvStage(nn.Module):
    def __init__(self, n_stage):
        super(ConvStage, self).__init__()
        stageConfig = config["litepose"]["backbone"].stages[n_stage]

        listBlock = []

        for i in range(stageConfig.size):
            blockConfig = stageConfig.blocks[i]
            listBlock.append(ConvMobileBlock(blockConfig.in_channels, blockConfig.out_channels, blockConfig.kernel_size, blockConfig.stride))
        
        self.stage = nn.Sequential(*listBlock)

    def forward(self, x):
        return self.stage(x)