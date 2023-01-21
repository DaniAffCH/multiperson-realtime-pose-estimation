import torch.nn as nn

# it must be used always inside a Sequential()
class convBlockBase(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=True):
        if(activation):
            super(convBlockBase, self).__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        else:
            super(convBlockBase, self).__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(out_channels),
            )

