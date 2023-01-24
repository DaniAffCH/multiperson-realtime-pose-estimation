from dataclasses import dataclass
from typing import List

@dataclass
class MobileBlock_settings():
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int

class Stage_settings():
    def __init__(self, blocks: List[MobileBlock_settings]):
        self.blocks = blocks
        self.size = len(blocks)

class Backbone_settings():
    def __init__(self, stages: List[Stage_settings]):
        self.stages = stages
        self.size = len(stages)

class Deconv_settings():
    def __init__(self, channels: List[int], kernel: List[int]):
        if(len(channels) != len(kernel)):
            Exception("[Deconv_settings] channels and kernel lists must have the same dimension")
        self.channels = channels
        self.size = len(self.channels)
        self.kernel = kernel

config = dict(

    litepose = dict(
        largeKernels = 7,
        
        backbone = Backbone_settings([
            # STAGE 1 
            Stage_settings([
                MobileBlock_settings(3,32,3,4),
                MobileBlock_settings(5,6,7,8),
                MobileBlock_settings(9,10,11,12)
            ]),
            # STAGE 2
            Stage_settings([
                MobileBlock_settings(3,32,3,4),
                MobileBlock_settings(5,6,7,8),
                MobileBlock_settings(9,10,11,12)
            ]),
            # STAGE 3
            Stage_settings([
                MobileBlock_settings(3,32,3,4),
                MobileBlock_settings(5,6,7,8),
                MobileBlock_settings(9,10,11,12)
            ]),
            # STAGE 4
            Stage_settings([
                MobileBlock_settings(3,32,3,4),
                MobileBlock_settings(5,6,7,8),
                MobileBlock_settings(9,10,11,12)
            ]),
            # STAGE 5
            Stage_settings([
                MobileBlock_settings(3,32,3,4),
                MobileBlock_settings(5,6,7,8),
                MobileBlock_settings(9,10,11,12)
            ])
        ]),

        # deconvLayers must be <= #stages - 2
        deconvLayers = Deconv_settings([64, 32, 16], [3,4,3]),

        joints = 17
    )
)