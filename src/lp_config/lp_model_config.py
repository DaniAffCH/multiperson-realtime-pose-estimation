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

arch1 = dict(        
        backbone = Backbone_settings([
            # STAGE 1 
            Stage_settings([
                MobileBlock_settings(16,16,7,2),
                MobileBlock_settings(16,32,7,1),
                MobileBlock_settings(32,32,7,1),
                MobileBlock_settings(32,24,7,1)
            ]),
            # STAGE 2
            Stage_settings([
                MobileBlock_settings(24,64,7,2),
                MobileBlock_settings(64,64,7,1),
                MobileBlock_settings(64,64,7,1),
                MobileBlock_settings(64,64,7,1),
                MobileBlock_settings(64,64,7,1),
            ]),
            # STAGE 3
            Stage_settings([
                MobileBlock_settings(64,64,7,2),
                MobileBlock_settings(64,64,7,1),
                MobileBlock_settings(64,72,7,1),
                MobileBlock_settings(72,72,7,1),
                MobileBlock_settings(72,72,7,1),
            ]),
            # STAGE 4
            Stage_settings([
                MobileBlock_settings(72,72,7,1),
                MobileBlock_settings(72,80,7,1),
                MobileBlock_settings(80,100,7,1),
                MobileBlock_settings(100,120,7,1),
                MobileBlock_settings(120,140,7,1),
                MobileBlock_settings(140,160,7,1)
            ])
        ])
    )

arch2 = Backbone_settings([
            # STAGE 1 
            Stage_settings([
                MobileBlock_settings(16,24,7,2),
                MobileBlock_settings(24,24,7,1),
                MobileBlock_settings(24,24,7,1),
                MobileBlock_settings(24,24,7,1),
                MobileBlock_settings(24,24,7,1),
                MobileBlock_settings(24,24,7,1)
            ]),
            # STAGE 2
            Stage_settings([
                MobileBlock_settings(24,64,7,2),
                MobileBlock_settings(64,64,7,1),
                MobileBlock_settings(64,64,7,1),
                MobileBlock_settings(64,64,7,1),
                MobileBlock_settings(64,64,7,1),
                MobileBlock_settings(64,64,7,1),
                MobileBlock_settings(64,64,7,1),
                MobileBlock_settings(64,64,7,1),
            ]),
            # STAGE 3
            Stage_settings([
                MobileBlock_settings(64,64,7,2),
                MobileBlock_settings(64,64,7,1),
                MobileBlock_settings(64,72,7,1),
                MobileBlock_settings(72,72,7,1),
                MobileBlock_settings(72,72,7,1),
                MobileBlock_settings(72,72,7,1),
                MobileBlock_settings(72,72,7,1),
                MobileBlock_settings(72,72,7,1),
            ]),
            # STAGE 4
            Stage_settings([
                MobileBlock_settings(72,72,7,1),
                MobileBlock_settings(72,80,7,1),
                MobileBlock_settings(80,100,7,1),
                MobileBlock_settings(100,120,7,1),
                MobileBlock_settings(120,140,7,1),
                MobileBlock_settings(140,140,7,1),
                MobileBlock_settings(140,140,7,1),
                MobileBlock_settings(140,140,7,1),
                MobileBlock_settings(140,140,7,1),
                MobileBlock_settings(140,160,7,1)
            ])
        ])


config = dict(

    litepose = dict(        
        backbone = arch2,

        # #deconvLayers must be <= #stages - 2
        deconvLayers = Deconv_settings([48, 24, 24], [4,4,4])
    )
)