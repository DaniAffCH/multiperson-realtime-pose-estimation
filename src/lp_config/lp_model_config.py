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
            ])
        ])
    )
)