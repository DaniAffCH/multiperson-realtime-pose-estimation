from lp_config.lp_model_config import config
import torch
from torch import nn 
import lp_common_layers as lcl
class LitePose(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        conf = config["litepose"]

        # Large kernel convs 
        self.largek = nn.Sequential(
            lcl.convBlockBase(3, 32, conf["largeKernels"], 2),
            lcl.convBlockBase(32, 16, conf["largeKernels"], 1)
        )
    
    def forward(self, x):
        x = self.largek(x)

        return x
