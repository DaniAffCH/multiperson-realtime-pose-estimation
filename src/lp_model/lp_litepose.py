from lp_config.lp_model_config import config
import lp_config.lp_common_config as cmc
import torch
from torch import nn 
import lp_model.lp_common_layers as lcl
class LitePose(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        conf = config["litepose"]

        in_channels = 16

        self.c1 = nn.Sequential(
            lcl.ConvBlockBase(3, 32, 3, 2),
            lcl.ConvBlockBase(32, in_channels, 3, 1)
        )

        # Backbone
        backboneConf = conf["backbone"]
        self.stages = []
        self.channels = [in_channels]

        for s in range(backboneConf.size):
            self.stages.append(lcl.ConvStage(s))
            self.channels.append(backboneConf.stages[s].blocks[-1].out_channels)
        
        self.backbone = nn.ModuleList(self.stages)

        # Deconv Head 
        self.loopLayers = []
        self.refineLayers = []
        self.refineChannels = self.channels[-1]
        deconvConf = conf["deconvLayers"]
        for l in range(deconvConf.size):
            rawChannels = self.channels[-l-2]

            pad, out_pad = self.get_deconv_paddings(deconvConf.kernel[l])

            self.refineLayers.append(
                nn.ConvTranspose2d(
                    self.refineChannels, 
                    deconvConf.channels[l],
                    deconvConf.kernel[l],
                    2,
                    pad,
                    out_pad,
                    bias=False)
            )
            self.loopLayers.append(
                nn.ConvTranspose2d(
                    rawChannels, 
                    deconvConf.channels[l],
                    deconvConf.kernel[l],
                    2,
                    pad,
                    out_pad,
                    bias=False)
            )
            self.refineChannels = deconvConf.channels[l]

        self.loopLayers = nn.ModuleList(self.loopLayers)
        self.refineLayers = nn.ModuleList(self.refineLayers)

        # Output 
        self.loopFinal = []
        self.refineFinal = []
        self.finalChannel = []
        for l in range(1, deconvConf.size):
            # 2*num_joints: num_joints channels represent heatmaps for each joint, the others num_joints channels are the tags 
            self.refineFinal.append(nn.Sequential(
                lcl.ConvBlockBase(deconvConf.channels[l], deconvConf.channels[l], 5),
                lcl.ConvBlockBase(deconvConf.channels[l], 2*cmc.config["num_joints"], 5)
            ))

            self.loopFinal.append(nn.Sequential(
                lcl.ConvBlockBase(self.channels[-l-3], self.channels[-l-3], 5),
                lcl.ConvBlockBase(self.channels[-l-3], 2*cmc.config["num_joints"], 5)
            ))

        self.refineFinal = nn.ModuleList(self.refineFinal)
        self.loopFinal = nn.ModuleList(self.loopFinal)



    def get_deconv_paddings(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return padding, output_padding


    
    def forward(self, x):
        # Large Kernels Convs
        x = self.c1(x)

        # Backbone
        x_checkpoints = [x]
        for l in range(len(self.backbone)):
            x = self.stages[l](x)
            x_checkpoints.append(x)

        # Deconv Head 
        outputs = []
        for l in range(len(self.refineLayers)):
            x = self.refineLayers[l](x)
            x_loop = self.loopLayers[l](x_checkpoints[-l-2])
            x = x + x_loop

            # Final
            if l > 0:
                finalForward = self.refineFinal[l-1](x)
                finalLoop = self.loopFinal[l-1](x_checkpoints[-l-3])
                outputs.append(finalForward+finalLoop)

        return outputs
