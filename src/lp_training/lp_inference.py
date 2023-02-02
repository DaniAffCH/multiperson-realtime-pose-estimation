import torch
import cv2
import lp_config.lp_common_config as ccfg
import lp_utils.lp_image_processing as ip
import numpy as np

# assume a minibatch of m and k joints
#output structure:
# [ 
#   heatmap_group_1(size 64): [m, k, 64, 64]
#   heatmap_group_2(size 128): [m, k, 128, 128]
# ]

TRESHOLD = 100

@torch.no_grad()
def inference(model, images):
    outputs = model(images)
    kps = []
    
    # it can be improved: use 64 hm for small images
    hml = outputs[1].cpu()
    for n,b in enumerate(hml):
        assert images[n].shape[1] == images[n].shape[2], "non-square image"
        b = ip.scaleImage(b.cpu(), images[n].shape[1]).numpy()
        bkp = []
        for joint in b:
            joint = ip.normalizeImage(joint)
            bkp.append(ip.getMostPromisingPoint(joint).numpy())
        kps.append(bkp)
    
    return outputs, kps