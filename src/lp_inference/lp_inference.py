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

@torch.no_grad()
def suppression(det):
    pool = torch.nn.MaxPool2d(7,1,3)
    maxm = pool(det)
    maxm = torch.eq(maxm, det).float()
    det = det * maxm
    return det
    
@torch.no_grad()
def inference(model, images):
    outputs = model(images)
    kps = []
    
    hml = [elem[:ccfg.config["num_joints"]].cpu() for elem in outputs]
    imgw = images[0].shape[1]

    kps = []
    
    for j in range(hml[0].shape[0]):
        hm1 = ip.scaleImage(hml[0][j], imgw).numpy()
        hm2 = ip.scaleImage(hml[1][j], imgw).numpy()
        hmavg = torch.tensor((hm1+hm2)/2)
        hmavg = hmavg/hmavg.max()
        hmavg = suppression(hmavg)

        bkp = []
        for joint in hmavg:
            joint = joint.view(-1)
            values,idxs = joint.topk(ccfg.config["max_people"])
            subpeoples = []
            for n,v in enumerate(values):
                if v < ccfg.config["confidence_threshold"]:
                    break
                subpeoples.append([int(idxs[n]%imgw), int(idxs[n]/imgw)])
            bkp.append(subpeoples)
        kps.append(bkp)
            
    return outputs, kps