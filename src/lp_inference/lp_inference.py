import torch
import cv2
import lp_config.lp_common_config as ccfg
import lp_utils.lp_image_processing as ip
import numpy as np

# assume a minibatch of m and k joints
# output structure:
# [ 
#   heatmap_group_1(size 64): [m, k, 64, 64]
#   heatmap_group_2(size 128): [m, k, 128, 128]
# ]

@torch.no_grad()
def suppression(det):
    pool = torch.nn.MaxPool2d(15,1,7)
    maxm = pool(det)
    maxm = torch.eq(maxm, det).float()
    det = det * maxm
    return det
    
@torch.no_grad()
def getkpsfromhms(hms,scale):

    kps = []
    for j in range(hms[0].shape[0]):
        
        # assuming that 2 training scales are used. This should be generalized 
        hm1 = ip.scaleImage(hms[0][j], scale).numpy()
        hm2 = ip.scaleImage(hms[1][j], scale).numpy()

        hmavg = torch.tensor((hm1+hm2)/2)
        hmavg = hmavg/hmavg.max()
        hmavg = suppression(hmavg)

        bkp = []
        # for each joint
        for joint in hmavg:
            joint = joint.view(-1)
            values,idxs = joint.topk(ccfg.config["max_people"])
            subpeoples = []
            for n,v in enumerate(values):
                if v < ccfg.config["confidence_threshold"]:
                    break
                x = int(idxs[n]%scale)
                y = int(idxs[n]/scale)
                subpeoples.append([x,y])
            bkp.append(subpeoples)
        kps.append(bkp)
    return kps

@torch.no_grad()
def inference(model, images):
    outputs = model(images)
    
    outputs = [elem.cpu() for elem in outputs]

    kps = []
    # for each batch element
    for j in range(outputs[0].shape[0]):
        imgw = images[j].shape[1]
        imgh = images[j].shape[2]
        
        # assuming that 2 training scales are used. This should be generalized 
        hm1 = ip.scaleImage(outputs[0][j][:ccfg.config["num_joints"]], imgw).numpy()
        hm2 = ip.scaleImage(outputs[1][j][:ccfg.config["num_joints"]], imgw).numpy()

        tg1 = ip.scaleImage(outputs[0][j][ccfg.config["num_joints"]:], imgw).numpy()
        tg2 = ip.scaleImage(outputs[1][j][ccfg.config["num_joints"]:], imgw).numpy()

        hmavg = torch.tensor((hm1+hm2)/2)
        hmavg = hmavg/hmavg.max()
        hmavg = suppression(hmavg)

        tgavg = torch.tensor((tg1+tg2)/2)
        tgavg = tgavg/tgavg.max()

        bkp = []
        # for each joint
        for jn, joint in enumerate(hmavg):
            joint = joint.view(-1)
            values,idxs = joint.topk(ccfg.config["max_people"])
            subpeoples = []
            for n,v in enumerate(values):
                if v < ccfg.config["confidence_threshold"]:
                    break
                x = int(idxs[n]%imgw)
                y = int(idxs[n]/imgw)
                subpeoples.append({
                    "x":x,
                    "y":y,
                    "tag":float(tgavg[jn][y][x])
                })
            bkp.append(subpeoples)
        kps.append(bkp)
            
    return outputs, kps


@torch.no_grad()
def getCloserElement(value, listb):
    bestCandidate = 256
    bestCandidatecpy = None
    bestCandidateIdx = None
    for idx, element in enumerate(listb):
        diff = abs(value - element["tag"])
        if diff < bestCandidate:
            bestCandidate = diff
            bestCandidatecpy = element.copy()
            bestCandidateIdx = idx

    return bestCandidate, bestCandidatecpy, bestCandidateIdx


# Output: list of [(x,y),(x',y'))] where each pair indicate an edge between those two points
@torch.no_grad()
def assocEmbedding(kps):
    totout = []
    #batch element
    for single in kps:
        singleImgOut = []
        for a,b in ccfg.crowd_pose_part_orders:
            idxa = ccfg.crowd_pose_part_idx[a]
            idxb = ccfg.crowd_pose_part_idx[b]
            if(len(single) < idxb):
                continue
            bcopy = single[idxb].copy()
            for elem in single[idxa]:
                distance, bestMatch, rmidx = getCloserElement(elem["tag"], bcopy)
                if distance < 1-ccfg.config["confidence_embedding"]:
                    del bcopy[rmidx]
                    singleImgOut.append({"xf": elem["x"] ,"yf":elem["y"], "xt":bestMatch["x"],"yt":bestMatch["y"]})

        totout.append(singleImgOut)
    
    return totout