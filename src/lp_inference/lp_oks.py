import torch
import math

from lp_inference.lp_inference import getkpsfromhms
from lp_config.lp_common_config import config

# This was developed in a hurry, is inefficient and ugly. It needs a strong refactoring

def getCloserDistance(point, gtlist):
    bestDistance = torch.inf
    bestIdx = None
    for i, elem in enumerate(gtlist):
        dist = math.dist(elem, point)
        if(dist < bestDistance):
            bestDistance = dist
            bestIdx = i
    return bestDistance, bestIdx

@torch.no_grad()
def getOks(model, batch, k=7):
    images = batch[0].to(config["device"])
    gthm = batch[1]

    out = model(images)

    out = [elem[:,:config["num_joints"],:,:].cpu() for elem in out]

    gtkeypoints = getkpsfromhms(gthm, 255)
    outkeypoints = getkpsfromhms(out, 255)

    oks = []
    for b, batch in enumerate(outkeypoints):
        visible = 0
        score = 0
        for j, joint in enumerate(batch):
            gtcpy = gtkeypoints[b][j].copy()
            for detection in joint:
                distance, idx = getCloserDistance(detection, gtcpy)
                
                if(idx is None):
                    continue

                del gtcpy[idx]
                visible+=1
                score += math.exp(-(distance**2)/(2*k**2))
        if visible > 0:
            oks.append(score/visible)
    return oks
