import numpy as np
from lp_config.lp_common_config import config
def heatmapMSE(y_pred, y_true):
    loss = (y_pred - y_true)**2
    return loss.mean(dim=3).mean(dim=2).mean(dim=1)

def computeLoss(y_preds, y_trues):
    loss = 0
    for n, heatmap_pred in enumerate(y_preds):
        heatmap_true = y_trues[n]
        if(heatmap_pred != None):
            heatmaps_rest = heatmap_pred[:, :config["num_joints"]]
            loss+=heatmapMSE(heatmaps_rest, heatmap_true)

    return loss/len(y_preds)