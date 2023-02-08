import numpy as np
import torch
from lp_config.lp_common_config import config

def heatmapMSE(y_pred, y_true, mask):
    assert y_pred.size() == y_true.size()

    loss = ((y_pred - y_true)**2) * mask[:, None, :, :].expand_as(y_pred)
    return loss.mean(dim=3).mean(dim=2).mean(dim=1)

def tagLoss(tags, gtJoints, OMEGA = 1):
    outtags = tags.contiguous().view(tags.size()[0], -1)

    batch_size, max_person, num_joints = int(gtJoints.shape[0]), int(gtJoints.shape[1]), int(gtJoints.shape[2])

    validJoints = gtJoints[:, :, :, 1].float()
    jointsValue = gtJoints[:, :, :, 0].reshape(batch_size, -1).long()

    joint_perPerson = validJoints.sum(2, keepdim=True)

    tags = torch.gather(outtags, index=jointsValue, dim=1)
    tags = tags.reshape(batch_size, max_person, num_joints)

    realtags = tags*validJoints

    person_cnt = (joint_perPerson > 0).float().squeeze(2).sum(dim=1, keepdim=True) 

    person_cnt[person_cnt == 0] = 1
    joint_perPerson[joint_perPerson==0]=1

    # Hn line in the paper

    mean_h = realtags.sum(2, keepdim=True)
    mean_h = mean_h/joint_perPerson
    mean_h = mean_h.nan_to_num(0)

    # First term L

    diff = (mean_h-realtags)**2
    diff *= validJoints

    aggregate = diff.sum(2, keepdim=True)/joint_perPerson
    aggregate = aggregate.nan_to_num(0)

    aggregate = aggregate.squeeze(2).sum(1, keepdim=True)/person_cnt

    tagmse = aggregate.squeeze(1)

    # Second term L

    # Efficient way to implement the difference of every couple of n elements array

    repMatrix = mean_h.expand(batch_size, max_person, max_person).float()

    repMatrixTrasnspose = repMatrix.transpose(1,2)

    diffElementwise = torch.square(repMatrix-repMatrixTrasnspose) # symmetric matrices, this can be improved by exploiting this property (still O(n^2))

    diffElementwise *= 1/(2*OMEGA)

    diffElementwise = 1/torch.exp(diffElementwise)

    tagexp = diffElementwise.mean(2).mean(1)

    return tagexp+tagmse

def computeLoss(y_preds, gtHeatmaps, gtMask, gtJoints):
    heatmapLoss = 0
    tLoss = 0
    n = 0
    for n, heatmap_pred in enumerate(y_preds):
        heatmap_true = gtHeatmaps[n]
        joints_true = gtJoints[n]
        mask_true = gtMask[n]
        if(heatmap_pred != None):
            heatmaps_rest = heatmap_pred[:, :config["num_joints"]]
            heatmapLoss += heatmapMSE(heatmaps_rest, heatmap_true, mask_true)

            tag_rest = heatmap_pred[:, config["num_joints"]:]
            if n < 1:
                tLoss += tagLoss(tag_rest, joints_true) * config["tag_loss_weight"]
        n+=1

    return heatmapLoss, tLoss