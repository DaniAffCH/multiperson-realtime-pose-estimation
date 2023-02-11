import numpy as np
import torch
from lp_config.lp_common_config import config
from torch import nn

class Lp_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def heatmapMSE(self, y_pred, y_true, mask):
        loss = ((y_pred - y_true)**2) * mask[:, None, :, :].expand_as(y_pred)
        return loss.mean(dim=3).mean(dim=2).mean(dim=1)

    def tagLoss(self, pred_tag_map, joints):
        if pred_tag_map.dim() == 3:
            pred_tag_map = pred_tag_map.squeeze(2)

        batch_size, max_person, num_joints = int(joints.shape[0]), int(joints.shape[1]), int(joints.shape[2])

        joints_vis = joints[:, :, :, 1].float()
        person_joints_cnt = joints_vis.sum(2, keepdim=True)
        joints_loc = joints[:, :, :, 0].reshape(batch_size, -1).long()

        tags = torch.gather(pred_tag_map, index=joints_loc, dim=1)
        tags = tags.reshape(batch_size, max_person, num_joints) * joints_vis

        person_cnt = (person_joints_cnt > 0).float().squeeze(2).sum(dim=1, keepdim=True)
        person_cnt[person_cnt == 0] = 1
        person_vis = (person_joints_cnt > 0).expand(batch_size, max_person, max_person).float()
        person_vis = person_vis * person_vis.permute(0, 2, 1)
        person_joints_cnt[person_joints_cnt == 0] = 1

        # Minimize variance on each person
        tags_mean = tags.sum(2, keepdim=True) / person_joints_cnt
        assert torch.isnan(tags_mean).sum() == 0

        pull = torch.sum(joints_vis * (tags - tags_mean) ** 2, dim=2, keepdim=True) / person_joints_cnt
        pull[person_joints_cnt == 0] = 0

        pull = pull.squeeze(2).sum(1, keepdim=True) / person_cnt
        pull = torch.mean(pull)

        # Maximize mean distance between peoples
        tags_mean = (tags_mean).expand(batch_size, max_person, max_person) 
        diff = (tags_mean - tags_mean.permute(0, 2, 1)) * person_vis

        diff = torch.exp(- diff ** 2) * person_vis
        diff = 0.5 * (torch.sum(diff, dim=(1, 2)) - person_cnt.squeeze(1)) / torch.clamp((person_cnt - 1) * person_cnt, min=1).squeeze(1)
        diff[person_cnt.squeeze(1) < 2] = 0

        push = torch.mean(diff)

        return push+pull
    
    def forward(self,y_preds, gtHeatmaps, gtMask, gtJoints):
        heatmaps_losses = []
        tag_losses = []
        for idx in range(len(y_preds)):
            heatmaps_pred = y_preds[idx][:, :config["num_joints"]]

            heatmaps_loss = self.heatmapMSE(heatmaps_pred, gtHeatmaps[idx], gtMask[idx])
            heatmaps_loss = heatmaps_loss
            heatmaps_losses.append(heatmaps_loss)

            tags_pred = y_preds[idx][:, config["num_joints"]:]
            batch_size = tags_pred.size()[0]
            tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)

            tag_loss = self.tagLoss(tags_pred, gtJoints[idx])

            tag_loss = tag_loss * config["tag_loss_weight"]

            tag_losses.append(tag_loss)


        return heatmaps_losses, tag_losses
