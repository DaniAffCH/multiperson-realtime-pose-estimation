from lp_training.lp_loss import computeLoss
from lp_config.lp_common_config import config
import tqdm

def trainOneEpoch(model, dataloader, optimizer, epoch, testing=False):
    lossavg = 0
    loss_l = []
    model.train()
    for images, heatmaps, masks, joints in tqdm.tqdm(dataloader):
        optimizer.zero_grad()
        images = images.to(config["device"])
        heatmaps = [h.to(config["device"]) for h in heatmaps]
        joints = [j.to(config["device"]) for j in joints]
        masks = [m.to(config["device"]) for m in masks]
        y_pred = model(images)
        heatmapLoss, tagLoss = computeLoss(y_pred, heatmaps, masks, joints)
        totLoss = 0

        for scaleIdx in range(len(heatmapLoss)):
            hl = heatmapLoss[scaleIdx].mean(0)
            totLoss+=hl
            tl = tagLoss[scaleIdx].mean(0)
            totLoss+=tl

        loss_l.append(float(totLoss))
        totLoss.backward()
        optimizer.step()
        
        if(testing):
            break

    lossavg = sum(loss_l)/len(loss_l)
    return lossavg