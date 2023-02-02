from lp_training.lp_loss import computeLoss
from lp_config.lp_common_config import config
import tqdm

def trainOneEpoch(model, dataloader, optimizer, epoch, testing=False):
    lossavg = 0
    loss_l = []
    model.train()
    for i, (images, heatmaps, masks, joints) in tqdm.tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        images = images.to(config["device"])
        heatmaps = [h.to(config["device"]) for h in heatmaps]
        y_pred = model(images)
        loss = computeLoss(y_pred, heatmaps)
        loss = loss.mean(axis=0)
        loss_l.append(float(loss))
        loss.backward()
        optimizer.step()
        
        if(testing):
            break

    lossavg = sum(loss_l)/len(loss_l)
    return lossavg