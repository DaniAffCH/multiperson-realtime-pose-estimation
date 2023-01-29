from lp_training.lp_loss import computeLoss
import numpy
from lp_config.lp_common_config import config
import tqdm

#TODO logger 

def trainOneEpoch(model, dataloader, optimizer, earlyStopper, epoch, testing=False):
    lossavg = 0
    for i, (images, heatmaps, masks, joints) in tqdm.tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        images = images.to(config["device"])
        heatmaps = [h.to(config["device"]) for h in heatmaps]
        y_pred = model(images)
        loss = computeLoss(y_pred, heatmaps)
        loss = loss.mean(axis=0)
        lossavg += float(loss)
        loss.backward()
        optimizer.step()
        
        if(testing):
            break

    lossavg /= len(dataloader)
        
    if(earlyStopper(lossavg)):
        return True, lossavg
    
    return False, lossavg