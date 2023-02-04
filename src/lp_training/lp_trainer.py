from lp_coco_utils.lp_getDataset import getDatasetProcessed
from lp_model.lp_litepose import LitePose
from lp_config.lp_common_config import config
from lp_training.lp_earlyStop import EarlyStopping
from lp_training.lp_trainOne import trainOneEpoch
from lp_training.lp_loss import computeLoss
import tqdm
import time

import torch

def train(batch_size):
    ds = getDatasetProcessed("train")

    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size
    )

    val_ds = getDatasetProcessed("validation")

    val_data_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size//2
    )

    model = LitePose().to(config["device"])

    optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
    #optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])
    es = EarlyStopping(model, config["earlyStop_eps"], config["earlyStop_threshold"], config["backup_name"])

    start_time = time.time()

    for i in range(config["epochs"]):
        train_loss = trainOneEpoch(model, data_loader, optimizer, i)
        val_loss_avg = 0
        val_loss_l = []
        with torch.no_grad():
            model.eval()
            for images, heatmaps, masks, joints in tqdm.tqdm(val_data_loader):
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

                val_loss_l.append(totLoss)
            val_loss_avg = sum(val_loss_l)/len(val_loss_l)

        print(f"epoch #{i+1} training loss {train_loss}   validation loss {val_loss_avg}")

        if(es(val_loss_avg)):
            break
    
    print(f"end training, exec time: {str(time.time() - start_time)} seconds, final validation loss: {val_loss_avg}")