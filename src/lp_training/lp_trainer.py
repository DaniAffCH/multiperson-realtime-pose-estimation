from lp_coco_utils.lp_getDataset import getDatasetProcessed
from lp_model.lp_litepose import LitePose
from lp_config.lp_common_config import config
from lp_training.lp_earlyStop import EarlyStopping
from lp_training.lp_trainOne import trainOneEpoch
from lp_training.lp_loss import Lp_Loss

import tqdm
import time

import torch

def train(batch_size):
    ds = getDatasetProcessed("trainval")

    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size
    )

    val_ds = getDatasetProcessed("validation")

    val_data_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size
    )

    model = LitePose().to(config["device"])

    optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
    #optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])
    es = EarlyStopping(model, config["earlyStop_eps"], config["earlyStop_threshold"], config["backup_name"])

    start_time = time.time()

    loss_fac = Lp_Loss()

    for i in range(config["epochs"]):
        tot_loss, heatmap_loss, tag_loss = trainOneEpoch(model, data_loader, optimizer, i,loss_fac)
        loss_tot_l = []
        loss_hm_l = []
        loss_t_l = []
        with torch.no_grad():
            model.eval()
            for images, heatmaps, masks, joints in tqdm.tqdm(val_data_loader):
                images = images.to(config["device"])
                heatmaps = [h.to(config["device"]) for h in heatmaps]
                joints = [j.to(config["device"]) for j in joints]
                masks = [m.to(config["device"]) for m in masks]
                y_pred = model(images)

                heatmaps_losses, tag_losses = loss_fac(y_pred, heatmaps, masks, joints)
                heatmapLoss_val = sum(heatmaps_losses)
                tagLoss_val = sum(tag_losses)
                heatmapLoss_val = heatmapLoss_val.mean(0)
                tagLoss_val = tagLoss_val.mean(0)
                totLoss_val = heatmapLoss_val+tagLoss_val

                loss_tot_l.append(float(totLoss_val))
                loss_hm_l.append(float(heatmapLoss_val))
                loss_t_l.append(float(tagLoss_val))
        
        avgLossTot_val = sum(loss_tot_l)/len(loss_tot_l)
        avgLossHm_val = sum(loss_hm_l)/len(loss_hm_l)
        avgLossT_val = sum(loss_t_l)/len(loss_t_l)

        print(f"epoch #{i+1} \n\nTRAINING LOSS:\ntotal Loss = {tot_loss}\nheatmap Loss = {heatmap_loss}\ntag Loss = {tag_loss}\n\nVALIDATION LOSS:\ntotal Loss = {avgLossTot_val}\nheatmap Loss = {avgLossHm_val}\ntag Loss = {avgLossT_val}\n\n\n")

        if(es(avgLossTot_val)):
            break
    
    print(f"end training, exec time: {str(time.time() - start_time)}")