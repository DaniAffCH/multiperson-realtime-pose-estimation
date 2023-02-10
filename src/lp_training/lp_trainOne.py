from lp_config.lp_common_config import config
import tqdm

def trainOneEpoch(model, dataloader, optimizer, epoch, loss, testing=False):
    loss_tot_l = []
    loss_hm_l = []
    loss_t_l = []
    model.train()
    for images, heatmaps, masks, joints in tqdm.tqdm(dataloader):
        optimizer.zero_grad()
        images = images.to(config["device"])
        heatmaps = [h.to(config["device"]) for h in heatmaps]
        joints = [j.to(config["device"]) for j in joints]
        masks = [m.to(config["device"]) for m in masks]
        y_pred = model(images)
        heatmaps_losses, tag_losses = loss(y_pred, heatmaps, masks, joints)
        heatmapLoss = sum(heatmaps_losses)
        tagLoss = sum(tag_losses)

        heatmapLoss = heatmapLoss.mean(0)
        tagLoss = tagLoss.mean(0)
        totLoss = heatmapLoss+tagLoss

        totLoss.backward()
        optimizer.step()

        loss_tot_l.append(float(totLoss))
        loss_hm_l.append(float(heatmapLoss))
        loss_t_l.append(float(tagLoss))
        
        if(testing):
            break

    return sum(loss_tot_l)/len(loss_tot_l), sum(loss_hm_l)/len(loss_hm_l), sum(loss_t_l)/len(loss_t_l)