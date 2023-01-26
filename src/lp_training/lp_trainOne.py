from lp_training.lp_loss import computeLoss

#TODO early stopping and logger 

def trainOneEpoch(model, dataloader, optimizer, epoch, testing=False):
    for i, (images, heatmaps, masks, joints) in enumerate(dataloader):
        optimizer.zero_grad()
        y_pred = model(images)
        loss = computeLoss(y_pred, heatmaps)
        loss.backward()
        optimizer.step()
        
        if(testing):
            break
        