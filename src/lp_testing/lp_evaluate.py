from lp_coco_utils.lp_getDataset import getDatasetProcessed
from lp_config.lp_common_config import config
from lp_inference.lp_oks import getOks
import torch
import tqdm

def evaluateModel(model):
    ds = getDatasetProcessed("test")
    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=config["batch_size"]
    )
    score = []
    for batch in tqdm.tqdm(data_loader):
        tmpscore = getOks(model, batch)
        score.append(sum(tmpscore)/len(tmpscore))
    
    return sum(score)/len(score)
