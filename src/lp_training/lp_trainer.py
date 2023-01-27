from lp_coco_utils.lp_getDataset import getDatasetProcessed
from lp_model.lp_litepose import LitePose
from lp_config.lp_common_config import config
from lp_training.lp_earlyStop import EarlyStopping
from lp_training.lp_trainOne import trainOneEpoch
import torch

def train(batch_size):
    ds = getDatasetProcessed()

    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size
    )

    model = LitePose().to(config["device"])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    es = EarlyStopping(model, 1e-1, 10, "test1")

    for i in range(100):
        ret = trainOneEpoch(model, data_loader, optimizer, es, i)
        if(ret):
            break
    