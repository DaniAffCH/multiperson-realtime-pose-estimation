from lp_coco_utils.lp_getDataset import getDatasetProcessed
from lp_model.lp_litepose import LitePose
from lp_config.lp_common_config import config
from lp_training.lp_earlyStop import EarlyStopping
from lp_training.lp_trainOne import trainOneEpoch

import torch

def train(batch_size):
    ds = getDatasetProcessed("train")

    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size
    )

    model = LitePose().to(config["device"])

    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])
    es = EarlyStopping(model, config["earlyStop_eps"], config["earlyStop_threshold"], config["backup_name"])

    for i in range(config["epochs"]):
        ret, loss = trainOneEpoch(model, data_loader, optimizer, es, i)
        print(f"epoch #{i} loss {loss}")
        if(ret):
            break
    