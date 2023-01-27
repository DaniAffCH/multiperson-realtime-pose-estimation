from lp_coco_utils.lp_getDataset import getDatasetProcessed
from lp_model.lp_litepose import LitePose
from lp_training.lp_trainOne import trainOneEpoch
from lp_training.lp_earlyStop import EarlyStopping
from PIL import Image
from torchvision.transforms import functional as func
from lp_config.lp_common_config import config
import torch.nn.functional as F
import random
import torch

def test():
    ok = "\033[92m[PASSED]\033[0m"
    no = "\033[91m[FAILED]\033[0m"
    ds = None
    model = None
    data_loader = None
    passed = 0
    tot = 0

    try:
        ds = getDatasetProcessed("train")
        print("[TEST] Dataset loading and preprocessing... "+ok)
        passed+=1
    except Exception as e: 
        print("[TEST] Dataset loading and preprocessing... "+no)
        print(e)
    tot+=1 

    try:
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=8
        )
        print("[TEST] Data Loader... "+ok)
        passed+=1
    except Exception as e: 
        print("[TEST] Data Loader... "+no)
        print(e)
    tot+=1 

    try:
        model = LitePose().to(config["device"])
        print("[TEST] Model loading... "+ok)
        passed+=1
    except Exception as e: 
        print("[TEST] Model loading... "+no)
        print(e)
    tot+=1 

    try:
        for row in data_loader:
            images = row[0]
            img_size = 256 + random.randint(0, 4) * 64
            images = F.interpolate(images, size = (img_size, img_size))
            images = images.to(config["device"])
            model(images)
            print("[TEST] Model feedforward scale invariant... "+ok)
            passed+=1
            break
    except Exception as e: 
        print("[TEST] Model feedforward scale invariant... "+no)
        print(e)
    tot+=1 

    try:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
        er = EarlyStopping(model, 1e-5, 10)

        trainOneEpoch(model, data_loader, optimizer, er, 1, True)
        print("[TEST] Train step.... "+ok)
        passed+=1

    except Exception as e: 
        print("[TEST] Train step... "+no)
        print(e)
    tot+=1 


    print(f"[TEST] {passed}/{tot} tests passed")