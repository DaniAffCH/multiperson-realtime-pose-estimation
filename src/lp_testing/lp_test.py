from lp_coco_utils.lp_getDataset import getDataset
from lp_model.lp_litepose import LitePose
def test():
    ok = "\033[92m[PASSED]\033[0m"
    no = "\033[91m[FAILED]\033[0m"
    ds = None
    model = None
    try:
        ds = getDataset("validation")
        print("[TEST] Dataset loading and preprocessing... "+ok)
    except Exception as e: 
        print("[TEST] Dataset loading and preprocessing... "+no)
        print(e)
        return

    try:
        model = LitePose()
        print("[TEST] Model loading... "+ok)
    except Exception as e: 
        print("[TEST] Model loading... "+no)
        print(e)
        return

    #TODO: model feedforward
    
