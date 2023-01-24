from __future__ import absolute_import

import argparse
import torch
import torchvision
from lp_coco_utils.lp_getDataset import getDataset
from lp_utils.lp_realtime import keypointOnCam
from lp_config.lp_common_config import config
from lp_testing.lp_test import test

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--train',
                        help='train the network on Coco dataset',
                        action='store_true')

    parser.add_argument('--resnet-live', 
                        help="use keypoint Resnet50 pretrained model as live detector",
                        action="store_true")

    parser.add_argument('--test', 
                        help="just for testing purposes",
                        action="store_true")

    args = parser.parse_args()

    return args

def handleResnetLive():
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, 
                                        weights= torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.eval()
    model.to(config["device"])
    keypointOnCam(model, "~/Videos/r1.mp4v")

def handleTrain():
    ds = getDataset("validation")

def handleTest():
    test()

def main():
    args = parse_args()

    if(args.train):
        handleTrain()
    elif(args.resnet_live):
        handleResnetLive()
    elif(args.test):
        handleTest()
    
    

if __name__ == '__main__':
    main()