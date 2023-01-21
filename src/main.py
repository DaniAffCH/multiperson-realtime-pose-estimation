from __future__ import absolute_import

import argparse
import torch
import torchvision
from lp_coco_utils.lp_getDataset import getDataset
from lp_utils.lp_realtime import keypointOnCam

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--train',
                        help='train the network on Coco dataset',
                        action='store_true')

    parser.add_argument('--resnet-live', 
                        help="use keypoint Resnet50 pretrained model as live detector",
                        action="store_true")

    args = parser.parse_args()

    return args

def handleResnet(device):
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)
    keypointOnCam(model, device, "~/Videos/r1.mp4v")

def handleTrain(device):
    ds = getDataset("validation")

def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if(args.train):
        handleTrain(device)
    elif(args.resnet):
        handleResnet(device)
    
    

if __name__ == '__main__':
    main()