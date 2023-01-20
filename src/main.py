from __future__ import absolute_import

import argparse
import torch
from lp_coco_utils.lp_getDataset import getDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--train',
                        help='train the network on Coco dataset',
                        action='store_true')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ds = getDataset("validation")
    

if __name__ == '__main__':
    main()