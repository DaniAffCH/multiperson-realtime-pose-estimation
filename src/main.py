from __future__ import absolute_import

import argparse
import torch
import torchvision
from lp_utils.lp_realtime import keypointOnCam
from lp_config.lp_common_config import config
from lp_testing.lp_test import test
from lp_training.lp_trainer import train
from lp_model.lp_litepose import LitePose
from lp_coco_utils.lp_getDataset import getDatasetProcessed
from lp_inference.lp_inference import inference, assocEmbedding
import cv2
from lp_utils.lp_image_processing import drawHeatmap, drawKeypoints, drawSkeleton

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--train',
                        help='train the network on Coco dataset',
                        action='store_true')

    parser.add_argument('--resnet-live', 
                        help="use keypoint Resnet50 pretrained model as live detector",
                        action="store_true")

    parser.add_argument('--test', 
                        help="just for testing purposes",
                        action="store_true")

    parser.add_argument('--inference', 
                        help="perform inference and shows the results",
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
    train(config["batch_size"])

def handleTest():
    test()

def handleInference():
    model = LitePose().to(config["device"])
    model.load_state_dict(torch.load("lp_trained_models/bigarch"))

    ds = getDatasetProcessed("train")

    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=8
    )

    row = next(iter(data_loader))
    images = row[0].to(config["device"])

    gthm = row[1]
    output, keypoints = inference(model, images)

    embedding = assocEmbedding(keypoints)

    jointsHeatmap = output[1][2][:config["num_joints"]]

    img, finalHm, superimposed = drawHeatmap(images[2], jointsHeatmap)
    img, gtfinalHm, gtsuperimposed = drawHeatmap(images[2], gthm[1][2])
    cv2.imshow("Image", img)
    cv2.imshow("Final heatmap", finalHm)
    cv2.imshow("Superimposed", superimposed)

    cv2.imshow("Ground Truth heatmap", gtfinalHm)
    cv2.imshow("Ground Truth Superimposed", gtsuperimposed)
    cv2.waitKey()

    cv2.destroyAllWindows()

    img = drawKeypoints(images[6], keypoints[6])
    cv2.imshow("Image Keypoints", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    img = drawSkeleton(images[6], embedding[6])
    cv2.imshow("Image Skeleton", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def main():
    args = parse_args()

    if(args.train):
        handleTrain()
    elif(args.resnet_live):
        handleResnetLive()
    elif(args.test):
        handleTest()
    elif(args.inference):
        handleInference()
    
if __name__ == '__main__':
    main()