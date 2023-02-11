import torch
import numpy as np
import cv2

def normalizeImage(img):
    return img/img.max()*255 if img.max() > 0 else img

def getMostPromisingPoint(heatmap, isTensor = False):
    hm = torch.tensor(heatmap) if not isTensor else heatmap
    _,w = hm.shape
    hm = hm.view(-1)
    _, mostPromisingPoint = hm.topk(1)
    return torch.cat(((mostPromisingPoint % w).unsqueeze(1), (mostPromisingPoint // w).unsqueeze(1)), dim=1)[0]

def mergeMultipleHeatmaps(heatmaps):
    hm = np.mean(heatmaps, axis=0)
    return hm


"""
    @param img: Torch tensor of shape [ch,w,h]
    @param b: scalar (representing a square image)
"""
def scaleImage(img, output_size):
    sf = output_size/img.shape[1]
    img = img.unsqueeze(0)
    scaled = torch.nn.functional.interpolate(img,scale_factor=sf, mode='bilinear')
    return scaled[0]

def drawKeypoints(img, keypoints):
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = normalizeImage(img)
    img = img.astype(np.uint8).copy() 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for kp in keypoints:
        for n,person in enumerate(kp):
            img = cv2.circle(img, (person["x"],person["y"]), radius=3, color=(0, 0, 255), thickness=-1)
    return img

def drawSkeleton(img, edgelist):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    img = img.transpose(1, 2, 0)
    img = normalizeImage(img)
    img = img.astype(np.uint8).copy() 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for edge in edgelist:
        img = cv2.line(img, (edge["xf"], edge["yf"]), (edge["xt"], edge["yt"]), color=(0, 255, 0), thickness=3)
    return img

def drawHeatmap(img, heatmaps):
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = normalizeImage(img)
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    heatmaps = scaleImage(heatmaps, img.shape[1]).cpu().numpy()

    finalHm = mergeMultipleHeatmaps(heatmaps)
    finalHm = normalizeImage(finalHm)
    finalHm = cv2.applyColorMap(np.uint8(finalHm), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(finalHm, 0.5, img, 0.5, 0)

    return img, finalHm, superimposed

#heatmaps = list(map(lambda x: normalizeImage(x) , heatmaps))
#heatmaps[0] = cv2.applyColorMap(np.uint8(heatmaps[0]), cv2.COLORMAP_JET)