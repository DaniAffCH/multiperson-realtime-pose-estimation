import cv2 
import torch
import numpy as np
from lp_config.lp_common_config import config

@torch.no_grad()
def keypointOnCam(model, savePath, confidenceThreshold = 0):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if (cap.isOpened() == False):
        print('Error while trying to open webcam')
        return
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(f"{savePath}", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 20, 
                      (frame_width, frame_height))

    while(cap.isOpened()):
    # capture each frame of the video
        ret, frame = cap.read()
        if ret == True:

            image = frame
            image = cv2.resize(image, (224, 224))
            orig_frame = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0).to(config["device"])
            outputs = model(image)
            for output in outputs:
                if(output["keypoints"].shape[0] == 0):
                    continue
                keypointsScore = output["keypoints_scores"]
                keypointsScore = keypointsScore.cpu().detach().numpy()
                output = output["keypoints"][0]
                output = output.cpu().detach().numpy()
                output = np.delete(output, 2, 1)
                output = output.reshape(-1, 2)
                keypoints = output
                for p in range(keypoints.shape[0]):
                    meanks = keypointsScore[0,p]
                    if(meanks < confidenceThreshold):
                        continue
                    cv2.circle(orig_frame, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                                1, (0, 0, 255), -1, cv2.LINE_AA)
                orig_frame = cv2.resize(orig_frame, (frame_width, frame_height))
            cv2.imshow('Keypoint Frame', orig_frame)
            out.write(orig_frame)

            if cv2.waitKey(27) & 0xFF == ord('q'):
                break
    
        else: 
            break
        
    cap.release()
    cv2.destroyAllWindows()

