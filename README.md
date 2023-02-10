# LitePose pose estimation

LitePose is an efficient and scale invariant multi-person pose estimation model. Its light architecture allows to perform real-time inference with low computational power devices such as smartphones. This is a simpler reimplementation of the original [LitePose][1].

![Network Architecture!](/assets/keypoints.png)


## How does it work?

LitePose follows a bottom-up pose estimation approach. The single-branch architecture ensures high efficiency, while the Fusion Deconv Head implements the scale invariance by using high resolution features.
MobileNet structure with large kernels deconvolution is used as backbone. The whole network is scalable with respect to the number of joints and the maximum number of people that the image may contain.

![Network Architecture!](/assets/structure.png)

## Installation
Clone the repository:
```
git clone https://github.com/DaniAffCH/litepose-pose-estimation
```

Install python requirements:
```
pip install -r requirements.txt
```

Download both annotations and images of CrowdPose Dataset from [the official repository](https://github.com/Jeff-sjtu/CrowdPose#dataset). 
Then recreate a directory structure as:
```
crowdpose
├─── images
│    ├── 112934.jpg
│   ...
└── json
    ├── crowdpose_test.json
    ├── crowdpose_train.json
    ├── crowdpose_trainval.json
    └── crowdpose_val.json
```

Finally edit `src/lp_config/lp_common_config.py` and modify the variable `dataset_root` with your installation path (default is `~/dataset/crowdpose`). 
In order to check if the installation was successful you can run `python main.py --test`, if it passes all the test cases then the set up is working correctly.

## Usage 

Every settings can be modified in `src/lp_config` where there are `lp_model_config.py` that contains the settings about network architecture and `lp_common_config.py` that contains the general configurations about training and inference. 

For training run:
```
python main.py --train
```

For inference run:
```
python main.py --inference
```

`demo.ipynb` is a notebook that shows an example of code usage and provides further details about the project.

## Acknowledgements
This work is based on [LitePose][2] paper and [HigherHRNet][3]. Moreover it uses the [Associative Embedding][4] to cluster the keypoints.

[1]:https://github.com/mit-han-lab/litepose
[2]:https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Lite_Pose_Efficient_Architecture_Design_for_2D_Human_Pose_Estimation_CVPR_2022_paper.pdf
[3]:https://arxiv.org/pdf/1908.10357.pdf
[4]:https://papers.nips.cc/paper/2017/file/8edd72158ccd2a879f79cb2538568fdc-Paper.pdf
