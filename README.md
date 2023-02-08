# LitePose keypoint estimation

LitePose is an efficient and scale invariant multi-person pose estimation model. Its light architecture allows to perform real-time inference with low computational power devices such as smartphones. This is a simpler reimplementation of the original [LitePose][1].

## How does it work?

LitePose follows a bottom-up pose estimation approach. The single-branch architecture ensures high efficiency, while the Fusion Deconv Head implements the scale invariance by using high resolution features.
MobileNet structure with large kernels deconvolution is used as backbone.

## Installation

## Usage 

## Acknowledgements
This work is based on [LitePose][2] paper and [HigherHRNet][3]. Moreover it uses the [Associative Embedding][4] to cluster the keypoints.

[1]:https://github.com/mit-han-lab/litepose
[2]:https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Lite_Pose_Efficient_Architecture_Design_for_2D_Human_Pose_Estimation_CVPR_2022_paper.pdf
[3]:https://arxiv.org/pdf/1908.10357.pdf
[4]:https://papers.nips.cc/paper/2017/file/8edd72158ccd2a879f79cb2538568fdc-Paper.pdf
