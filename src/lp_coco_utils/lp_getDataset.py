import os
import os.path
import cv2
import json_tricks as json
import numpy as np
from torch.utils.data import Dataset

from lp_coco_utils import lp_transform as T
from lp_coco_utils.lp_generators import HeatmapGenerator, JointsGenerator
from lp_config.lp_common_config import config


class CrowdPoseDataset(Dataset):
    def __init__(self, root, dataset, data_format, transform=None,
                 target_transform=None):
        from crowdposetools.coco import COCO
        self.name = 'CROWDPOSE'
        self.root = root
        self.dataset = dataset
        self.data_format = data_format
        self.coco = COCO(self._get_anno_file_name())
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

    def _get_anno_file_name(self):
        return os.path.join(
            self.root,
            'json',
            'crowdpose_{}.json'.format(
                self.dataset
            )
        )

    def _get_image_path(self, file_name):
        images_dir = os.path.join(self.root, 'images')
        return os.path.join(images_dir, file_name)

    def __getitem__(self, index):

        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        file_name = coco.loadImgs(img_id)[0]['file_name']

        img = cv2.imread(
            self._get_image_path(file_name),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

class CrowdPoseKeypoints(CrowdPoseDataset):
    def __init__(self,
                 root,
                 dataset_name,
                 remove_images_without_annotations,
                 heatmap_generator,
                 joints_generator,
                 transforms=None):
        super().__init__(root,
                         dataset_name,
                         ".jpg")

        self.num_scales = len(heatmap_generator)

        self.num_joints = config["num_joints"]
        self.num_joints_without_center =self.num_joints


        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.transforms = transforms
        self.heatmap_generator = heatmap_generator
        self.joints_generator = joints_generator

    def __getitem__(self, idx):
        img, anno = super().__getitem__(idx)

        mask = self.get_mask(anno, idx)

        anno = [
            obj for obj in anno
            if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
        ]

        joints = self.get_joints(anno)

        mask_list = [mask.copy() for _ in range(self.num_scales)]
        joints_list = [joints.copy() for _ in range(self.num_scales)]
        target_list = list()

        if self.transforms:
            img, mask_list, joints_list = self.transforms(
                img, mask_list, joints_list
            )

        for scale_id in range(self.num_scales):
            target_t = self.heatmap_generator[scale_id](joints_list[scale_id])
            joints_t = self.joints_generator[scale_id](joints_list[scale_id])

            target_list.append(target_t.astype(np.float32))
            mask_list[scale_id] = mask_list[scale_id].astype(np.float32)
            joints_list[scale_id] = joints_t.astype(np.int32)

        return img, target_list, mask_list, joints_list

    def get_joints(self, anno):
        num_people = len(anno)

        joints = np.zeros((num_people, self.num_joints, 3))

        for i, obj in enumerate(anno):
            joints[i, :self.num_joints_without_center, :3] = np.array(obj['keypoints']).reshape([-1, 3])
            
        return joints

    def get_mask(self, anno, idx):
        coco = self.coco
        img_info = coco.loadImgs(self.ids[idx])[0]

        m = np.zeros((img_info['height'], img_info['width']))

        return m < 0.5
    

def getDatasetProcessed(split):

    if split not in ["train", "validation", "trainval","test"]:
        raise Exception(f"Expected a dataset split train, validation or test, given {split}")

    datasetPath = config["dataset_root"]

    split = "val" if split == "validation" else "val"

    hm = [
    HeatmapGenerator(
            output_size, config["num_joints"], 2
        ) for output_size in [64, 128]
    ]

    j = [
        JointsGenerator(
            config["max_people"],
            config["num_joints"],
            output_size,
            True
        ) for output_size in [64, 128]
    ]

    transforms = T.Compose(
            [
                T.RandomAffineTransform(
                    256,
                    [64, 128],
                    30,
                    0.75,
                    1.5,
                    'short',
                    40,
                    scale_aware_sigma=None
                ),
                T.RandomHorizontalFlip([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13], [64, 128], 0.5),
                T.ToTensor()
            ]
        )
    
    return CrowdPoseKeypoints(datasetPath,split,True,hm,j,transforms)
