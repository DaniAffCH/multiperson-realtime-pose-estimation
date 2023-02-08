import torch

config = dict(
    #COMMON
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    num_joints = 14,
    max_people = 30,

    #TRAINING
    batch_size = 8,
    learning_rate = 1e-3, 
    momentum = 0.5,
    epochs = 500,
    earlyStop_threshold = 8,
    earlyStop_eps = 1e-8,
    backup_name = "theirtag",
    tag_loss_weight = 1e-2,

    #INFERENCE
    confidence_threshold = 0.22,
    confidence_embedding = 0.95,
)


crowd_pose_part_labels = [
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'head', 'neck'
]
crowd_pose_part_idx = {
    b: a for a, b in enumerate(crowd_pose_part_labels)
}
crowd_pose_part_orders = [
    ('head', 'neck'), ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'), ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'), ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'), ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle')
]