import torch

config = dict(

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    learning_rate = 1e-3, 
    momentum = 0.5,
    epochs = 500,
    batch_size = 8,
    earlyStop_eps = 1e-8,
    earlyStop_threshold = 5,
    backup_name = "remove",
    num_joints = 14,
    max_people = 30,
    tag_loss_weight = 1e-2
)