import torch

config = dict(

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    learning_rate = 0.001,
    momentum = 0.5,
    epochs = 100,
    batch_size = 32,
    earlyStop_eps = 1e-4,
    earlyStop_threshold = 5,
    backup_name = "2901",
    num_joints = 17
)