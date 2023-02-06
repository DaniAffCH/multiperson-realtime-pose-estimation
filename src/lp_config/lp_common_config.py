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
    earlyStop_threshold = 5,
    earlyStop_eps = 1e-8,
    backup_name = "definitiveCrowd",
    tag_loss_weight = 1e-2,

    #INFERENCE
    confidence_threshold = 0.19
)