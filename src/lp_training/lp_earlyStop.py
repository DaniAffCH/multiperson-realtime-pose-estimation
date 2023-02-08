import torch
import os 
class EarlyStopping():
    def __init__(self, model, eps, threshold, savePath="backupModel"):
        self.eps = eps
        self.threshold = threshold
        self.count = 0
        self.savePath = os.path.join("lp_trained_models", savePath)
        self.model = model
        self.best = torch.inf
    def __call__(self, loss):
        if loss < self.best - self.eps:
            torch.save(self.model.state_dict(), self.savePath)
            self.count = 0
            self.best = loss
        else:
            self.count += 1
            if self.count > self.threshold:
                return True #stop!
        
        return False