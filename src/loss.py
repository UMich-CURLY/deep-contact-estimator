import torch
import torch.nn as nn

""" 
TODO: might need to implement different loss_ft for different task,
      and add loss weights
"""
class MultiTaskLoss(nn.Module):
    def __init__(self, tasks_dict):
        super(MultiTaskLoss, self).__init__()
        self.tasks_dict = tasks_dict
        self.loss_ft = nn.CrossEntropyLoss()

    def forward(self, pred, gt):
        out = {task: self.loss_ft(pred[task], gt[task]) for task,_ in self.tasks_dict.items()}
        out['total'] = torch.sum(torch.stack([out[task] for task,_ in self.tasks_dict.items()]))
        return out
