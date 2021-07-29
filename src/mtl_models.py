import torch.nn as nn

class FCHead(nn.Sequential):
    def __init__(self, in_features, num_classes):
      num_features = [2048, 512]
      super(FCHead, self).__init__(
        nn.Linear(in_features, out_features=num_features[0]),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=num_features[0], out_features=num_features[1]),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=num_features[1], out_features=num_classes)
        )

class MultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared conv layers + task-specific fc layers """
    def __init__(self, tasks_dict: dict):
        super(MultiTaskModel, self).__init__()
        self.tasks_dict = tasks_dict

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=54,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=2,
                         stride=2) 
        )

        """ TODO: change the 640 accordingly, or make it a parameter"""
        self.heads = nn.ModuleDict({task: FCHead(4736, num_classes) for task, num_classes in self.tasks_dict.items()})

    def forward(self, x):
        x = x.permute(0,2,1)
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        shared_features = block2_out.view(block2_out.shape[0], -1)
        return {task: self.heads[task](shared_features) for task, _ in self.tasks_dict.items()}
