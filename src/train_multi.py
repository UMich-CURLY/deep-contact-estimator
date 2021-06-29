import numpy as np
import torch

from mtl_models import MultiTaskModel
from loss import MultiTaskLoss

# Initialize multi-task models and losses
tasks_dict = {
  "contact" : 16,
  "terrain" : 8
}
model = MultiTaskModel(tasks_dict)
criterion = MultiTaskLoss(tasks_dict)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Generate some random data and labels
input_data = torch.from_numpy(np.random.rand(100, 20, 54)).float()
labels = {
  "contact" : torch.from_numpy(np.random.randint(16, size=100)).long(),
  "terrain" : torch.from_numpy(np.random.randint(8, size=100)).long()
}

# Start training
model.train()
for epoch in range(1000):
    output = model(input_data)
    loss = criterion(output, labels) # labels and loss are a dict
    print(loss)
    
    optimizer.zero_grad()
    loss['total'].backward()
    optimizer.step()
