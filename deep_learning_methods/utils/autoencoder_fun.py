import torch.nn as nn
import numpy as np
from tqdm import tqdm


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
                    nn.Linear(in_features=200,
                              out_features=512),
                    nn.ReLU(),
                    nn.Linear(in_features=512,
                              out_features=1024),
                    nn.ReLU(),
                    nn.Linear(in_features=1024,
                              out_features=4096),
                    nn.ReLU(),
                    nn.Linear(in_features=4096,
                              out_features=20),
                )

        self.decoder = nn.Sequential(
                    nn.Linear(in_features=20,
                              out_features=4096),
                    nn.ReLU(),
                    nn.Linear(in_features=4096,
                              out_features=1024),
                    nn.ReLU(),
                    nn.Linear(in_features=1024,
                              out_features=512),
                    nn.ReLU(),
                    nn.Linear(in_features=512,
                              out_features=200),
                )

    def forward(self, x):
        out = self.encoder(x)
        x_h = self.decoder(out)
        return x_h


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=200,
                      out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,
                      out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,
                      out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024,
                      out_features=20),
        )

    def forward(self, x):
        out = self.encoder(x)

        return out


def transfer_data(dataloader, nt):
    features = []
    labels = []
    for data in tqdm(dataloader):
        x, y = data['data'], data['label']
        x = x.reshape(x.shape[0], -1)
        x = x.cuda()
        y_h = nt(x)
        y_np = y_h.detach().cpu().numpy().reshape(10, )
        features.append(y_np)
        labels.append(y.numpy()[0])

    return np.array(features), np.array(labels)