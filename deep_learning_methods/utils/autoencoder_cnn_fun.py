import torch.nn as nn
from tqdm import tqdm
import torch

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=2,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2, out_channels=16, kernel_size=(2, 2), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=1,
                      kernel_size=3,
                      stride=1,
                      padding=1),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, 10, 20)
        out = self.encoder(x)
        x_h = self.decoder(out)

        return x_h

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=4,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, 10, 20)
        out = self.encoder(x)
        out = out.reshape(out.shape[0], -1)

        return out

def transfer_data(dataloader, nt):
    features = []
    labels = []

    for data in tqdm(dataloader):
        x, y = data['data'], data['label']
        x = x.reshape(x.shape[0], -1)
        x = x.cuda()
        y_h = nt(x)
        y_np = y_h.detach().cpu()
        features.append(y_np)
        labels.append(y)
    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()
    return features, labels




