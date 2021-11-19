import torch
import torch.nn as nn


class ClassifierNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.Dropout(.3),
            nn.LeakyReLU(.2),

            #nn.BatchNorm1d(512),

            nn.Linear(512, 512 + 256),
            nn.Dropout(.3),
            nn.LeakyReLU(.2),

            #nn.BatchNorm1d(512 + 256),

            nn.Linear(512 + 256, 256+128),
            nn.Dropout(.3),
            nn.LeakyReLU(.2),

            #nn.BatchNorm1d(256+128),

            nn.Linear(256+128, 128),
            nn.Dropout(.3),
            nn.LeakyReLU(.2),

            #nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            # nn.Dropout(.2),
            nn.LeakyReLU(.2),

            nn.Linear(64, 16),
            # nn.Dropout(.2),
            nn.LeakyReLU(.2),

            nn.Linear(16, 1),
            #nn.Sigmoid()
        ).type(torch.float)

    def forward(self, x):
        return self.net.forward(x)
