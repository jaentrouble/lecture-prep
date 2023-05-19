import torch
from torch import nn

class DCGANGenerator(nn.Module):
    def __init__(
            self,
            noise_size : int,
            output_size : tuple,
            output_channels : int,
    ):
        """DCGAN Generator
        Upscales 3 times from noise to output_size

        Parameters
        ----------
        noise_size : int
            size of the noise vector
        output_size : tuple
            output size (H,W)
            Assert that output_size[0] and output_size[1] are divisible by 8
        output_channels : int
            number of output channels
        """
        super().__init__()
        self.noise_size = noise_size
        self.output_size = output_size
        self.output_channels = output_channels
        self.first_layer_size = (output_size[0]//8, output_size[1]//8)
        assert self.first_layer_size[0]*8 == output_size[0], "output_size[0] must be divisible by 8"
        assert self.first_layer_size[1]*8 == output_size[1], "output_size[1] must be divisible by 8"
        self.fc = nn.Linear(noise_size, 256*self.first_layer_size[0]*self.first_layer_size[1])
        self.layers = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, output_channels, 5, stride=1, padding=2),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(-1, 256, self.first_layer_size[0], self.first_layer_size[1])
        x = self.layers(x)
        return x
