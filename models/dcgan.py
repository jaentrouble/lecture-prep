from torch import nn


class ResBlock(nn.Module):
    def __init__(
            self,
            input_channels : int,
            output_channels : int,
            leaky_relu : bool = False,
    ):
        """Residual block

        Assert that input_channels == output_channels
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.leaky_relu = leaky_relu
        self.relu = nn.LeakyReLU() if leaky_relu else nn.ReLU()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)
        return x + x1
    



class DCGANGenerator(nn.Module):
    def __init__(
            self,
            noise_size : tuple,
            output_size : tuple,
            output_channels : int,
    ):
        """DCGAN Generator
        Upscales 3 times from noise to output_size

        Parameters
        ----------
        noise_size : tuple
            size of the noise vector
            To match the convention, noise_size is a tuple, but it should have only one int
        output_size : tuple
            output size (H,W)
            Assert that output_size[0] and output_size[1] are divisible by 8
        output_channels : int
            number of output channels
        """
        super().__init__()
        assert len(noise_size) == 1, "noise_size should be a tuple with only one int"
        self.noise_size = noise_size[0]
        self.output_size = output_size
        self.output_channels = output_channels
        self.first_layer_size = (output_size[0]//8, output_size[1]//8)
        assert self.first_layer_size[0]*8 == output_size[0], "output_size[0] must be divisible by 8"
        assert self.first_layer_size[1]*8 == output_size[1], "output_size[1] must be divisible by 8"
        self.fc = nn.Linear(self.noise_size, 256*self.first_layer_size[0]*self.first_layer_size[1])
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

class DCGANGeneratorEX(nn.Module):
    def __init__(
        self,
        noise_size : tuple,
        output_size : tuple,
        output_channels : int,
        res_blocks: int,
    ):
        """DCGANGeneratorEX
        Upscales 3 times from noise to output_size
        
        Parameters
        ----------
        noise_size : tuple
            size of the noise vector
            To match the convention, noise_size is a tuple, but it should have only one int
        output_size : tuple
            output size (H,W)
            Assert that output_size[0] and output_size[1] are divisible by 8
        output_channels : int
            number of output channels
        res_blocks: int
            number of residual blocks
        """
        super().__init__()
        assert len(noise_size) == 1, "noise_size should be a tuple with only one int"
        self.noise_size = noise_size[0]
        self.output_size = output_size
        self.output_channels = output_channels
        self.first_layer_size = (output_size[0]//8, output_size[1]//8)
        assert self.first_layer_size[0]*8 == output_size[0], "output_size[0] must be divisible by 8"
        assert self.first_layer_size[1]*8 == output_size[1], "output_size[1] must be divisible by 8"
        self.fc = nn.Linear(self.noise_size, 256*self.first_layer_size[0]*self.first_layer_size[1])
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
            *[ResBlock(32, 32) for _ in range(res_blocks)],
            nn.Conv2d(32, output_channels, 5, stride=1, padding=2),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(-1, 256, self.first_layer_size[0], self.first_layer_size[1])
        x = self.layers(x)
        return x
    

class DCGANDiscriminator(nn.Module):
    def __init__(
        self,
        input_size : tuple,
        input_channels : int,
        normalize_layer : str = None,
    ):
        """DCGAN Discriminator

        Parameters
        ----------
        input_size : tuple
            input size (H,W)
        input_channels : int
            number of input channels
        normalize_layer : str
            None: no normalization
        """
        super().__init__()
        self.input_channels = input_channels
        
        h, w = input_size
        for _ in range(4):
            h = h // 2 + h % 2
            w = w // 2 + w % 2
        fc_flatten_size = h * w * 256

        if normalize_layer is None:
            norm_layer = nn.Identity
        elif normalize_layer.lower() == 'batchnorm':
            norm_layer = nn.BatchNorm2d
        elif normalize_layer.lower() == 'instancenorm':
            norm_layer = nn.InstanceNorm2d

        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, stride=2, padding=2),
            norm_layer(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            norm_layer(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            norm_layer(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            norm_layer(256),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(fc_flatten_size, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class DCGANDiscriminatorEX(nn.Module):
    def __init__(
        self,
        input_size : tuple,
        input_channels : int,
        res_blocks: int,
        normalize_layer : str = None,
    ):
        """DCGAN Discriminator"""
        super().__init__()
        self.input_channels = input_channels
        
        fc_flatten_size = input_size[0] * input_size[1]
        for _ in range(4):
            fc_flatten_size = fc_flatten_size // 2 + fc_flatten_size % 2
        fc_flatten_size *= 256

        if normalize_layer is None:
            norm_layer = nn.Identity
        elif normalize_layer.lower() == 'batchnorm':
            norm_layer = nn.BatchNorm2d
        elif normalize_layer.lower() == 'instancenorm':
            norm_layer = nn.InstanceNorm2d

        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, 1, stride=1, padding=0),
            norm_layer(16),
            nn.LeakyReLU(),
            *[ResBlock(16, 16) for _ in range(res_blocks)],
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            norm_layer(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            norm_layer(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            norm_layer(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            norm_layer(256),
            nn.LeakyReLU(),
            
            nn.Flatten(),
            nn.Linear(fc_flatten_size, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x