# generator.py 

from torch import nn


# Reference: https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar.py

class Generator(nn.Module):
    def __init__(self, latent_len: int = 100, dim: int = 64):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # Input is Z, going into a linear layer
            nn.Linear(latent_len, 4*4*4*dim),
            nn.BatchNorm1d(4*4*4*dim),
            nn.ReLU(True),

            # Reshape to a 4x4 feature map
            nn.Unflatten(1, (4*dim, 4, 4)),
            # Upsample to 8x8

            nn.ConvTranspose2d(4*dim, 2*dim, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(True),

            # Upsample to 16x16
            nn.ConvTranspose2d(2*dim, dim, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            
            # Upsample to 32x32
            nn.ConvTranspose2d(dim, 3, 5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )
        

    def forward(self, noise):
        output = self.main(noise)
        return output.view(-1, 3, 32, 32)

