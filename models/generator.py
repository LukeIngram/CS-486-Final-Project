# generator.py 

from torch import nn

# Translated to pytorch from this repo:
# Reference: https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar.py
# TODO REMOVE REFERENCE 

"""
class Generator(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super(Generator, self).__init__()

        dim = 256 * 4 * 4 
        self.main = nn.Sequential( 
            nn.Linear(latent_dim, dim),
            nn.LeakyReLU(0.2)

        )

         n_nodes = 256 * 4 * 4

        '''
        self.main = nn.Sequential(
            # Input is Z, going into a linear layer
            nn.Linear(dim, 4*4*4*dim),
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
        '''
        

    def forward(self, noise):
        output = self.main(noise)
        return output.view(-1, 3, 32, 32)
"""

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        # foundation for 4x4 image
        n_nodes = 256 * 4 * 4

        self.model = nn.Sequential(
            nn.Linear(latent_dim, n_nodes),
            nn.LeakyReLU(0.2),

            # Reshape layer, implemented in forward method
            # ConvTranspose layers with LeakyReLU
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Output layer
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Linear layer with reshape
        x = self.model[0](x)
        x = x.view(-1, 256, 4, 4)  # Reshape to (batch_size, 256, 4, 4)

        # Sequential layers
        for layer in self.model[1:]:
            x = layer(x)
        return x
