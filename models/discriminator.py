# discriminator.py

from typing import Tuple

import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, in_shape: Tuple[int, ...]) -> None:
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
           
            # Input 
            nn.Conv2d(in_shape[0], 64, 3, stride=2, padding=2), 
            nn.LeakyReLU(0.2),

            # Downsampling layers
            nn.Conv2d(64, 128, 3, stride=2, padding=2),
            nn.LeakyReLU(0.2), 

            nn.Conv2d(128, 128, 3, stride=2, padding=2),
            nn.LeakyReLU(0.2), 

            nn.Conv2d(128, 256, 3, stride=2, padding=2),
            nn.LeakyReLU(0.2), 

            # Outputs
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=4096, out_features=1),
            nn.Sigmoid() 
        )


    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


"""

# Translated to pytorch from this repo:
# Reference: https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar.py

class Discriminator(nn.Module):
    def __init__(self, dim: int = 64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            # Input is 3 x 32 x 32
            nn.Conv2d(3, dim, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),

            # State size: dim x 16 x 16
            nn.Conv2d(dim, 2 * dim, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),

            # State size: 2*dim x 8 x 8
            nn.Conv2d(2 * dim, 4 * dim, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),

            # State size: 4*dim x 4 x 4
            nn.Flatten(),
            nn.Linear(4 * 4 * 4 * dim, 1)
        )

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)

"""