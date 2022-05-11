"""Discriminator model for ADDA."""

from torch import nn


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2),
            nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
