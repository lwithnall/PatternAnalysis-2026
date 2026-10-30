import torch.nn as nn
from dataset import NUM_CLASSES


class ADNICNN(nn.Module):
    """
    Basic CNN implementation to classify AD in ADNI Dataset
    """

    def __init__(self):
        super(ADNICNN, self).__init__()
        self.features = nn.Sequential(
            self._cnn_block(1, 32),
            self._cnn_block(32, 64),
            self._cnn_block(64, 128),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, NUM_CLASSES),
        )

    @staticmethod
    def _cnn_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
