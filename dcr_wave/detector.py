from torch import nn


class DCRWaveDetector(nn.Module):
    def __init__(self, backbone: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, images, targets=None):
        features = self.backbone(images)
        return self.decoder(features, targets)
