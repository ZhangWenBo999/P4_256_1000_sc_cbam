import torch
import torch.nn as nn

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureFusion, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=8, dilation=8)

    def forward(self, x):
        relu = nn.ReLU()

        out1 = self.conv1(x)
        out1 = relu(out1)

        out2 = self.conv2(x)
        out2 = relu(out2)

        out3 = self.conv3(x)
        out3 = relu(out3)

        out4 = self.conv4(x)
        out4 = relu(out4)

        out = torch.cat([out1, out2, out3, out4], dim=1)

        return out

