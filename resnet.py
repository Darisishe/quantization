import torch
import torch.nn as nn

from activation_fn import get_activation_function

def conv3x3(in_channels: int, out_channels: int, stride: int = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None, activation="relu", quantize=False
    ):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act_fn1 = get_activation_function(activation, quantize=quantize)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.skip_add = nn.quantized.FloatFunctional()

        self.act_fn2 = get_activation_function(activation, quantize=quantize)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_fn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # Use FloatFunctional for addition for quantization compatibility
        out = self.skip_add.add(residual, out)
        out = self.act_fn2(out)

        return out
    
class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        activation="relu",
        initial_channels=16,
        quantize=False,
    ):
        super(ResNet, self).__init__()
        self.quantize = quantize
        if quantize:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()

        self.in_channels = initial_channels
        self.initial_layer = nn.Sequential(
            conv3x3(3, initial_channels),
            nn.BatchNorm2d(initial_channels),
            # Never quantize first activation func
            get_activation_function(activation, quantize=False)
        )

        self.layer1 = self.make_layer(
            block, initial_channels, layers[0], activation=activation
        )
        self.layer2 = self.make_layer(
            block, initial_channels * 2, layers[1], stride=2, activation=activation, 
        )
        self.layer3 = self.make_layer(
            block, initial_channels * 4, layers[2], stride=2, activation=activation
        )
        if len(layers) == 4:
            self.layer4 = self.make_layer(
                block, initial_channels * 8, layers[3], stride=2, activation=activation
            )
        else:
            self.layer4 = None

        pool_size = 8 if initial_channels == 16 else 4
        fc_in_features = (
            initial_channels * 4 if len(layers) == 3 else initial_channels * 8
        )
        self.avg_pool = nn.AvgPool2d(pool_size)
        self.fc = nn.Linear(fc_in_features, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1, activation="relu"):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample, activation, quantize=self.quantize)
        )
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, activation=activation, quantize=self.quantize))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.initial_layer(x)
        # Start quantizing AFTER first layer
        if self.quantize:
            out = self.quant(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.layer4:
            out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        
        if self.quantize:
            out = self.dequant(out)
        out = self.fc(out)
        return out
    
def ResNet20(num_classes=10, activation='relu', quantize=False):
    return ResNet(ResidualBlock, [3, 3, 3], num_classes, activation, initial_channels=16, quantize=quantize)

def ResNet18(num_classes=10, activation='relu', quantize=False):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes, activation, initial_channels=64, quantize=quantize)
