
import torch
import torch.nn as nn
from activation_fn import get_activation_function

class LeNet5(nn.Module):
    def __init__(self, num_classes=10, activation="relu", quantize=False):
        super(LeNet5, self).__init__()
        self.quantize = quantize
        if quantize:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            # Never quantize first activation func
            get_activation_function(activation, quantize=False, layer_name="conv1.activation"),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            get_activation_function(activation, quantize=quantize, layer_name="conv2.activation"),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(400, 120),
            get_activation_function(activation, quantize=quantize, layer_name="fc1.activation")
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            get_activation_function(activation, quantize=quantize, layer_name="fc2.activation")
        )
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        if self.quantize:
            x = self.quant(x)
            
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        if self.quantize:
            x = self.dequant(x)
        x = self.fc3(x)
        return x
