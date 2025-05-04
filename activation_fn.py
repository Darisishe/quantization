import torch
import torch.nn as nn

AVAILABLE_ACTIVATIONS = ["relu", "hardtanh", "relu6"]

def get_activation_function(activation: str, quantize=False):
    if activation == 'relu6':
        # ReLU6 is already quantizable
        return nn.ReLU6(inplace=True)
    if activation == 'relu':
        act = nn.ReLU(inplace=True)
    elif activation == 'hardtanh':
        act = nn.Hardtanh(inplace=True)
    else:
        raise ValueError("Unsupported activation: %s" % activation)
    
    if not quantize:
        return act
    else:
        # Put QuantStub after activation, to force quantization of it's output
        # (By default, Pytorch framework supposes that user fuses ReLU with previous layers
        # but you can't fuse with Hardtanh anyway)
        return nn.Sequential(act, torch.quantization.QuantStub())
