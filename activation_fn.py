import torch
import torch.nn as nn
import numpy as np
import os

# Module for activation logging
class LoggingActivation(nn.Module):
    def __init__(self, activation, layer_name, log_interval=15):
        super(LoggingActivation, self).__init__()
        self.activation = activation
        self.layer_name = layer_name
        self.log_interval = log_interval
        self.model_name = None
        self.epoch = 0
        self.processed = False

    def forward(self, x):
        x = self.activation(x)
        
        # Log activations only during eval at each log_interval epoch and only with 8 imgs of first batch
        if not self.training and not self.processed and ((self.epoch + 1) % self.log_interval == 0):
            self.processed = True
            save_path = f'raw_np/{self.model_name}/activations/{self.layer_name}_epoch_{self.epoch + 1}.npy'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, x[:8].detach().cpu().numpy())
            
        return x

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.processed = False

    def set_model_name(self, model_name):
        self.model_name = model_name

# Utility function for models to inform activations about current epoch
def set_epoch(model, epoch):
    for module in model.modules():
        if isinstance(module, LoggingActivation):
            module.set_epoch(epoch)

# Utility function for models to inform activations about new model_name
def set_model_name(model, model_name):
    for module in model.modules():
        if isinstance(module, LoggingActivation):
            module.set_model_name(model_name)


AVAILABLE_ACTIVATIONS = ["relu", "hardtanh", "relu6"]
def get_activation_function(activation: str, quantize=False, layer_name=None):
    if activation == 'relu6':
        # ReLU6 is already quantizable
        act = nn.ReLU6(inplace=True)
    elif activation == 'relu':
        act = nn.ReLU(inplace=True)
    elif activation == 'hardtanh':
        act = nn.Hardtanh(inplace=True)
    else:
        raise ValueError("Unsupported activation: %s" % activation)
    
    if quantize and activation != 'relu6':
        # Put QuantStub after activation, to force quantization of it's output
        # (By default, Pytorch framework supposes that user fuses ReLU with previous layers
        # but you can't fuse with Hardtanh anyway)
        act = nn.Sequential(act, torch.quantization.QuantStub())

    if layer_name:
        act = LoggingActivation(act, layer_name=layer_name)
    
    return act
