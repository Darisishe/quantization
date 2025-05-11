import torch
import torch.nn as nn
import numpy as np
import os

class ParametrizedReLU(nn.Module):
    def __init__(self, init_alpha=6.0):
        super(ParametrizedReLU, self).__init__()
        # Define alpha as a learnable parameter
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, x):
        # Clip activations to [0, alpha]
        min_val = torch.tensor(0.0, dtype=self.alpha.dtype, device=self.alpha.device)
        x = torch.clamp(x, min_val, self.alpha)
        return x
    
class ParameterizedHardtanh(nn.Module):
    def __init__(self, init_beta=-1.0, init_alpha=1.0):
        super(ParameterizedHardtanh, self).__init__()
        self.beta = nn.Parameter(torch.tensor(init_beta))  # Learnable lower bound
        self.alpha = nn.Parameter(torch.tensor(init_alpha))  # Learnable upper bound

    def forward(self, x):
        # Clip activations to the range [beta, alpha]
        x = torch.clamp(x, self.beta, self.alpha)
        return x


# Module for activation logging
class LoggingActivation(nn.Module):
    def __init__(self, activation, layer_name, log_interval=20):
        super(LoggingActivation, self).__init__()
        self.activation = activation
        self.layer_name = layer_name
        self.log_interval = log_interval
        self.model_name = None
        self.epoch = 0
        self.processed = False
        self.params_logs = []

    def forward(self, x):
        x = self.activation(x)
        
        if not self.training and not self.processed:
            params = [param.item() for param in self.get_params()]
            # Log learnable params each eval
            if params:
                self.params_logs.append(params)
            
            self.processed = True
            
            # Log activations only during eval at each log_interval epoch and only with 8 imgs of first batch
            if (self.epoch + 1) % self.log_interval == 0:
                save_path = f'raw_np/{self.model_name}/activations/{self.layer_name}_epoch_{self.epoch + 1}.npz'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savez_compressed(save_path, activations=x[:8].detach().cpu().numpy(), params=params)

        return x

    def dump_params_stat(self):
        if self.params_logs:
            save_path = f'raw_np/{self.model_name}/activations_parameters/{self.layer_name}.npy'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, np.array(self.params_logs))

    def get_params(self):
        for _, module in self.named_modules():
            if isinstance(module, ParameterizedHardtanh):
                return [module.beta, module.alpha]
            elif isinstance(module, ParametrizedReLU):
                return [module.alpha]
            
        return []

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

# Utility function to dump parametrized activation parameters statistic to files
def dump_params_stat(model):
    for module in model.modules():
        if isinstance(module, LoggingActivation):
            module.dump_params_stat()



AVAILABLE_ACTIVATIONS = ["relu", "hardtanh", "relu6", "parametrized_relu", "parametrized_hardtanh"]
def get_activation_function(activation: str, quantize=False, layer_name=None):
    if activation == 'relu6':
        # ReLU6 is already quantizable
        act = nn.ReLU6(inplace=True)
    elif activation == 'relu':
        act = nn.ReLU(inplace=True)
    elif activation == 'hardtanh':
        act = nn.Hardtanh(inplace=True)
    elif activation == "parametrized_relu":
        act = ParametrizedReLU()
    elif activation == "parametrized_hardtanh":
        act = ParameterizedHardtanh()
    else:
        raise ValueError("Unsupported activation: %s" % activation)
    
    if activation != 'relu6':
        if quantize:
            # Put QuantStub after activation, to force quantization of it's output
            # (By default, Pytorch framework supposes that user fuses ReLU with previous layers
            # but you can't fuse with Hardtanh anyway)
            act = nn.Sequential(act, torch.quantization.QuantStub())
        else:
            # Do this, so load_state_dict will work properly
            act = nn.Sequential(act)

    if layer_name:
        act = LoggingActivation(act, layer_name=layer_name)
    
    return act
