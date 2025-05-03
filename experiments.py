import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from torch.quantization import FakeQuantize, QConfig

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lenet5 import LeNet5
from resnet import ResNet18, ResNet20, ResNet

cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 120
BATCH_SIZE = 128
LR = 1e-3
NUM_CLASSES = 10
QAT_EPOCHS = 60
QAT_LR = 1e-3

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    normalize])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

train_dataset = datasets.CIFAR10(root='data',
                                 train=True,
                                 transform=train_transform,
                                 download=True)

test_dataset = datasets.CIFAR10(root='data',
                                train=False,
                                transform=test_transform)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=8,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         num_workers=8,
                         shuffle=False)


def evaluate_model(model, device=cuda_device, criterion=None):
    model.eval()
    # We don't want Observers to change quantization params
    model.apply(torch.ao.quantization.disable_observer)
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    model.apply(torch.ao.quantization.enable_observer)
    return eval_loss, eval_accuracy

def train_model(
    model,
    learning_rate,
    num_epochs=NUM_EPOCHS,
    writer=None,
    model_name="model",
    device=cuda_device,
):
    patience = 15
    cur_idle = 0
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4
    )

    if "quantized" in model_name:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 45], gamma=0.25, last_epoch=-1
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[70, 100, 130], gamma=0.1, last_epoch=-1
        )

    best_eval_acc = 0
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0
        running_corrects = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        eval_loss, eval_accuracy = evaluate_model(
            model=model, device=device, criterion=criterion
        )

        scheduler.step()
        if writer:
            writer.add_scalar(f"TrainLoss/{model_name}", train_loss, epoch)
            writer.add_scalar(f"TrainAcc/{model_name}", train_accuracy, epoch)
            writer.add_scalar(f"EvalLoss/{model_name}", eval_loss, epoch)
            writer.add_scalar(f"EvalAcc/{model_name}", eval_accuracy, epoch)

        print(
            "[{}] Epoch: {:03d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
                model_name,
                epoch + 1,
                train_loss,
                train_accuracy,
                eval_loss,
                eval_accuracy,
            )
        )

        cur_idle += 1
        if eval_accuracy > best_eval_acc:
            best_eval_acc = eval_accuracy
            torch.save(model.state_dict(), f"checkpoint/best_{model_name}.ckpt")
            cur_idle = 0
        elif cur_idle >= patience:
            print("Early stopping was triggered!")
            break


def train_orig_model(model_class, activation):
    model_name = f"{model_class.__name__}_{activation}"
    model = model_class(activation=activation).to(cuda_device)
    writer = SummaryWriter(f"runs/{model_name}")

    train_model(
        model,
        learning_rate=LR,
        num_epochs=NUM_EPOCHS,
        writer=writer,
        model_name=model_name,
        device=cuda_device,
    )

    model.eval()
    torch.save(model.state_dict(), f"checkpoint/{model_name}.ckpt")
    writer.close()
    return model


def configure_qat(model, activation_bitwidth=4, weight_bitwidth=4):
    # Fake quantizer for activations
    fq_activation = FakeQuantize.with_args(
        observer=torch.quantization.MovingAverageMinMaxObserver.with_args(
            quant_min=0,
            quant_max=2**activation_bitwidth - 1,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
        )
    )

    # Fake quantizer for weights
    fq_weights = FakeQuantize.with_args(
        observer=torch.quantization.MovingAveragePerChannelMinMaxObserver.with_args(
            quant_min=-(2**weight_bitwidth) // 2,
            quant_max=(2**weight_bitwidth) // 2 - 1,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=False,
            ch_axis=0,
        )
    )

    # We don't want non-activation layers to quantize it's output
    # (for example, Conv2d)
    # Because it will harm subsequent Activation performance
    # (other solution will be to fuse Activation with previous layer
    # but pytorch can't fuse with anything, except for nn.ReLU)
    weight_only_qconfig = QConfig(
        activation=torch.quantization.NoopObserver.with_args(dtype=torch.float32),
        weight=fq_weights,
    )
    
    activation_qconfig = QConfig(activation=fq_activation, weight=fq_weights)
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Hardtanh, nn.ReLU6, nn.ReLU, torch.quantization.QuantStub, torch.quantization.DeQuantStub)):
            module.qconfig = activation_qconfig
        else:
            module.qconfig = weight_only_qconfig

     # Avoid quantizing first and last layers
    if isinstance(model, LeNet5):
        # First Layer
        model.conv1.qconfig = None
        for name, module in model.conv1.named_modules():
            module.qconfig = None
        model.fc3.qconfig = None # Last Linear

    elif isinstance(model, ResNet):
        model.initial_layer.qconfig = None
        for name, module in model.initial_layer.named_modules():
            module.qconfig = None
        model.fc.qconfig = None # Last Linear

    torch.quantization.prepare_qat(model, inplace=True)

def train_quantized_model(model_class, activation, ckpt_path):
    device=cuda_device
    
    model_name = f"{model_class.__name__}_{activation}_quantized"
    model = model_class(activation=activation, quantize=True).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    model.train()
    configure_qat(model)
    
    print(f"[{model_name}] after configure_qat:", model)
    
    # fine-tune via QAT
    writer = SummaryWriter(f"runs/{model_name}")
    train_model(model, learning_rate=QAT_LR, num_epochs=QAT_EPOCHS, writer=writer, model_name=model_name, device=device)

    # Save quantized model
    model.eval()
    torch.save(model.state_dict(), f"checkpoint/{model_name}.ckpt")
    writer.close()
    return model


# All possible architectures and activations
models = [LeNet5, ResNet20, ResNet18]
activations = ['relu', 'hardtanh', 'relu6']
bit_widths = [4, 3, 2]

if __name__ == '__main__':
    # Train all combinations (without quantization)
    trained_models = {}
    for model_class in models:
        for activation in activations:
            print(f"\nTraining {model_class.__name__} with {activation}")
            trained_models[(model_class.__name__, activation)] = train_orig_model(model_class, activation)

    # Train quantized versions
    quantized_models = {}
    for model_class in models:
        for activation in activations:
            ckpt_path = f"checkpoint/{model_class.__name__}_{activation}.ckpt"
            print(f"\nQuantizing {model_class.__name__} with {activation}")
            quantized_models[(model_class.__name__, activation)] = train_quantized_model(model_class, activation, ckpt_path)
    


    results = {}
    for model_class in models:
        for activation in activations:

            key = (model_class.__name__, activation)
            orig_model = model_class(activation=activation).to(cuda_device)
            quant_model = model_class(activation=activation, quantize=True).to(cuda_device)
            configure_qat(quant_model)
            
            orig_model.eval()
            quant_model.eval()
            orig_model.load_state_dict(torch.load(f"checkpoint/best_{model_class.__name__}_{activation}.ckpt"))
            quant_model.load_state_dict(torch.load(f"checkpoint/best_{model_class.__name__}_{activation}_quantized.ckpt"), strict=True)

            print(quant_model)
            _, orig_acc = evaluate_model(orig_model)
            _, quant_acc = evaluate_model(quant_model)

            results[key] = {
                "orig_acc": orig_acc,
                "quant_acc": quant_acc,
            }


    for key, result in results.items():
        print(f"\nModel: {key[0]} with {key[1]}")
        print(f"Original Accuracy: {result['orig_acc']:.4f}%")
        print(f"Quantized Accuracy: {result['quant_acc']:.4f}%")