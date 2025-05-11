import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.quantization
from torch.quantization import FakeQuantize, QConfig

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lenet5 import LeNet5
from resnet import ResNet18, ResNet20, ResNet
import activation_fn

from logger import setup_logging, LoggerStream
import sys
import os

import atexit
import queue

import numpy as np

def terminate_children():
    for p in mp.active_children():
        p.terminate()


atexit.register(terminate_children)


NUM_EPOCHS = 120
BATCH_SIZE = 128
LR = 1e-3
NUM_CLASSES = 10
QAT_EPOCHS = 60
QAT_LR = 1e-3

def prepare_dataloader():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose(
        [
            transforms.Pad(4),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(
        root="data", train=True, transform=train_transform, download=True
    )

    test_dataset = datasets.CIFAR10(root="data", train=False, transform=test_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )

    return (train_loader, test_loader)


def print_model_with_weights(model, max_sample_size=25):
    # Print the model architecture
    print("Model Architecture:")
    print(model)
    print("\nModel Weights:")

    for name, module in model.named_modules():
        if not hasattr(module, 'weight'):
            continue
        weights = module.weight
        if hasattr(module, 'weight_fake_quant'):
            weights = module.weight_fake_quant(weights)

        print(f"\nLayer: {name}\nLayer Shape: {weights.shape}")


        # Flatten the parameter tensor for sampling
        flat_param = weights.flatten()
        total_elements = flat_param.numel()

        # Adjust sample_size to not exceed total elements
        sample_size = min(max_sample_size, total_elements)

        # Randomly sample indices and values
        sample_indices = torch.randperm(total_elements)[:sample_size]
        sample_values = flat_param[sample_indices].tolist()

        # Print the sampled values
        print(f"Sample Values ({sample_size} elements): {sample_values}")

        # Print summary statistics
        print(f"Mean: {weights.mean().item():.8f}")
        print(f"Min: {weights.min().item():.8f}")
        print(f"Max: {weights.max().item():.8f}")


def evaluate_model(model, device, test_loader, criterion=None):
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
    device,
    train_loader,
    test_loader,
    num_epochs=NUM_EPOCHS,
    writer=None,
    model_name="model",
):
    patience = 20
    cur_idle = 0

    criterion = nn.CrossEntropyLoss()

    model.to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4
    )
    # L2 Regularization for Activations Parameters already applied by SGD's weight_decay 

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[30, 45] if "quantized" in model_name else [70, 100, 130],
        gamma=0.25 if "quantized" in model_name else 0.1,
        last_epoch=-1,
    )

    activation_fn.set_model_name(model, model_name)

    train_losses = []
    train_accuracies = []
    eval_losses = []
    eval_accuracies = []
    learning_rates = []

    best_eval_acc = 0
    for epoch in range(num_epochs):
        activation_fn.set_epoch(model, epoch)
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
            model=model, device=device, test_loader=test_loader, criterion=criterion
        )

        scheduler.step()
        if writer:
            writer.add_scalar(f"TrainLoss", train_loss, epoch)
            writer.add_scalar(f"TrainAcc", train_accuracy, epoch)
            writer.add_scalar(f"EvalLoss", eval_loss, epoch)
            writer.add_scalar(f"EvalAcc", eval_accuracy, epoch)
            writer.add_scalar(
                f"LearningRate", optimizer.param_groups[0]["lr"], epoch
            )

            # Append to NumPy logging lists
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            eval_losses.append(eval_loss)
            eval_accuracies.append(eval_accuracy)
            learning_rates.append(optimizer.param_groups[0]["lr"])

        print(
            "[{}] Epoch: {:03d} Train Loss: {:.4f} Train Acc: {:.4f} Eval Loss: {:.4f} Eval Acc: {:.4f} (LR: {:.6f})".format(
                model_name,
                epoch + 1,
                train_loss,
                train_accuracy,
                eval_loss,
                eval_accuracy,
                optimizer.param_groups[0]["lr"],
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

    print(
        "[{}] Best Eval Accuracy: {:.4f}".format(
            model_name,
            best_eval_acc,
        )
    )

    # Save NumPy logs
    epochs = list(range(len(train_losses)))
    scalars = {
        'epoch': np.array(epochs),
        'train_loss': np.array(train_losses),
        'train_acc': np.array([acc.item() for acc in train_accuracies]),
        'eval_loss': np.array(eval_losses),
        'eval_acc': np.array([acc.item() for acc in eval_accuracies]),
        'learning_rate': np.array(learning_rates)
    }
    os.makedirs(f'raw_np/{model_name}', exist_ok=True)
    np.savez(f'raw_np/{model_name}/training_stats.npz', **scalars)

    activation_fn.dump_params_stat(model)


def train_orig_model(model_class, activation, device, train_loader, test_loader):
    model_name = f"{model_class.__name__}_{activation}"
    model = model_class(activation=activation).to(device)
    writer = SummaryWriter(f"runs/{model_name}")

    train_model(
        model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=LR,
        num_epochs=NUM_EPOCHS,
        writer=writer,
        model_name=model_name,
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
        observer=torch.quantization.MovingAverageMinMaxObserver.with_args(
            quant_min=0,
            quant_max=2**weight_bitwidth - 1,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
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
        if isinstance(
            module,
            (
                nn.Hardtanh,
                nn.ReLU6,
                nn.ReLU,
                torch.quantization.QuantStub,
                torch.quantization.DeQuantStub,
            ),
        ):
            module.qconfig = activation_qconfig
        else:
            module.qconfig = weight_only_qconfig

    # Avoid quantizing first and last layers
    if isinstance(model, LeNet5):
        # First Layer
        model.conv1.qconfig = None
        for name, module in model.conv1.named_modules():
            module.qconfig = None
        model.fc3.qconfig = None  # Last Linear

    elif isinstance(model, ResNet):
        model.initial_layer.qconfig = None
        for name, module in model.initial_layer.named_modules():
            module.qconfig = None
        model.fc.qconfig = None  # Last Linear

    torch.quantization.prepare_qat(model, inplace=True)


def train_quantized_model(
    model_class, activation, bits: int, ckpt_path, device, train_loader, test_loader
):
    model_name = f"{model_class.__name__}_{activation}_quantized_{bits}_bits"
    model = model_class(activation=activation, quantize=True).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    model.train()
    configure_qat(model, activation_bitwidth=bits, weight_bitwidth=bits)

    print(f"[{model_name}] after configure_qat:", model)

    # fine-tune via QAT
    writer = SummaryWriter(f"runs/{model_name}")
    train_model(
        model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=QAT_LR,
        num_epochs=QAT_EPOCHS,
        writer=writer,
        model_name=model_name,
    )

    # Save quantized model
    model.eval()
    torch.save(model.state_dict(), f"checkpoint/{model_name}.ckpt")
    writer.close()
    return model


def parallel_train(model_class, activation):
    cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_dataloader()

    bit_widths = [4, 3, 2]

    logger = setup_logging(f"{model_class.__name__}_{activation}")
    # Redirect prints to logfile
    sys.stdout = LoggerStream(logger)

    print(f"\nTraining {model_class.__name__} with {activation}")
    fp_model = train_orig_model(
        model_class,
        activation,
        device=cuda_device,
        train_loader=train_loader,
        test_loader=test_loader,
    )
    print(f"\nTraining of full-precision model finished!")
    print_model_with_weights(fp_model)

    fp_ckpt_path = f"checkpoint/{model_class.__name__}_{activation}.ckpt"
    quantized_models = dict()
    for bits in bit_widths:
        print(
            f"\n\n\nQAT of {model_class.__name__} with {activation} down to {bits} bits..."
        )
        quantized_model = train_quantized_model(
            model_class,
            activation,
            bits,
            ckpt_path=fp_ckpt_path,
            device=cuda_device,
            train_loader=train_loader,
            test_loader=test_loader,
        )
        quantized_models[(model_class.__name__, activation, bits)] = quantized_model
        print(f"\n\n\nQuantization of model down to {bits} bits finished")
        print_model_with_weights(quantized_model)


# Function to run tasks in parallel
def run_task(task_queue):
    while not task_queue.empty():
        try:
            model_class, activation = task_queue.get_nowait()
            parallel_train(model_class, activation)
        except queue.Empty:
            break


if __name__ == "__main__":
    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("raw_np", exist_ok=True)

    # All possible architectures, activations and bitwidths
    models = [LeNet5, ResNet20, ResNet18]
    activations = activation_fn.AVAILABLE_ACTIVATIONS
    bit_widths = [4, 3, 2]

    tasks = [(model, act) for model in models for act in activations]

    num_processes = min(len(tasks), 5)  # Limit processes to manage GPU memory

    task_queue = mp.Queue()
    for task in tasks:
        task_queue.put(task)

    # Start parallel processes
    processes = []
    for i in range(num_processes):
        p = mp.Process(target=run_task, args=(task_queue,))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # results = {}
    # for model_class in models:
    #     for activation in activations:

    #         key = (model_class.__name__, activation)
    #         orig_model = model_class(activation=activation).to(cuda_device)
    #         quant_model = model_class(activation=activation, quantize=True).to(
    #             cuda_device
    #         )
    #         configure_qat(quant_model)

    #         orig_model.eval()
    #         quant_model.eval()
    #         orig_model.load_state_dict(
    #             torch.load(f"checkpoint/best_{model_class.__name__}_{activation}.ckpt")
    #         )
    #         quant_model.load_state_dict(
    #             torch.load(
    #                 f"checkpoint/best_{model_class.__name__}_{activation}_quantized.ckpt"
    #             ),
    #             strict=True,
    #         )

    #         print(quant_model)
    #         _, orig_acc = evaluate_model(orig_model)
    #         _, quant_acc = evaluate_model(quant_model)

    #         results[key] = {
    #             "orig_acc": orig_acc,
    #             "quant_acc": quant_acc,
    #         }

    # for key, result in results.items():
    #     print(f"\nModel: {key[0]} with {key[1]}")
    #     print(f"Original Accuracy: {result['orig_acc']:.4f}%")
    #     print(f"Quantized Accuracy: {result['quant_acc']:.4f}%")
