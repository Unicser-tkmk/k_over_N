import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import argparse

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument parser for hyperparameters
parser = argparse.ArgumentParser(description='Train ResNet101 on ImageNet with custom settings.')
parser.add_argument('--num_epochs', type=int, default=400, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for optimizer')
parser.add_argument('--accumulation_steps', type=int, default=10, help='Number of steps to accumulate gradients')
parser.add_argument('--loss_rate', type=float, default=0.0, help='Rate of gradients to be set to zero or ignored')
parser.add_argument('--mtu', type=int, default=300, help='Maximum Transmission Unit for gradient fragmentation')
parser.add_argument('--use_zero', action='store_true', help='Set to zero out unselected gradients')
parser.add_argument('--use_avg', action='store_true', help='Set to use mean of selected gradients')
parser.add_argument('--lab_round', type=int, default=5, help='Number of times to repeat the training process')
parser.add_argument('--worker_drop', action='store_true', help='Set to drop from workers side')

args = parser.parse_args()

# Set hyperparameters from arguments
num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
accumulation_steps = args.accumulation_steps
loss_rate = args.loss_rate
MTU = args.mtu
use_zero = args.use_zero
use_avg = args.use_avg
lab_round = args.lab_round
worker_drop=args.worker_drop

output_dir = f"./WD.{worker_drop}_Zero.{use_zero}__Avg.{use_avg}/MTU{MTU}_Wnum{accumulation_steps}_LR{loss_rate}"
os.makedirs(output_dir, exist_ok=True)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Path to the ImageNet dataset
data_dir = './imagenette2'

image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                                          transform=data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                             shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Load the pre-trained ResNet101 model
model = models.resnet101(pretrained=False)

# Modify the final layer to match the number of classes in the dataset
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# Move the model to the GPU
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Function to zero gradients in a list of tensors
def zero_grads(grads):
    for grad in grads:
        if grad is not None:
            grad.zero_()

# Function to accumulate gradients from multiple batches
def accumulate_grads(grads, model):
    for param, grad in zip(model.parameters(), grads):
        if grad is not None:
            if param.grad is None:
                param.grad = grad.clone().view(param.size())
            else:
                param.grad += grad.view(param.size())

# Function to randomly set loss_rate percent of gradients to zero or ignore them
def apply_loss_rate_to_fragments(batch_gradients, loss_rate, use_zero):
    fragmented_gradients = list(zip(*batch_gradients))
    for i, grads in enumerate(fragmented_gradients):
        grads = list(grads)  # Convert tuple to list for mutability
        num_to_zero = int(loss_rate * len(grads))
        if num_to_zero >= len(grads):
            num_to_zero = len(grads) - 1  # Ensure at least one gradient remains
        zero_indices = random.sample(range(len(grads)), num_to_zero)
        for idx in zero_indices:
            if use_zero:
                grads[idx] = torch.zeros_like(grads[idx])
            else:
                grads[idx] = None
        fragmented_gradients[i] = grads  # Assign modified list back to fragmented_gradients
    return list(zip(*fragmented_gradients))  # Convert back to original format

# Worker-defined pkt drop
def Worker_apply_loss_rate_to_fragments(batch_gradients, loss_rate, use_zero):
    fragmented_gradients = list(batch_gradients)
    for i, grads in enumerate(fragmented_gradients):
        grads = list(grads)  # Convert tuple to list for mutability
        num_to_zero = int(loss_rate * len(grads))
        if num_to_zero >= len(grads):
            num_to_zero = len(grads) - 1  # Ensure at least one gradient remains
        zero_indices = random.sample(range(len(grads)), num_to_zero)
        for idx in zero_indices:
            if use_zero:
                grads[idx] = torch.zeros_like(grads[idx])
            else:
                grads[idx] = None
        fragmented_gradients[i] = grads  # Assign modified list back to fragmented_gradients

    
    fragmented_gradients = list(zip(*fragmented_gradients))
    for i in range(len(fragmented_gradients)):
        if fragmented_gradients[i]==None:
            fragmented_gradients[i]=[0]
    fragmented_gradients = list(zip(*fragmented_gradients))
    return fragmented_gradients  # Convert back to original format


# Function to save the training state
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

# Function to load the training state
def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return model, optimizer, epoch
    else:
        return model, optimizer, 0

# Function to save metrics to a CSV file
def save_metrics(epoch, train_loss, val_loss, train_acc, val_acc, path):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write('epoch,train_loss,val_loss,train_acc,val_acc\n')
    with open(path, 'a') as f:
        f.write(f'{epoch},{train_loss},{val_loss},{train_acc},{val_acc}\n')

# Training function
def train_model(model, criterion, optimizer, num_epochs, metrics_path, checkpoint_path):
    since = time.time()

    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        epoch_start = time.time()

        train_loss = 0.0
        val_loss = 0.0
        train_corrects = 0
        val_corrects = 0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            batch_gradients = []
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()

                        # Accumulate gradients
                        batch_gradients.append([param.grad.clone() for param in model.parameters() if param.requires_grad])

                        if len(batch_gradients) == accumulation_steps:
                            # Apply loss rate to gradients
                            if worker_drop:
                                batch_gradients = Worker_apply_loss_rate_to_fragments(batch_gradients, loss_rate, use_zero)
                            else:
                                batch_gradients = apply_loss_rate_to_fragments(batch_gradients, loss_rate, use_zero)
                                
                            if use_avg:
                                selected_gradients = [torch.stack([g for g in grads if g is not None]).mean(dim=0) for grads in zip(*batch_gradients)]
                            else:
                                selected_gradients = [torch.stack([g for g in grads if g is not None]).sum(dim=0) for grads in zip(*batch_gradients)]

                            # Zero the model's gradients
                            optimizer.zero_grad()
                            # Accumulate the selected gradients
                            accumulate_grads(selected_gradients, model)
                            # Perform optimizer step
                            optimizer.step()
                            # Clear the accumulated gradients
                            batch_gradients.clear()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Calculate time per batch and estimate remaining time
                batch_time = time.time() - epoch_start
                total_batches = len(dataloaders[phase])
                estimated_total_time = batch_time / (batch_idx + 1) * total_batches
                remaining_time = estimated_total_time - batch_time
                print(f'\r[{phase}] Batch {batch_idx + 1}/{total_batches} - '
                      f'ETA: {int(remaining_time // 60)}m {int(remaining_time % 60)}s', end='')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss = epoch_loss
                train_corrects = epoch_acc.item()
            else:
                val_loss = epoch_loss
                val_corrects = epoch_acc.item()

            print(f'\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start
            print(f'Epoch {epoch} completed in {int(epoch_time // 60)}m {int(epoch_time % 60)}s')

        save_metrics(epoch, train_loss, val_loss, train_corrects, val_corrects, metrics_path)
        save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)

    total_time = time.time() - since
    print(f'Training complete in {int(total_time // 3600)}h {int((total_time % 3600) // 60)}m {int(total_time % 60)}s')

    return model

# Function to plot metrics from the CSV file
def plot_metrics(path, output_dir):
    if os.path.exists(path):
        data = pd.read_csv(path)
        epochs = data['epoch'] + 1

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, data['train_loss'], label='Training Loss')
        plt.plot(epochs, data['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, data['train_acc'], label='Training Accuracy')
        plt.plot(epochs, data['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        plt.savefig(os.path.join(output_dir, 'metrics_plot.png'))
        plt.show()
    else:
        print(f"No metrics file found at {path}")


# Run training for the specified number of lab rounds
for round_num in range(1, lab_round + 1):
    print(f'Starting training round {round_num}/{lab_round}')
    
    # Reinitialize model and optimizer for each round
    model = models.resnet101(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    round_metrics_path = os.path.join(output_dir, f'metrics{round_num}.csv')
    round_checkpoint_path = os.path.join(output_dir, f'checkpoint{round_num}.pth')
    
    # Train the model
    model = train_model(model, criterion, optimizer, num_epochs, round_metrics_path, round_checkpoint_path)
    
    # Save the final model for this round
    #torch.save(model.state_dict(), os.path.join(output_dir, f'resnet101_final_round{round_num}.pth'))
    
    # Plot the metrics for this round
    plot_metrics(round_metrics_path, output_dir)

print('All training rounds completed.')
