# Nicholas Moreland
# 05/02/2024

import time
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
import copy
from helper import AverageMeter, accuracy

# Train the model


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Copy the model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Iterate over the dataset multiple times
    for epoch in range(num_epochs):
        losses = AverageMeter()
        top1 = AverageMeter()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Iterate over the dataset
        for i, (inputs, labels) in enumerate(training_data):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outputs = outputs.float()
            loss = loss.float()

            prec = accuracy(outputs.data, labels)[0]

            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec.item(), inputs.shape[0])

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            scheduler.step()

            epoch_acc = running_corrects.double() / training_size

            print('Epoch {}/{} step {}/{} train Loss: {:.4f} Acc: {:.4f}'.format(epoch,
                  num_epochs - 1, i, len(training_data), losses.avg, epoch_acc))
            logger.add_scalar('Training Loss', losses.avg, (epoch + 1)*i)
            logger.add_scalar('Training Accuracy', top1.avg, (epoch + 1)*i)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    return model

# Validate the model


def validate_model(model, criterion, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Iterate over the dataset multiple times
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.eval()

        running_loss = 0.0
        running_corrects = 0

        for i, (inputs, labels) in enumerate(validation_data):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / validation_size
        epoch_acc = running_corrects.double() / validation_size

        print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


# Create a summary writer
logger = SummaryWriter("runs/transfer_learning")

# Define the mean and standard deviation
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

# Define the data directory
data_dir = './food101data'

# Define the transformation
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Load the dataset
dataset = torchvision.datasets.Food101(data_dir, transform=transforms.Compose([
    transforms.RandAugment(),
    transforms.RandomResizedCrop(100),
    transforms.ToTensor(),
    normalize
]), download=True)

# Get the size of the dataset
dataset_size = len(dataset)
training_size = int(dataset_size * .8)
validation_size = dataset_size - training_size

# Split the dataset into training and validation sets
training_dataset, validation_dataset = torch.utils.data.random_split(
    dataset, [training_size, validation_size])

# Load the training data
training_data = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=4, shuffle=True, pin_memory=False)

# Load the validation data
validation_data = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=4, shuffle=False, pin_memory=False)

# Set device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load a pretrained model and reset final fully connected layer.
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 101)

# Send the model to GPU
model = model.to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train and evaluate
model = train_model(model, criterion, optimizer,
                    step_lr_scheduler, num_epochs=50)

# Load a pretrained model and reset final fully connected layer.
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 101)

# Send the model to GPU
model_conv = model_conv.to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()


optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# Validate the model
model_conv = validate_model(model_conv, criterion, optimizer_conv,
                            exp_lr_scheduler, num_epochs=50)
