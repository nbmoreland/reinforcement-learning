# Nicholas Moreland
# 05/02/2024

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from helper import AverageMeter, accuracy

# Define the Food101 class


class Food101(nn.Module):
    def __init__(self, dropout=True, num_classes=101):
        super(Food101, self).__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(3, 6, 5, padding=1)
        self.conv3 = nn.Conv2d(6, 16, 5, padding=1)
        self.class_conv = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        if self.dropout:
            x = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x))
        conv3_out = F.relu(self.conv3(conv1_out))
        class_out = F.relu(self.class_conv(conv3_out))
        pool_out = class_out.reshape(
            class_out.size(0), class_out.size(1), -1).mean(-1)
        return pool_out


# Define the model parameters
batch_size = 100
learning_rate = 1e-2
epochs = 50
print_frequency = 50

# Set device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = Food101()
print(sum(dict((p.data_ptr(), p.numel())
      for p in model.parameters()).values()))

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# Define the data path
data_path = "./food101data"

# Load the data
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Transform the training data
training_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor()])

# Load the dataset
dataset = torchvision.datasets.Food101(
    data_path, transform=training_transform, download=True)

# Get the size of the dataset
dataset_size = len(dataset)
training_size = int(dataset_size * .95)
validation_size = dataset_size - training_size

# Split the dataset into training and validation sets
training_dataset, validation_dataset = torch.utils.data.random_split(
    dataset, [training_size, validation_size])

# Load the training and validation data
training_data = torch.utils.data.DataLoader(
    dataset=training_dataset, batch_size=batch_size, shuffle=True)

# Load the validation data
validation_data = torch.utils.data.DataLoader(
    validation_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

# Load the testing data
test_data = torch.utils.data.DataLoader(
    torchvision.datasets.Food101(data_path, transform=training_transform),
    batch_size=batch_size, shuffle=False, pin_memory=False)

# Set up the logger
logger = SummaryWriter("runs/convolution_net")

# Initialize the model
total_steps = len(training_data)

# Define running loss and running correct
running_loss = 0.0
running_correct = 0

# Training Phase
temp = 0
for epoch in range(epochs):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    pbar = tqdm(enumerate(training_data), total=len(training_data))

    # Loop through the training data
    for i, (input, target) in pbar:

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        prec = accuracy(output.data, target)[0]

        running_loss += loss.item()
        _, predictions = torch.max(output, 1)
        running_correct += (predictions == target).sum().item()

        losses.update(loss.item(), input.shape[0])
        top1.update(prec.item(), input.shape[0])

        print("epoch ", epoch, " i ", i, " loss ", losses.avg)

        logger.add_scalar('Training Loss', losses.avg, epoch*total_steps + i)
        logger.add_scalar('Training Accuracy', top1.avg,
                          epoch*total_steps + i)
        temp += 1
        if temp == 50:
            break
    break
print("Accuracy of training is ", running_correct / 100)

# Define the running loss and running correct
running_loss = 0.0
running_correct = 0

# Validation Phase
print("Validation Phase")
temp = 0
for epoch in range(epochs):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    pbar = tqdm(enumerate(validation_data), total=len(validation_data))

    # Loop through the validation data
    for i, (input, target) in pbar:

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        prec = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.shape[0])
        top1.update(prec.item(), input.shape[0])

        running_loss += loss.item()
        _, predictions = torch.max(output, 1)
        running_correct += (predictions == target).sum().item()

        print(
            f'epoch {epoch+1} / {epochs}, step {i+1}/{total_steps}, loss=  {loss.item():.4f}')
        logger.add_scalar('Validation Loss', loss.item(),
                          epoch*total_steps + i)
        logger.add_scalar('Validation Accuracy', top1.avg,
                          epoch*total_steps + i)
        temp += 1
        if temp == 100:
            break
    break

print('Finished Training and Validation Phase')

# Save the model
torch.save(model.state_dict(), './all_convolution_net.pth')

print("Testing Phase")
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(101)]
    n_class_samples = [0 for i in range(101)]

    # Loop through the testing data
    for i, (images, labels) in enumerate(test_data):
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        print("", i, " accuracy ", 100.0 * n_correct / n_samples)

        for j in range(batch_size):
            label = labels[j]
            pred = predicted[j]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    # Compute the accuracy of the network
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    # Compute the accuracy of each class
    for i in range(101):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of class {i}: {acc} %')
