"""PyTorch CIFAR-10 image classification.

The code is generally adapted from 'PyTorch: A 60 Minute Blitz'. Further
explanations are given in the official PyTorch tutorial:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""


# mypy: ignore-errors
# pylint: disable=W0223


from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torchvision.datasets import CIFAR10
from torchvision import models
from data_loader import createDataLoaders
import pickle
from dotenv import load_dotenv
import os

load_dotenv()

DATA_ROOT = "./dataset"

data_folder = os.environ['data_dir']
images_pickle = os.path.join(data_folder, 'image_data.pickle')
labels_pickle = os.path.join(data_folder, 'label_data.pickle')
with open(images_pickle, 'rb') as image_pickle_file:
    all_images = pickle.load(image_pickle_file)
with open(labels_pickle, 'rb') as label_pickle_file:
    all_labels = pickle.load(label_pickle_file)



# pylint: disable=unsubscriptable-object
# class Net(nn.Module):
#     """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.bn1 = nn.BatchNorm2d(6)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.bn2 = nn.BatchNorm2d(16)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.bn3 = nn.BatchNorm1d(120)
#         self.fc2 = nn.Linear(120, 84)
#         self.bn4 = nn.BatchNorm1d(84)
#         self.fc3 = nn.Linear(84, 10)

#     # pylint: disable=arguments-differ,invalid-name
#     def forward(self, x: Tensor) -> Tensor:
#         """Compute forward pass."""
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.bn3(self.fc1(x)))
#         x = F.relu(self.bn4(self.fc2(x)))
#         x = self.fc3(x)
#         return x

class CustomDenseNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomDenseNet, self).__init__()
        self.densenet = models.densenet121(pretrained=False)
        self.densenet.features.conv0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False) # To adjust input size

        
        # Modify the classifier for the specified number of output classes
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.densenet(x)

# def load_data() -> (
#     Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]
# ):
#     """Load CIFAR-10 (training and test set)."""
#     transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )
#     trainset = CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
#     testset = CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
#     num_examples = {"trainset": len(trainset), "testset": len(testset)}
#     return trainloader, testloader, num_examples

# def load_custom_data() -> (
#     Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]
# ):
#     """Load CBIS-DDSM dataset."""
#     transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(degrees=45),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#     trainset = CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
#     testset = CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
#     num_examples = {"trainset": len(trainset), "testset": len(testset)}
#     return trainloader, testloader, num_examples


def train(
    net: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    # Train the network
    net.to(device)
    net.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


def test(
    net: nn.Module,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    # Define loss and metrics
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    # Evaluate the network
    net.to(device)
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def main():
    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, testloader = createDataLoaders(all_images, all_labels, 0.7, 0.3, 32)
    net = CustomDenseNet(num_classes=2).to(DEVICE)
    net.eval()
    print("Start training")
    train(net=net, trainloader=trainloader, epochs=2, device=DEVICE)
    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


if __name__ == "__main__":
    main()
