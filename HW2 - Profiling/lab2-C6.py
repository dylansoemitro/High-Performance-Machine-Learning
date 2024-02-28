import torch
import time
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        #conv block 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        #conv block 1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        #basic blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

model = ResNet18()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()


transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Random cropping
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flipping
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),  # Normalize RGB channels
])

# Load the CIFAR10 dataset
cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_loader = DataLoader(cifar10_dataset, batch_size=128, shuffle=True, num_workers=2)


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training with Different Optimizers')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--data-path', default='./data', type=str, help='Path to dataset')
    parser.add_argument('--num-workers', default=12, type=int, help='Number of data loader workers')
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss()

    optimizer_configs = {
        'SGD': lambda model: optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4),
        'SGD_nesterov': lambda model: optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True),
        'Adagrad': lambda model: optim.Adagrad(model.parameters(), lr=0.1, weight_decay=5e-4),
        'Adadelta': lambda model: optim.Adadelta(model.parameters(), lr=0.1, weight_decay=5e-4),
        'Adam': lambda model: optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    }

    for name, optimizer_fn in optimizer_configs.items():
        print(f"\nTraining with {name}")
        model = ResNet18().to(device)  # Reset model
        optimizer = optimizer_fn(model)  # Get optimizer
        total_time = 0
        model.train()

        for epoch in range(1, 6):
            epoch_start_time = time.time()
            running_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            epoch_end_time = time.time()
            total_time += epoch_end_time - epoch_start_time
            print(f'Epoch: {epoch+1}, Loss: {running_loss / len(train_loader)}, Accuracy: {100.0 * correct / total}%')
        average_time = total_time / 5
        loss = running_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        print(f"Optimizer: {name}, Average Time: {average_time:.3f}s, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
if __name__ == '__main__':
    main()