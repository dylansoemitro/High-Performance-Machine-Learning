import torch
import time
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt


def measure_dataloader_time(num_workers, data_path):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers)
    start_time = time.perf_counter()
    for _, (inputs, _) in enumerate(train_loader, 0):
        pass # no processing
    end_time = time.perf_counter()
    return end_time - start_time

def main():
    parser = argparse.ArgumentParser(description="Measure DataLoader Time with Different Number of Workers")
    parser.add_argument('--data-path', default='./data', type=str, help='Path to dataset')
    args = parser.parse_args()

    num_workers_list = [0, 4, 8, 12, 16]
    times = []
    for num_workers in num_workers_list:
        time_spent = measure_dataloader_time(num_workers, args.data_path)
        print(f'Num Workers: {num_workers}, Data-loading Time: {time_spent:.3f}s')
        times.append(time_spent)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(num_workers_list, times, marker='o')
    plt.title('Data-loading Time vs. Number of Workers')
    plt.xlabel('Number of Workers')
    plt.ylabel('Data-loading Time (s)')
    plt.grid(True)
    plt.show()

    optimal_workers = num_workers_list[times.index(min(times))]
    print(f'Optimal number of workers for best runtime performance: {optimal_workers}')

if __name__ == '__main__':
    main()