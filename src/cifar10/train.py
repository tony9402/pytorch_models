import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class TrainCifar10():
    def __init__(self, config:[dict]):

        self.device = config.get('device')

        if self.device == None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = torch.device(self.device)

        # Load Model
        if 'model' not in config:
            raise "Not Found Model"

        model = config['model']
        self.model  = model.to(self.device)
        self.num_epochs = config.get('epoch', 10)
        self.learning_rate = config.get('learning_rate', 0.001)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        batch_size = config.get('batch_size', 100)
        self._load_dataset(batch_size)

        print(f"Model : {self.model.__class__.__name__}")
        print(f"Epoch : {self.num_epochs}")
        print(f"Learning Rate : {self.learning_rate}")
        print(f"Batch Size : {batch_size}")
        print(f"device : {self.device}")

    def _load_dataset(self, batch_size):
        transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor()
        ])

        train_dataset = torchvision.datasets.CIFAR10(root='./../../data/',
                                                     train=True,
                                                     transform=transform,
                                                     download=True)

        test_dataset = torchvision.datasets.CIFAR10(root='./../../data/',
                                                    train=False,
                                                    transform=transforms.ToTensor())

        self.train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

        self.test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=batch_size,
                                     shuffle=True)

    def train(self):
        total_step = len(self.train_loader)
        curr_lr = self.learning_rate

        for epoch in range(self.num_epochs):
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for i, (images, labels) in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix({
                    "Epoch": f"{epoch + 1}/{self.num_epochs}",
                    "Step": f"{i + 1}/{total_step}",
                    "Loss": f"{loss.item():.4f}"
                })

                if (epoch + 1) % 20 == 0:
                    curr_lr /= 3
                    update_lr(self.optimizer, curr_lr)
            pbar.close()

            self.model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                pbar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
                current_accuracy = 0
                for i, (images, labels) in pbar:
                    current_accuracy = current_accuracy * total
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    current_correct = (predicted == labels).sum().item()
                    current_accuracy = (current_accuracy + current_correct) / total
                    correct += current_correct
                    pbar.set_postfix({ "Accuracy": f"{100*current_accuracy:.2f}%" })

                print(f'Accuracy : {100 * correct / total}')
