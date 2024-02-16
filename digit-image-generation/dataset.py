from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from abc import ABC

def initialize_datasets():
    FashionMNIST.initialize()
    DigitMNIST.initialize()
    

class BaseDataset(ABC):
    def __init__(self, batch_size, transforms):
        self.batch_size = batch_size
        self.transforms = transforms
        self.train_dataset = None
        self.test_dataset = None
        
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_test_dataset(self):
        return self.test_dataset
    
    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
    

class FashionMNIST(BaseDataset):
    def __init__(self, batch_size=64, transforms=transforms.Compose([transforms.ToTensor()])):
        super().__init__(batch_size=batch_size, transforms=transforms)
        self.train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=self.transforms)
        self.test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=self.transforms)
    
    @classmethod
    def initialize(cls):
        datasets.FashionMNIST(root="./data", train=True, download=True)
        datasets.FashionMNIST(root="./data", train=False, download=True)
    
    
class DigitMNIST(BaseDataset):
    def __init__(self, batch_size=64, transforms=transforms.Compose([transforms.ToTensor()])):
        super().__init__(batch_size=batch_size, transforms=transforms)
        self.train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=self.transforms)
        self.test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=self.transforms)
    
    @classmethod
    def initialize(cls):
        datasets.MNIST(root="./data", train=True, download=True)
        datasets.MNIST(root="./data", train=False, download=True)


if __name__ == '__main__':
    initialize_datasets()
    