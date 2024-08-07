from torchvision import datasets, transforms
from abc import ABC
    

class BaseDataset(ABC):
    def __init__(self):
        self.train_dataset = None
        self.test_dataset = None
        
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_test_dataset(self):
        return self.test_dataset
    

class FashionMNIST(BaseDataset):
    def __init__(
        self, 
        dir="./data/", 
        transforms=transforms.Compose([transforms.ToTensor()])
    ):
        super().__init__()
        self.train_dataset = datasets.FashionMNIST(root=dir, train=True, download=True, transform=transforms)
        self.test_dataset = datasets.FashionMNIST(root=dir, train=False, download=True, transform=transforms)
    
    
class DigitMNIST(BaseDataset):
    def __init__(
        self, 
        dir="./data/", 
        transforms=transforms.Compose([transforms.ToTensor()])
    ):
        super().__init__()
        self.train_dataset = datasets.MNIST(root=dir, train=True, download=True, transform=transforms)
        self.test_dataset = datasets.MNIST(root=dir, train=False, download=True, transform=transforms)


if __name__ == '__main__':
    pass
    