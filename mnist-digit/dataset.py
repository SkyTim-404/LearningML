import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class Dataset:
    transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 1.0)])
    
    @classmethod
    def get_train_dataset(cls, dir="./data/"):
        trainset = datasets.MNIST(root=dir, train=True, download=True, transform=cls.transforms)
        return trainset

    @classmethod
    def get_test_dataset(cls, dir):
        testset = datasets.MNIST(root=dir, train=False, download=True, transform=cls.transforms)
        return testset
    
def test():
    train = Dataset.get_train_dataset()

    (img, label) = train[0]
    img = img.squeeze()

    plt.imshow(img, cmap='gray')
    plt.show()
    
if __name__ == '__main__':
    test()