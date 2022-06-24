import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def get_train_dataset():
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    return trainset

def get_test_dataset():
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    return testset
    
def test():
    train = get_train_dataset()
    train = DataLoader(train, shuffle=True)

    img, label = next(iter(train))

    # print(label)
    # print(type(img))
    # print(img.size)

    plt.imshow(img, cmap='gray')
    plt.show()
    
if __name__ == '__main__':
    test()