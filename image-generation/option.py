import torch
from torch.optim import Adam
from dataset import FashionMNIST
from model.discriminator import Discriminator
from model.generator import Generator
from loss import DiscriminatorLoss, GeneratorLoss

class Option():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.learning_rate = 3e-4
        self.latent_dim = 64 # 128, 256
        self.batch_size = 32
        self.num_epochs = 50
        self.weight_decay = 1e-5
        self.load_model = False
        self.discriminator_filename = "image-generation/weights/fashion_mnist_discriminator.pth"
        self.generator_filename = "image-generation/weights/fashion_mnist_generator.pth"
        self.dataset = FashionMNIST(batch_size=self.batch_size)
        self.discriminator = self.get_discriminator()
        self.generator = self.get_generator()
        self.discriminator_loss_fn = DiscriminatorLoss()
        self.generator_loss_fn = GeneratorLoss()
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=self.learning_rate)
        self.generator_optimizer = Adam(self.generator.parameters(), lr=self.learning_rate)
        
        
    def get_discriminator(self):
        if self.load_model:
            return torch.load(self.discriminator_filename)
        return Discriminator().to(self.device)
    
    def get_generator(self):
        if self.load_model:
            return torch.load(self.generator_filename)
        return Generator(self.latent_dim).to(self.device)
        