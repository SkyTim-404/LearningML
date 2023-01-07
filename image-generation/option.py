import torch
from torch.optim import Adam, RMSprop
from dataset import FashionMNIST
from model.discriminator import Discriminator
from model.generator import Generator
from model.critic import Critic
from loss import DiscriminatorLoss, GeneratorLoss, WassersteinCriticLoss, WassersteinGeneratorLoss

class Option():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.learning_rate = 5e-5
        self.latent_dim = 128 # 64, 128, 256
        self.batch_size = 64
        self.num_epochs = 45
        self.weight_decay = 1e-5
        self.weight_clip = 0.01
        self.critic_iterations = 5
        self.dataset = FashionMNIST(batch_size=self.batch_size)
        
        self.load_model = True
        self.discriminator_filename = "image-generation/weights/fashion_mnist_discriminator.pth"
        self.generator_filename = "image-generation/weights/fashion_mnist_generator.pth"
        self.critic_filename = "image-generation/weights/fashion_mnist_critic.pth"
        self.discriminator = self.get_discriminator()
        self.generator = self.get_generator()
        self.critic = self.get_critic()
        self.discriminator_loss_fn = DiscriminatorLoss()
        self.generator_loss_fn = WassersteinGeneratorLoss()
        self.critic_loss_fn = WassersteinCriticLoss()
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.generator_optimizer = RMSprop(self.generator.parameters(), lr=self.learning_rate)
        self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.learning_rate)
        
        
    def get_discriminator(self):
        if self.load_model:
            return torch.load(self.discriminator_filename)
        return Discriminator().to(self.device)
    
    def get_generator(self):
        if self.load_model:
            return torch.load(self.generator_filename)
        return Generator(self.latent_dim).to(self.device)
    
    def get_critic(self):
        if self.load_model:
            return torch.load(self.critic_filename)
        return Critic().to(self.device)
        