import torch
import sys
from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop, AdamW
from torch.nn import BCELoss
from torchvision import transforms

sys.path.append(r"../../")

from option import Option
from dataset import FashionMNIST
from models import Discriminator, Generator, Critic
from loss import DiscriminatorLoss, GeneratorLoss

def main(opt: Option):
    dataset = FashionMNIST(dir=opt.data_path, transforms=transforms.Compose([
        transforms.ToTensor(), 
    ]))
    train_dataloader = DataLoader(dataset.get_train_dataset(), batch_size=opt.batch_size, shuffle=True, drop_last=True)
    
    # Initialize models and optimizers and loss functions
    generator = Generator(opt.latent_dim).to(opt.device)
    discriminator = Discriminator().to(opt.device)
    
    loss = BCELoss()
    
    generator_optimizer = AdamW(generator.parameters(), lr=opt.generator_learning_rate, weight_decay=opt.generator_weight_decay)
    discriminator_optimizer = AdamW(discriminator.parameters(), lr=opt.disciminator_learning_rate, weight_decay=opt.discriminator_weight_decay)
    
    # Load models if load_model is True
    if opt.load_model:
        generator.load_state_dict(torch.load(opt.generator_path))
        discriminator.load_state_dict(torch.load(opt.discriminator_path))
    
    for e in range(opt.num_epochs):
        generator_losses = []
        discriminator_losses = []
        for batch_idx, (real_imgs, _) in enumerate(train_dataloader):
            real_imgs = real_imgs.to(opt.device)
            
            # Train discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()
            y_hat_real = discriminator(real_imgs)
            y_real = torch.ones(opt.batch_size, 1).to(opt.device)
            discriminator_real_loss = loss(y_hat_real, y_real)
            
            latent = torch.randn((opt.batch_size, opt.latent_dim)).to(opt.device)
            fake_imgs = generator(latent)
            y_hat_fake = discriminator(fake_imgs.detach())
            y_fake = torch.zeros(opt.batch_size, 1).to(opt.device)
            discriminator_fake_loss = loss(y_hat_fake, y_fake)
            
            discriminator_loss = (discriminator_real_loss + discriminator_fake_loss) / 2
            discriminator_loss.backward()
            discriminator_optimizer.step()
            
            # Train generator: maximize log(D(G(z)))
            latent = torch.randn((opt.batch_size, opt.latent_dim)).to(opt.device)
            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()
            fake_imgs = generator(latent)
            y_hat = discriminator(fake_imgs)
            y = torch.ones(opt.batch_size, 1).to(opt.device) # the best we can get is 1
            
            generator_loss = loss(y_hat, y)
            generator_loss.backward()
            generator_optimizer.step()
            
            generator_losses.append(generator_loss.item())
            discriminator_losses.append(discriminator_loss.item())
            print(f"epoch: {e+1}, {batch_idx}/{len(train_dataloader)}, generator loss: {generator_loss.item():.2f}, discriminator loss: {discriminator_loss.item():.2f}", end="\r")
        
        print("================================================================================================")
        print(f"epoch: {e+1}, generator loss: {sum(generator_losses)/len(generator_losses):.2f}, discriminator loss: {sum(discriminator_losses)/len(discriminator_losses):.2f}")
            
        if (e+1) % opt.save_after_epochs == 0:
            torch.save(generator.state_dict(), opt.generator_path)
            torch.save(discriminator.state_dict(), opt.discriminator_path)
            print("Models saved")
            
    torch.save(generator.state_dict(), opt.generator_path)
    torch.save(discriminator.state_dict(), opt.discriminator_path)
            

if __name__ == "__main__":
    opt = Option()
    main(opt)