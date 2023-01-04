import torch
from option import Option

def train(opt: Option):
    train_dataloader = opt.dataset.get_train_dataloader()
    for e in range(opt.num_epochs):
        for batch_idx, (real_img, _) in enumerate(train_dataloader):
            print(f"epoch: {e}, {batch_idx}/{len(train_dataloader)}", end="\r")
            latent = torch.randn((opt.batch_size, opt.latent_dim)).to(opt.device)
            real_img = real_img.to(opt.device)
            fake_img = opt.generator(latent)
            predict_real = opt.discriminator(real_img)
            predict_fake = opt.discriminator(fake_img)
            
            generator_loss = opt.generator_loss_fn(predict_fake)
            discriminator_loss = opt.discriminator_loss_fn(predict_real, predict_fake)
            opt.generator_optimizer.zero_grad()
            opt.discriminator_optimizer.zero_grad()
            generator_loss.backward(retain_graph=True)
            discriminator_loss.backward()
            opt.generator_optimizer.step()
            opt.discriminator_optimizer.step()
            
    torch.save(opt.discriminator, opt.discriminator_filename)
    torch.save(opt.generator, opt.generator_filename)
            

if __name__ == "__main__":
    opt = Option()
    train(opt)