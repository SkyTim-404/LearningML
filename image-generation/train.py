import torch
from option import Option

def train(opt: Option):
    train_dataloader = opt.dataset.get_train_dataloader()
    for e in range(opt.num_epochs):
        for batch_idx, (real_img, _) in enumerate(train_dataloader):
            real_img = real_img.to(opt.device)
            # Train critic
            for _ in range(opt.critic_iterations):
                latent = torch.randn((opt.batch_size, opt.latent_dim)).to(opt.device)
                fake_img = opt.generator(latent)
                predict_real = opt.critic(real_img)
                predict_fake = opt.critic(fake_img)
                
                critic_loss = opt.critic_loss_fn(predict_real, predict_fake)
                opt.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                opt.critic_optimizer.step()
                for p in opt.critic.parameters():
                    p.data.clamp_(-opt.weight_clip, opt.weight_clip)
                print(f"epoch: {e}, {batch_idx}/{len(train_dataloader)} loss: {critic_loss.item()} ########################", end="\r")
            # Train generator
            predict_fake = opt.critic(fake_img)
            generator_loss = opt.generator_loss_fn(predict_fake)
            opt.generator_optimizer.zero_grad()
            generator_loss.backward()
            opt.generator_optimizer.step()
            
    torch.save(opt.critic, opt.critic_filename)
    torch.save(opt.generator, opt.generator_filename)
            

if __name__ == "__main__":
    opt = Option()
    train(opt)