import torch
import matplotlib.pyplot as plt
from option import Option

def main():
    with torch.no_grad():
        opt = Option()
        latent = torch.randn((1, opt.latent_dim)).to(opt.device)
        img = torch.squeeze(opt.generator(latent)).to("cpu")
        plt.imshow(img, cmap="gray")
        plt.show()
    

if __name__ == '__main__':
    main()