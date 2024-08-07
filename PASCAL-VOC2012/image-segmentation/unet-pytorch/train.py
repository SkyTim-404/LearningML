import sys
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

sys.path.append(r"../")

from utils.yaml_parser import YamlParser
from dataset import PascalVOCDataset

def train(hyperparameters, train_dataloader, val_dataloader, model):
    # Training initialization
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = AdamW(model.parameters(), lr=hyperparameters.learning_rate, weight_decay=hyperparameters.weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=hyperparameters.step_size, gamma=hyperparameters.gamma)
    
    for e in range(hyperparameters.num_epochs):
        train_losses = []
        val_losses = []
        for i, (images, masks) in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            images = images.to(hyperparameters.device)
            masks = masks.to(hyperparameters.device)
            masks = torch.argmax(masks, axis=1)
            pred = model(images)
            train_loss = loss_fn(pred, masks)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())
            
            print(f"Epoch {e+1}/{hyperparameters.num_epochs}, Train Batch {i+1}/{len(train_dataloader)}, Train Loss: {train_loss.item():.4f}", end="\r")
            
        model.eval()
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_dataloader):
                images = images.to(hyperparameters.device)
                masks = masks.to(hyperparameters.device)
                masks = torch.argmax(masks, axis=1)
                pred = model(images)
                val_loss = loss_fn(pred, masks)
                val_losses.append(val_loss.item())
                print(f"Epoch {e+1}/{hyperparameters.num_epochs}, Val Batch {i+1}/{len(train_dataloader)}, Val Loss: {val_loss.item():.4f}", end="\r")
        
        lr_scheduler.step()
        
        print(f"===== Epoch {e+1}/{hyperparameters.num_epochs}, Train Loss: {sum(train_losses)/len(train_losses):.4f}, Val Loss: {sum(val_losses)/len(val_losses):.4f} =====")
        
        if (e+1) % hyperparameters.save_interval == 0:
            torch.save(model.state_dict(), hyperparameters.model_dir)
    
    torch.save(model.state_dict(), hyperparameters.model_dir)
    

def main(hyperparameters):
    # Data initialization
    transform = A.Compose([
        A.Resize(height=160, width=256), 
        A.Rotate(limit=35, p=0.5), 
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.5), 
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255), 
        ToTensorV2(transpose_mask=True), 
    ])
    train_dataset = PascalVOCDataset(root=r"../../data", image_set="train", download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=hyperparameters.batch_size, shuffle=True, drop_last=True)
    val_dataset = PascalVOCDataset(root=r"../../data", image_set="val", download=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hyperparameters.batch_size, shuffle=False, drop_last=True)
    
    # Model initialization
    model = smp.Unet('resnet34', in_channels=hyperparameters.in_channels, classes=hyperparameters.classes).to(hyperparameters.device)
    if hyperparameters.load_model:
        model.load_state_dict(torch.load(hyperparameters.model_dir))
    
    train(hyperparameters, train_dataloader, val_dataloader, model)

if __name__ == "__main__":
    hyperparameters = YamlParser("configs/hyperparameters.yaml")
    main(hyperparameters)