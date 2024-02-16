import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils.yaml_parser import YamlParser
from utils.loss import NLSLoss
from dataset import PascalVOCDataset

def train(hyperparameters):
    transform = A.Compose([
        A.Resize(height=160, width=256), 
        A.Rotate(limit=35, p=0.5), 
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.5), 
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255), 
        ToTensorV2(transpose_mask=True), 
    ])
    train_dataset = PascalVOCDataset(root="./data", image_set="train", download=False, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=hyperparameters.batch_size, shuffle=True)
    val_dataset = PascalVOCDataset(root="./data", image_set="val", download=False)
    val_dataloader = DataLoader(val_dataset, batch_size=hyperparameters.batch_size, shuffle=False)
    if hyperparameters.load_model:
        model = torch.load(hyperparameters.model_dir)
    else:
        model = smp.Unet('resnet34', in_channels=hyperparameters.in_channels, classes=hyperparameters.classes)
    optimizer = AdamW(model.parameters(), lr=hyperparameters.learning_rate)
    loss_fn = NLSLoss(dim=1, reduction="mean")
    metrics = [smp_utils.metrics.IoU()]
    train_epoch = smp_utils.train.TrainEpoch(model, 
                                            loss=loss_fn, 
                                            metrics=metrics, 
                                            optimizer=optimizer, 
                                            device=hyperparameters.device, 
                                            verbose=True)
    val_epoch = smp_utils.train.ValidEpoch(model, 
                                        loss=loss_fn, 
                                        metrics=metrics, 
                                        device=hyperparameters.device, 
                                        verbose=True)
    for i in range(hyperparameters.num_epochs):
        print(f"\nEpoch: {i}")
        train_epoch.run(train_dataloader)
        val_epoch.run(val_dataloader)
        torch.save(model, hyperparameters.model_dir)
    

def main():
    hyperparameters = YamlParser("configs/hyperparameters.yaml")
    train(hyperparameters)

if __name__ == "__main__":
    main()