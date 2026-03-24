"""
Скрипт для запуска обучения 3D U-Net модели.
Запустить: python train_model.py
"""

import os
import torch
from app.models import UNet3D
from app.models.dataset import create_dataloaders
from app.models.unet3d import UNet3DConfiguration


def train():
    # Конфигурация
    DATA_DIR = "Task03_Liver_rs"
    CHECKPOINT_DIR = "checkpoints"
    NUM_EPOCHS = 30
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    TARGET_SIZE = (32, 64, 64)
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")
    
    # DataLoaders
    train_loader, val_loader = create_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        target_size=TARGET_SIZE,
        augment_train=True
    )
    
    # Модель
    config = UNet3DConfiguration(
        in_channels=1,
        out_channels=2,
        base_filters=32,
        depth=4,
        dropout_rate=0.2
    )
    model = UNet3D(config).to(device)
    print(f"Параметров: {model.get_num_parameters():,}")
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_dice = 0.0
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].squeeze(1).long().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                
                # Dice score
                intersection = (preds * (labels == 1)).sum()
                dice = (2. * intersection) / (preds.sum() + (labels == 1).sum() + 1e-8)
                val_dice += dice.item()
        
        val_dice /= len(val_loader)
        
        print(f"Эпоха {epoch+1}/{NUM_EPOCHS} | Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        # Save best
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"  -> Лучшая модель сохранена! Dice: {best_dice:.4f}")
    
    print(f"\nОбучение завершено! Лучший Dice: {best_dice:.4f}")


if __name__ == "__main__":
    train()
