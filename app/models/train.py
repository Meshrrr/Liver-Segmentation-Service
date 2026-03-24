"""
Обучение 3D U-Net для сегментации печени.

Функционал:
- Тренировочный цикл с валидацией
- Логирование в TensorBoard и консоль
- Метрики: Dice Score, IoU, Loss
- Сохранение лучшей модели
- Продолжение обучения с чекпоинта
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from app.models.unet3d import UNet3D, create_unet3d, UNet3DConfiguration
from app.models.dataset import create_dataloaders, LiverDataset


class DiceLoss(nn.Module):
    """
    Dice Loss для бинарной сегментации.
    Хорошо работает при несбалансированных классах.
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Предсказание формы (B, C, D, H, W) - logits
            target: Маска формы (B, D, H, W) - long
        """
        # Получаем предсказание класса 1 (печень)
        pred = torch.softmax(pred, dim=1)[:, 1, :, :, :]
        
        # flatten
        pred = pred.view(-1)
        target = target.view(-1).float()
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Комбинированный лосс: Dice + CrossEntropy
    """
    
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.dice_weight * dice + self.ce_weight * ce


def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Вычисляет Dice Score для бинарной сегментации.
    
    Args:
        pred: Предсказание (B, C, D, H, W) - after softmax
        target: Маска (B, D, H, W) - long
        
    Returns:
        Dice Score в диапазоне [0, 1]
    """
    pred_class = torch.argmax(pred, dim=1)  # (B, D, H, W)
    
    # Для класса 1 (печень)
    pred_positive = (pred_class == 1).float()
    target_positive = (target == 1).float()
    
    intersection = (pred_positive * target_positive).sum()
    union = pred_positive.sum() + target_positive.sum()
    
    if union == 0:
        return 1.0  # Оба пустые - идеально
    
    dice = (2. * intersection) / union
    
    return dice.item()


def calculate_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Вычисляет IoU (Intersection over Union)."""
    pred_class = torch.argmax(pred, dim=1)
    
    pred_positive = (pred_class == 1).float()
    target_positive = (target == 1).float()
    
    intersection = (pred_positive * target_positive).sum()
    union = pred_positive.sum() + target_positive.sum() - intersection
    
    if union == 0:
        return 1.0
    
    iou = intersection / union
    
    return iou.item()


class Trainer:
    """
    Класс для обучения модели сегментации.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        loss_type: str = "combined",  # "dice", "ce", "combined"
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "runs"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss
        if loss_type == "dice":
            self.criterion = DiceLoss()
        elif loss_type == "ce":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = CombinedLoss(dice_weight=0.5, ce_weight=0.5)
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Директории
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir / datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        # Лучшие метрики
        self.best_dice = 0.0
        self.best_epoch = 0
        
        # История
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_dice": [],
            "val_dice": [],
            "val_iou": []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """Один эпох обучения."""
        self.model.train()
        
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Loss
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Метрики
            dice = calculate_dice_score(outputs, labels)
            
            total_loss += loss.item()
            total_dice += dice
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice:.4f}"})
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return avg_loss, avg_dice
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """Валидация на тестовом наборе."""
        self.model.eval()
        
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward
            outputs = self.model(images)
            
            # Loss
            loss = self.criterion(outputs, labels)
            
            # Метрики
            dice = calculate_dice_score(outputs, labels)
            iou = calculate_iou(outputs, labels)
            
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_iou = total_iou / num_batches
        
        return avg_loss, avg_dice, avg_iou
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Сохранение чекпоинта."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_dice": self.best_dice,
            "history": self.history
        }
        
        # Сохраняем последний чекпоинт
        torch.save(checkpoint, self.checkpoint_dir / "last_checkpoint.pth")
        
        # Сохраняем лучший
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pth")
            print(f"✓ Лучшая модель сохранена! Dice: {self.best_dice:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Загрузка чекпоинта."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_dice = checkpoint.get("best_dice", 0.0)
        self.history = checkpoint.get("history", self.history)
        
        print(f"✓ Загружен чекпоинт: эпоха {checkpoint['epoch']}, лучший Dice: {self.best_dice:.4f}")
        
        return checkpoint["epoch"]
    
    def train(
        self, 
        num_epochs: int, 
        early_stopping_patience: int = 15,
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Основной цикл обучения.
        
        Args:
            num_epochs: Количество эпох
            early_stopping_patience: Остановка если нет улучшений
            resume_from: Путь к чекпоинту для продолжения
            
        Returns:
            История обучения
        """
        start_epoch = 0
        
        # Загрузка чекпоинта
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            start_epoch += 1
        
        patience_counter = 0
        
        print(f"\n{'='*50}")
        print(f"Начало обучения: {num_epochs} эпох")
        print(f"Устройство: {self.device}")
        print(f"Параметров модели: {self.model.get_num_parameters():,}")
        print(f"{'='*50}\n")
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\nЭпоха {epoch + 1}/{num_epochs}")
            print("-" * 30)
            
            # Обучение
            train_loss, train_dice = self.train_epoch()
            
            # Валидация
            val_loss, val_dice, val_iou = self.validate()
            
            # Learning rate scheduler
            self.scheduler.step(val_dice)
            
            # Логирование
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Dice/train", train_dice, epoch)
            self.writer.add_scalar("Dice/val", val_dice, epoch)
            self.writer.add_scalar("IoU/val", val_iou, epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]['lr'], epoch)
            
            # Сохранение в историю
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_dice"].append(train_dice)
            self.history["val_dice"].append(val_dice)
            self.history["val_iou"].append(val_iou)
            
            print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
            
            # Сохранение лучшей модели
            is_best = val_dice > self.best_dice
            
            if is_best:
                self.best_dice = val_dice
                self.best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Сохраняем чекпоинт
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Ранняя остановка на эпохе {epoch + 1}")
                print(f"Лучший Dice: {self.best_dice:.4f} на эпохе {self.best_epoch}")
                break
        
        # Финальное логирование
        print(f"\n{'='*50}")
        print("Обучение завершено!")
        print(f"Лучший Dice: {self.best_dice:.4f} на эпохе {self.best_epoch}")
        print(f"{'='*50}")
        
        # Сохраняем историю
        with open(self.checkpoint_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        self.writer.close()
        
        return self.history


def train_model(
    data_dir: str = "Task03_Liver_rs",
    num_epochs: int = 50,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    target_size: Tuple[int, int, int] = (64, 128, 128),
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
    resume_from: Optional[str] = None
) -> Trainer:
    """
    Основная функция для запуска обучения.
    
    Args:
        data_dir: Путь к датасету
        num_epochs: Количество эпох
        batch_size: Размер батча
        learning_rate: Скорость обучения
        target_size: Размер входных томов
        device: cuda или cpu
        checkpoint_dir: Директория для сохранения
        resume_from: Чекпоинт для продолжения
        
    Returns:
        Обученный Trainer
    """
    # Создание модели
    config = UNet3DConfiguration(
        in_channels=1,
        out_channels=2,
        base_filters=32,
        depth=4,
        dropout_rate=0.2
    )
    model = UNet3D(config)
    
    print(f"Создана модель: {model.get_num_parameters():,} параметров")
    
    # Даталоадеры
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        target_size=target_size,
        augment_train=True
    )
    
    # Создание тренера
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir
    )
    
    # Обучение
    trainer.train(num_epochs=num_epochs, resume_from=resume_from)
    
    return trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Обучение 3D U-Net для сегментации печени")
    parser.add_argument("--data_dir", type=str, default="Task03_Liver_rs", help="Путь к датасету")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох")
    parser.add_argument("--batch_size", type=int, default=2, help="Размер батча")
    parser.add_argument("--lr", type=float, default=1e-4, help="Скорость обучения")
    parser.add_argument("--device", type=str, default="cuda", help="Устройство (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Чекпоинт для продолжения")
    
    args = parser.parse_args()
    
    # Определение устройства
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA недоступна, используем CPU")
        args.device = "cpu"
    
    # Запуск обучения
    train_model(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        resume_from=args.resume
    )
