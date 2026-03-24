"""
Инференс: применение обученной 3D U-Net модели для сегментации печени.

Функционал:
- Загрузка весов модели
- Предобработка входных данных
- Предсказание маски сегментации
- Постобработка результатов
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from scipy.ndimage import zoom

from app.models.unet3d import UNet3D, UNet3DConfiguration


class LiverSegmentationModel:
    """
    Класс для инференса модели сегментации печени.
    
    Загружает обученную модель и применяет её к новым данным.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        config: Optional[UNet3DConfiguration] = None
    ):
        """
        Args:
            checkpoint_path: Путь к файлу модели (.pth)
            device: Устройство для инференса
            config: Конфигурация модели (если None - загрузится из чекпоинта)
        """
        self.device = device
        
        # Загрузка чекпоинта
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Создаем конфигурацию если не передана
        if config is None:
            # Пытаемся получить из чекпоинта
            config = UNet3DConfiguration(
                in_channels=1,
                out_channels=2,
                base_filters=32,
                depth=4,
                dropout_rate=0.2
            )
        
        # Создаем модель
        self.model = UNet3D(config).to(device)
        
        # Загружаем веса
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        print(f"[OK] Model loaded: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best Dice: {checkpoint.get('best_dice', 'N/A'):.4f}")
        
        # Параметры нормализации
        self.clip_range = (-200, 250)
    
    def _normalize_ct(self, image: np.ndarray) -> np.ndarray:
        """Нормализует КТ изображение."""
        image = np.clip(image, self.clip_range[0], self.clip_range[1])
        
        min_val, max_val = self.clip_range
        image = (image - min_val) / (max_val - min_val)
        
        return image.astype(np.float32)
    
    def _resize_volume(
        self, 
        volume: np.ndarray, 
        target_shape: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Изменяет размер объёма.
        
        Returns:
            Измененный объём и коэффициенты масштабирования
        """
        current_shape = np.array(volume.shape)
        target_shape = np.array(target_shape)
        
        factors = target_shape / current_shape
        
        resized = zoom(volume, factors, order=1)
        
        return resized, factors
    
    def _restore_size(
        self, 
        volume: np.ndarray, 
        original_shape: Tuple[int, int, int],
        factors: Tuple[float, float, float]
    ) -> np.ndarray:
        """Восстанавливает исходный размер."""
        # Обратный зум
        restored = zoom(volume, 1/np.array(factors), order=1)
        
        # Обрезаем до исходного размера
        result = np.zeros(original_shape)
        
        # Копируем с центрированием
        start_z = (restored.shape[0] - original_shape[0]) // 2
        start_y = (restored.shape[1] - original_shape[1]) // 2
        start_x = (restored.shape[2] - original_shape[2]) // 2
        
        end_z = start_z + original_shape[0]
        end_y = start_y + original_shape[1]
        end_x = start_x + original_shape[2]
        
        # Если размеры совпадают
        if restored.shape == original_shape:
            return restored
        
        # Ограничиваем индексы
        result = restored[
            max(0, start_z):min(restored.shape[0], end_z),
            max(0, start_y):min(restored.shape[1], end_y),
            max(0, start_x):min(restored.shape[2], end_x)
        ]
        
        return result
    
    def predict_volume(
        self,
        volume: np.ndarray,
        target_size: Tuple[int, int, int] = (64, 128, 128),
        apply_threshold: bool = True,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Применяет модель к объёму.
        
        Args:
            volume: Входной объём формы (D, H, W)
            target_size: Размер для подачи в модель
            apply_threshold: Применять порог
            threshold: Порог для бинаризации
            
        Returns:
            Маска сегментации формы (D, H, W)
        """
        original_shape = volume.shape
        
        # Нормализация
        volume = self._normalize_ct(volume)
        
        # Изменение размера
        volume_resized, factors = self._resize_volume(volume, target_size)
        
        # Добавляем канал
        volume_tensor = torch.from_numpy(volume_resized).unsqueeze(0).unsqueeze(0)
        volume_tensor = volume_tensor.to(self.device)
        
        # Инференс
        with torch.no_grad():
            output = self.model(volume_tensor)  # (1, 2, D, H, W)
            
            # softmax и берем класс печени
            probs = torch.softmax(output, dim=1)[:, 1, :, :, :]  # (1, D, H, W)
            
            if apply_threshold:
                mask = (probs > threshold).float()
            else:
                mask = probs
            
            mask = mask.squeeze().cpu().numpy()  # (D, H, W)
        
        # Восстанавливаем исходный размер
        mask_resored = self._restore_size(mask, original_shape, factors)
        
        return mask_resored
    
    def predict_from_nifti(
        self,
        nifti_path: str,
        output_path: Optional[str] = None,
        target_size: Tuple[int, int, int] = (64, 128, 128)
    ) -> np.ndarray:
        """
        Применяет модель к NIfTI файлу.
        
        Args:
            nifti_path: Путь к входному NIfTI файлу
            output_path: Путь для сохранения результата (опционально)
            target_size: Размер для подачи в модель
            
        Returns:
            Маска сегментации
        """
        # Загрузка NIfTI
        img = nib.load(nifti_path)
        volume = img.get_fdata().astype(np.float32)
        
        # Предсказание
        mask = self.predict_volume(volume, target_size)
        
        # Сохранение результата
        if output_path:
            result_img = nib.Nifti1Image(mask.astype(np.uint8), img.affine, img.header)
            nib.save(result_img, output_path)
            print(f"✓ Результат сохранён: {output_path}")
        
        return mask
    
    def predict_from_dicom(
        self,
        dicom_path: str,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Применяет модель к DICOM файлу (серия срезов).
        
        Note: Для полноценной поддержки DICOM нужно объединить срезы в объём.
        """
        import pydicom
        
        # Читаем DICOM
        dcm = pydicom.dcmread(dicom_path)
        volume = dcm.pixel_array.astype(np.float32)
        
        # Применяем Rescale Slope и Intercept если есть
        if hasattr(dcm, 'RescaleSlope'):
            volume = volume * dcm.RescaleSlope + dcm.RescaleIntercept
        
        mask = self.predict_volume(volume)
        
        if output_path:
            result_img = nib.Nifti1Image(mask.astype(np.uint8), np.eye(4))
            nib.save(result_img, output_path)
        
        return mask


def load_model(
    checkpoint_path: str,
    device: str = "cuda"
) -> LiverSegmentationModel:
    """
    Удобная функция для загрузки модели.
    
    Args:
        checkpoint_path: Путь к чекпоинту
        device: Устройство
        
    Returns:
        Готовый к использованию LiverSegmentationModel
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA недоступна, используем CPU")
        device = "cpu"
    
    return LiverSegmentationModel(checkpoint_path, device=device)


# Тестирование
if __name__ == "__main__":
    import sys
    
    # Проверка наличия модели
    checkpoint_path = "checkpoints/best_model.pth"
    
    if Path(checkpoint_path).exists():
        # Загрузка модели
        model = load_model(checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Тест на одном файле из датасета
        test_image = "Task03_Liver_rs/imagesTr/liver_0.nii"
        
        if Path(test_image).exists():
            output_path = "uploads/segmentation_result.nii.gz"
            
            mask = model.predict_from_nifti(
                test_image,
                output_path=output_path,
                target_size=(64, 128, 128)
            )
            
            print(f"Размер маски: {mask.shape}")
            print(f"Вокселей печени: {mask.sum()}")
            print(f"Общий объём: {mask.size}")
    else:
        print(f"Модель не найдена: {checkpoint_path}")
        print("Сначала обучите модель: python -m app.models.train")
