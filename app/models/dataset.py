"""
Датасет и DataLoader для загрузки NIfTI файлов печени.

Поддерживает:
- Автоматическую нормализацию (HU единицы для КТ)
- Аугментацию данных
- Преобразование в тензоры
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib


class LiverDataset(Dataset):
    """
    Датасет для сегментации печени из NIfTI файлов.
    
    Ожидаемая структура:
        - imagesTr/ - директория с изображениями
        - labelsTr/ - директория с масками
        
    Файлы должны именоваться одинаково: liver_0.nii, liver_1.nii и т.д.
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        clip_range: Optional[Tuple[int, int]] = (-200, 250),
        target_size: Optional[Tuple[int, int, int]] = None,
        augment: bool = False
    ):
        """
        Args:
            data_dir: Путь к директории с данными (содержит imagesTr и labelsTr)
            transform: Дополнительные трансформации
            normalize: Нормализовать значения в HU диапазон
            clip_range: Диапазон обрезания значений (для КТ это типично -200..250 HU)
            target_size: Целевой размер (D, H, W). Если None - не изменять
            augment: Применять аугментацию
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "imagesTr"
        self.labels_dir = self.data_dir / "labelsTr"
        
        self.transform = transform
        self.normalize = normalize
        self.clip_range = clip_range
        self.target_size = target_size
        self.augment = augment
        
        # Получаем список файлов
        self.file_ids = self._get_file_ids()
        
        print(f"Загружен датасет: {len(self.file_ids)} томов")
    
    def _get_file_ids(self) -> List[str]:
        """Получает список ID файлов (без расширения)."""
        image_files = list(self.images_dir.glob("liver_*.nii"))
        label_files = list(self.labels_dir.glob("liver_*.nii"))
        
        image_ids = {f.stem for f in image_files}
        label_ids = {f.stem for f in label_files}
        
        # Пересечение - только те файлы, где есть и изображение, и маска
        common_ids = sorted(image_ids & label_ids)
        
        return common_ids
    
    def __len__(self) -> int:
        return len(self.file_ids)
    
    def _load_nifti(self, file_path: Path) -> np.ndarray:
        """Загружает NIfTI файл и возвращает numpy массив."""
        img = nib.load(str(file_path))
        data = img.get_fdata()
        return data.astype(np.float32)
    
    def _normalize_ct(self, image: np.ndarray) -> np.ndarray:
        """
        Нормализует КТ изображение:
        1. Обрезает значения HU
        2. Нормализует в диапазон [0, 1]
        """
        if self.clip_range:
            image = np.clip(image, self.clip_range[0], self.clip_range[1])
        
        # Нормализация в [0, 1]
        min_val = self.clip_range[0] if self.clip_range else image.min()
        max_val = self.clip_range[1] if self.clip_range else image.max()
        
        image = (image - min_val) / (max_val - min_val)
        
        return image.astype(np.float32)
    
    def _resize_volume(
        self, 
        volume: np.ndarray, 
        target_size: Tuple[int, int, int]
    ) -> np.ndarray:
        """Изменяет размер объёма до целевого."""
        from scipy.ndimage import zoom
        
        current_shape = np.array(volume.shape)
        target_shape = np.array(target_size)
        
        factors = target_shape / current_shape
        
        # Интерполяция
        resized = zoom(volume, factors, order=1)
        
        return resized
    
    def _augment(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Аугментация данных:
        - Случайные отражения по осям
        - Случайные сдвиги
        """
        # Случайное отражение по осям
        if np.random.random() > 0.5:
            axis = np.random.choice([0, 1, 2])
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
        
        # Случайный сдвиг (небольшой)
        if np.random.random() > 0.5:
            shift = np.random.randint(-10, 10, size=3)
            image = np.roll(image, shift, axis=(0, 1, 2))
            label = np.roll(label, shift, axis=(0, 1, 2))
        
        return image, label
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Загружает один том и его маску.
        
        Returns:
            Словарь с 'image' и 'label'
        """
        file_id = self.file_ids[idx]
        
        # Загрузка изображения и маски
        image_path = self.images_dir / f"{file_id}.nii"
        label_path = self.labels_dir / f"{file_id}.nii"
        
        image = self._load_nifti(image_path)
        label = self._load_nifti(label_path)
        
        # Бинаризация маски (оставляем только печень = 1)
        label = (label > 0.5).astype(np.float32)
        
        # Нормализация изображения
        if self.normalize:
            image = self._normalize_ct(image)
        
        # Аугментация
        if self.augment:
            image, label = self._augment(image, label)
        
        # Изменение размера
        if self.target_size:
            image = self._resize_volume(image, self.target_size)
            label = self._resize_volume(label, self.target_size)
        
        # Дополнительные трансформации
        if self.transform:
            image, label = self.transform(image, label)
        
        # Добавляем размерность канала (C, D, H, W)
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)
        
        # Конвертация в тензоры
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.from_numpy(label).long()  # Для cross-entropy нужен long
        
        return {
            "image": image_tensor,
            "label": label_tensor,
            "file_id": file_id
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 2,
    train_ratio: float = 0.8,
    target_size: Tuple[int, int, int] = (64, 128, 128),
    num_workers: int = 0,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Создает train и validation DataLoaders.
    
    Args:
        data_dir: Путь к директории с данными
        batch_size: Размер батча
        train_ratio: Доля данных для обучения
        target_size: Размер томов
        num_workers: Количество рабочих процессов
        augment_train: Аугментация для train
        
    Returns:
        (train_loader, val_loader)
    """
    # Полный датасет
    full_dataset = LiverDataset(
        data_dir=data_dir,
        target_size=target_size,
        augment=False  # Без аугментации для валидации
    )
    
    # Разделение на train/val
    total_samples = len(full_dataset)
    train_size = int(train_ratio * total_samples)
    val_size = total_samples - train_size
    
    # Создаем индексы
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Разделяем датасеты
    train_dataset = LiverDataset(
        data_dir=data_dir,
        target_size=target_size,
        augment=augment_train
    )
    val_dataset = LiverDataset(
        data_dir=data_dir,
        target_size=target_size,
        augment=False
    )
    
    # Упрощенный способ - просто переопределяем индексы
    class SubsetDataset(Dataset):
        def __init__(self, dataset: Dataset, indices: List[int]):
            self.dataset = dataset
            self.indices = indices
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]
    
    train_subset = SubsetDataset(full_dataset, train_indices)
    val_subset = SubsetDataset(full_dataset, val_indices)
    
    # DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train: {len(train_subset)} томов, Val: {len(val_subset)} томов")
    
    return train_loader, val_loader


# Тестирование
if __name__ == "__main__":
    # Тест загрузки датасета
    data_dir = "Task03_Liver_rs"
    
    if Path(data_dir).exists():
        dataset = LiverDataset(data_dir, target_size=(32, 64, 64))
        
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Label shape: {sample['label'].shape}")
        print(f"File ID: {sample['file_id']}")
    else:
        print(f"Датасет не найден: {data_dir}")
