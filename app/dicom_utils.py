import pydicom
from pydicom.dataset import FileDataset
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from PIL import Image
import io
import base64

from app.schemas import DicomMetadata, SliceInfo, SeriesInfo, Modality


def read_dicom_file(file_path: str) -> Optional[FileDataset]:
    """
    Читает DICOM файл и возвращает dataset.
    """

    try:
        dicom_dataset = pydicom.dcmread(file_path)
        return dicom_dataset
    except Exception as e:
        print(f"Ошибка чтения DICOM файла {file_path}: {e}")
        return None


def extract_dicom_metadata(dicom_dataset: FileDataset) -> DicomMetadata:
    """
    Извлекает безопасные метаданные из DICOM файла.
    """
    metadata = {}

    def safe_get(attr_name, default=None):
        try:
            if hasattr(dicom_dataset, attr_name):
                value = getattr(dicom_dataset, attr_name)
                if value is not None:
                    return value
        except Exception:
            pass
        return default

    sop_uid = safe_get("SOPInstanceUID")
    if sop_uid:
        metadata["sop_instance_uid"] = str(sop_uid)

    series_uid = safe_get("SeriesInstanceUID")
    if series_uid:
        metadata["series_instance_uid"] = str(series_uid)

    study_uid = safe_get("StudyInstanceUID")
    if study_uid:
        metadata["study_instance_uid"] = str(study_uid)

    # Modality
    modality_str = safe_get("Modality")
    modality = Modality.UNKNOWN
    if modality_str:
        modality_str = str(modality_str).upper()
        if modality_str in [m.value for m in Modality]:
            modality = Modality(modality_str)
    metadata["modality"] = modality

    series_desc = safe_get("SeriesDescription")
    if series_desc:
        metadata["series_description"] = str(series_desc)

    rows = safe_get("Rows")
    if rows:
        metadata["rows"] = int(rows)

    columns = safe_get("Columns")
    if columns:
        metadata["columns"] = int(columns)

    bits_allocated = safe_get("BitsAllocated")
    if bits_allocated:
        metadata["bits_allocated"] = int(bits_allocated)

    bits_stored = safe_get("BitsStored")
    if bits_stored:
        metadata["bits_stored"] = int(bits_stored)

    slice_thickness = safe_get("SliceThickness")
    if slice_thickness:
        try:
            metadata["slice_thickness"] = float(slice_thickness)
        except (ValueError, TypeError):
            pass

    slice_location = safe_get("SliceLocation")
    if slice_location:
        try:
            metadata["slice_location"] = float(slice_location)
        except (ValueError, TypeError):
            pass

    # Pixel Spacing
    pixel_spacing = safe_get("PixelSpacing")
    if pixel_spacing:
        try:
            if hasattr(pixel_spacing, "__len__") and len(pixel_spacing) >= 2:
                metadata["pixel_spacing"] = [
                    float(pixel_spacing[0]),
                    float(pixel_spacing[1]),
                ]
        except (ValueError, TypeError):
            pass

    # Параметры для CT (единицы Хаунсфилда)
    rescale_intercept = safe_get("RescaleIntercept")
    if rescale_intercept is not None:
        try:
            metadata["rescale_intercept"] = float(rescale_intercept)
        except (ValueError, TypeError):
            pass

    rescale_slope = safe_get("RescaleSlope")
    if rescale_slope is not None:
        try:
            metadata["rescale_slope"] = float(rescale_slope)
        except (ValueError, TypeError):
            pass

    # Window Center и Width
    window_center = safe_get("WindowCenter")
    if window_center is not None:
        try:
            if hasattr(window_center, "__len__") and len(window_center) > 0:
                metadata["window_center"] = float(window_center[0])
            else:
                metadata["window_center"] = float(window_center)
        except (ValueError, TypeError):
            pass

    window_width = safe_get("WindowWidth")
    if window_width is not None:
        try:
            if hasattr(window_width, "__len__") and len(window_width) > 0:
                metadata["window_width"] = float(window_width[0])
            else:
                metadata["window_width"] = float(window_width)
        except (ValueError, TypeError):
            pass

    return DicomMetadata(**metadata)


def dicom_to_image(dicom_dataset: FileDataset) -> Optional[np.ndarray]:
    """
    Конвертирует DICOM пиксельные данные в numpy массив.
    """
    try:
        if not hasattr(dicom_dataset, "pixel_array"):
            return None

        pixel_array = dicom_dataset.pixel_array

        image_array = np.array(pixel_array, dtype=np.float32)

        return image_array

    except Exception as e:
        print(f"Ошибка конвертации DICOM в изображение: {e}")
        return None


def normalize_dicom_image(
    image_array: np.ndarray, window_center: float = 40.0, window_width: float = 400.0
) -> np.ndarray:
    """
    Нормализует DICOM изображение с использованием windowing.
    """
    try:
        # Вычисляем границы окна
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2

        # Обрезаем значения за пределами окна
        image_clipped = np.clip(image_array, window_min, window_max)

        # Нормализуем в диапазон 0-255
        image_normalized = ((image_clipped - window_min) / window_width) * 255

        # Конвертируем в 8-bit
        image_8bit = np.clip(image_normalized, 0, 255).astype(np.uint8)

        return image_8bit

    except Exception as e:
        print(f"Ошибка нормализации DICOM изображения: {e}")
        return image_array.astype(np.uint8)


def dicom_to_base64(
    dicom_dataset: FileDataset, window_center: float = 40.0, window_width: float = 400.0
) -> Optional[str]:
    """
    Конвертирует DICOM срез в base64 PNG изображение.
    """
    try:
        # Конвертируем в numpy массив
        image_array = dicom_to_image(dicom_dataset)
        if image_array is None:
            return None

        normalized_image = normalize_dicom_image(
            image_array, window_center, window_width
        )

        pil_image = Image.fromarray(normalized_image)

        # Конвертируем в base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("ascii")

        return f"data:image/png;base64,{img_base64}"

    except Exception as e:
        print(f"Ошибка конвертации DICOM в base64: {e}")
        return None


def get_slice_info(dicom_dataset: FileDataset, slice_index: int = 0) -> SliceInfo:
    """
    Получает информацию об одном срезе DICOM.
    """
    has_image = hasattr(dicom_dataset, "pixel_array")

    image_size = [0, 0]
    if hasattr(dicom_dataset, "Columns"):
        image_size[0] = int(dicom_dataset.Columns)
    if hasattr(dicom_dataset, "Rows"):
        image_size[1] = int(dicom_dataset.Rows)

    slice_location = None
    if hasattr(dicom_dataset, "SliceLocation"):
        slice_location = float(dicom_dataset.SliceLocation)

    return SliceInfo(
        slice_index=slice_index,
        slice_location=slice_location,
        image_size=image_size,
        has_image=has_image,
    )


def is_dicom_file(file_path: str) -> bool:
    """
    Проверяет, является ли файл DICOM файлом.
    """
    try:
        if not file_path.lower().endswith((".dcm", ".dicom")):
            return False

        dicom_dataset = pydicom.dcmread(file_path, force=True)

        # Проверяем наличие обязательных DICOM атрибутов
        required_attrs = ["SOPClassUID", "SOPInstanceUID"]
        for attr in required_attrs:
            if not hasattr(dicom_dataset, attr):
                return False

        return True

    except Exception:
        return False
