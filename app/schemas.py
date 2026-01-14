from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


class FileType(str, Enum):
    DICOM = "dicom"
    NIFTI = "nifti"
    NIFTI_GZ = "nifti_gz"


class Modality(str, Enum):
    CT = "CT"
    MRI = "MR"
    XRAY = "XR"
    UNKNOWN = "UNKNOWN"


class DicomMetadata(BaseModel):
    """Метаданные DICOM файла (только безопасные поля)"""

    sop_instance_uid: Optional[str] = Field(None, description="Уникальный ID среза")
    series_instance_uid: Optional[str] = Field(None, description="Уникальный ID серии")
    study_instance_uid: Optional[str] = Field(
        None, description="Уникальный ID исследования"
    )

    modality: Optional[Modality] = Field(
        None, description="Тип исследования (CT, MR и т.д.)"
    )
    series_description: Optional[str] = Field(None, description="Описание серии")

    # Размеры и параметры изображения
    rows: Optional[int] = Field(None, description="Высота изображения в пикселях")
    columns: Optional[int] = Field(None, description="Ширина изображения в пикселях")
    bits_allocated: Optional[int] = Field(None, description="Бит на пиксель")
    bits_stored: Optional[int] = Field(None, description="Используемых бит")

    slice_thickness: Optional[float] = Field(None, description="Толщина среза (мм)")
    slice_location: Optional[float] = Field(None, description="Позиция среза (мм)")
    pixel_spacing: Optional[List[float]] = Field(
        None, description="Размер пикселя [row, column] в мм"
    )

    rescale_intercept: Optional[float] = Field(None, description="Интерcept для HU")
    rescale_slope: Optional[float] = Field(None, description="Slope для HU")
    window_center: Optional[float] = Field(None, description="Центр окна")
    window_width: Optional[float] = Field(None, description="Ширина окна")

    class Config:
        json_schema_extra = {
            "example": {
                "modality": "CT",
                "rows": 512,
                "columns": 512,
                "slice_thickness": 2.5,
                "pixel_spacing": [0.703, 0.703],
            }
        }


class SliceInfo(BaseModel):
    """Информация об одном срезе DICOM"""

    slice_index: int = Field(..., description="Индекс среза в серии")
    slice_location: Optional[float] = Field(None, description="Позиция среза в мм")
    image_size: List[int] = Field(..., description="Размер [ширина, высота]")
    has_image: bool = Field(..., description="Есть ли пиксельные данные")

    class Config:
        json_schema_extra = {
            "example": {
                "slice_index": 0,
                "slice_location": -125.0,
                "image_size": [512, 512],
                "has_image": True,
            }
        }


class SeriesInfo(BaseModel):
    """Информация о DICOM серии"""

    series_instance_uid: Optional[str] = Field(None, description="Уникальный ID серии")
    modality: Optional[Modality] = Field(None, description="Тип исследования")
    series_description: Optional[str] = Field(None, description="Описание")

    # Количество и параметры срезов
    num_slices: int = Field(..., description="Количество срезов в серии")
    slice_thickness: Optional[float] = Field(None, description="Толщина среза (мм)")
    pixel_spacing: Optional[List[float]] = Field(
        None, description="Размер пикселя в мм"
    )

    # Размеры изображения
    image_width: Optional[int] = Field(None, description="Ширина изображения")
    image_height: Optional[int] = Field(None, description="Высота изображения")

    # Диапазон позиций срезов
    min_slice_location: Optional[float] = Field(
        None, description="Минимальная позиция (мм)"
    )
    max_slice_location: Optional[float] = Field(
        None, description="Максимальная позиция (мм)"
    )

    # Список срезов
    slices: List[SliceInfo] = Field(default=[], description="Информация о каждом срезе")

    class Config:
        json_schema_extra = {
            "example": {
                "modality": "CT",
                "num_slices": 120,
                "slice_thickness": 2.5,
                "image_width": 512,
                "image_height": 512,
                "min_slice_location": -150.0,
                "max_slice_location": 150.0,
            }
        }


class UploadResponse(BaseModel):
    """Схема ответа после успешной загрузки файла"""

    message: str = "Файл успешно загружен"
    file_id: str = Field(..., description="Уникальный идентификатор файла на сервере")
    filename: str = Field(..., description="Оригинальное имя файла")
    file_type: FileType = Field(..., description="Тип файла")
    file_size_bytes: int = Field(..., description="Размер файла в байтах")
    file_size_mb: float = Field(..., description="Размер файла в мегабайтах")

    # Для DICOM добавляем метаданные
    dicom_metadata: Optional[DicomMetadata] = Field(
        None, description="Метаданные DICOM файла"
    )
    is_dicom_series: bool = Field(default=False, description="Является ли частью серии")

    uploaded_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Файл успешно загружен",
                "file_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "CT_Abdomen_001.dcm",
                "file_type": "dicom",
                "file_size_bytes": 524288,
                "file_size_mb": 0.5,
                "is_dicom_series": True,
                "uploaded_at": "2024-01-15T14:30:00Z",
            }
        }


class HealthCheck(BaseModel):
    status: str = "Healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    service: str = "liver-segmentation-api"
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    message: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class FileInfoResponse(BaseModel):
    file_id: str
    filename: str
    file_size_bytes: int
    file_size_mb: float
    file_extension: str = Field(..., description="Расширение файла")
    file_type: str = Field(..., description="Тип файла: dicom или nifti")
    upload_date: datetime
    exists: bool = Field(..., description="Файл существует на сервере")

    dicom_metadata: Optional[DicomMetadata] = Field(
        None, description="Метаданные DICOM"
    )
    is_dicom_series: Optional[bool] = Field(
        None, description="Является ли частью серии"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "patient_001.nii.gz",
                "file_size_bytes": 52428800,
                "file_size_mb": 50.0,
                "file_extension": ".nii.gz",
                "file_type": "nifti",
                "upload_date": "2024-01-15T14:30:00Z",
                "exists": True,
            }
        }


class FileListItem(BaseModel):
    file_id: str
    filename: str
    upload_date: datetime
    file_size_mb: float
    file_type: str = Field(..., description="Тип файла: dicom или nifti")

    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "CT_Abdomen_001.dcm",
                "upload_date": "2024-01-15T14:30:00Z",
                "file_size_mb": 0.5,
                "file_type": "dicom",
            }
        }


class FileListResponse(BaseModel):
    files: List[FileListItem]
    total_count: int
    total_size_mb: float
    summary: Dict[str, int] = Field(
        default_factory=dict, description="Статистика по типам файлов"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "files": [
                    {
                        "file_id": "550e8400-e29b-41d4-a716-446655440000",
                        "filename": "CT_Abdomen_001.dcm",
                        "upload_date": "2024-01-15T14:30:00Z",
                        "file_size_mb": 0.5,
                        "file_type": "dicom",
                    }
                ],
                "total_count": 1,
                "total_size_mb": 0.5,
                "summary": {"dicom": 1, "nifti": 0},
            }
        }


class SegmentationResponse(BaseModel):
    slice_index: int = Field(..., description="Индекс обработанного среза")
    file_id: str = Field(..., description="ID файла")
    file_type: str = Field(..., description="Тип файла: dicom или nifti")

    mask: List[List[int]] = Field(
        ..., description="Бинарная маска сегментации 256x256 (0 - фон, 1 - печень)"
    )

    contours: List[List[List[int]]] = Field(
        ...,
        description="Список контуров. Каждый контур - список точек [[x1,y1], [x2,y2], ...]",
    )

    image: str = Field(
        ...,
        description="PNG изображение среза в формате data URI (base64)",
        example="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
    )

    image_size: List[int] = Field(
        default=[256, 256], description="Размер изображения [ширина, высота]"
    )

    status: str = Field(
        default="mock_data",
        description="Статус обработки: 'processing', 'completed', 'mock_data', 'error'",
    )

    note: Optional[str] = Field(
        default=None, description="Дополнительная информация или предупреждения"
    )

    processed_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "slice_index": 42,
                "file_id": "550e8400-e29b-41d4-a716-446655440000",
                "file_type": "dicom",
                "mask": [[0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0]],
                "contours": [[[100, 150], [120, 150], [120, 170], [100, 170]]],
                "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                "image_size": [256, 256],
                "status": "mock_data",
                "note": "Это тестовые данные. Реальная модель в разработке.",
                "processed_at": "2024-01-15T14:35:00Z",
            }
        }
