from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request, Form
from typing import Optional
import os
from pathlib import Path
import uuid
from datetime import datetime
import shutil
import io
import base64
import sys

print(f"=== SERVER STARTED ===")
print(f"Python path: {sys.executable}")
print(f"CWD: {os.getcwd()}")
print(f"Python files location: {__file__}")
print(f"Files in cwd: {os.listdir('.')}")
print(f"========================")

from fastapi.middleware.cors import CORSMiddleware

import torch
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from app.schemas import (
    HealthCheck,
    UploadResponse,
    ErrorResponse,
    FileInfoResponse,
    FileListResponse,
    SegmentationResponse,
    DicomMetadata,
    SeriesInfo,
    SliceInfo,
    FileType,
    Modality,
    FileListItem,
)

from app.dicom_utils import (
    read_dicom_file,
    extract_dicom_metadata,
    dicom_to_base64,
    get_slice_info,
    is_dicom_file,
)

# Импорт для сегментации
from app.models.inference import LiverSegmentationModel, load_model

# Глобальная переменная для модели
segmentation_model: Optional[LiverSegmentationModel] = None
MODEL_CHECKPOINT = Path(__file__).parent.parent / "checkpoints" / "best_model.pth"

def get_segmentation_model() -> Optional[LiverSegmentationModel]:
    global segmentation_model

    if segmentation_model is None:
        import os
        checkpoint_path = MODEL_CHECKPOINT

        print(f"Попытка загрузить модель из: {checkpoint_path}")
        print(f"Файл существует: {checkpoint_path.exists()}")

        if checkpoint_path.exists():
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                segmentation_model = load_model(str(checkpoint_path), device=device)
                print(f"✓ Модель сегментации загружена на устройство: {device}")
            except Exception as e:
                print(f"Ошибка загрузки модели: {e}")
                import traceback
                traceback.print_exc()
                segmentation_model = None
        else:
            print(f"Модель не найдена: {checkpoint_path}")

    return segmentation_model

app = FastAPI(
    title="Liver CT Segmentation API",
    description="API для сегментации печени на КТ-снимках с поддержкой DICOM",
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Настройки CORS для разрешения запросов с фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

DICOM_DIR = UPLOAD_DIR / "dicom_series"
DICOM_DIR.mkdir(exist_ok=True)

NIFTI_DIR = UPLOAD_DIR / "nifti"
NIFTI_DIR.mkdir(exist_ok=True)


def get_file_type(filename: str) -> FileType:
    filename_lower = filename.lower()

    if filename_lower.endswith(".dcm") or filename_lower.endswith(".dicom"):
        return FileType.DICOM
    elif filename_lower.endswith(".nii.gz"):
        return FileType.NIFTI_GZ
    elif filename_lower.endswith(".nii"):
        return FileType.NIFTI
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {filename}")


def save_uploaded_file(file: UploadFile, file_id: str, file_type: FileType) -> Path:

    if file_type == FileType.DICOM:
        ext = ".dcm"
        save_dir = DICOM_DIR
    elif file_type == FileType.NIFTI_GZ:
        ext = ".nii.gz"
        save_dir = NIFTI_DIR
    else:
        ext = ".nii"
        save_dir = NIFTI_DIR

    save_filename = f"{file_id}{ext}"
    file_path = save_dir / save_filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return file_path


def format_file_size(size_bytes: int) -> float:
    return round(size_bytes / (1024 * 1024), 2)


def process_dicom_file(file_path: Path) -> tuple[DicomMetadata, bool]:
    try:
        # Читаем DICOM файл
        dicom_dataset = read_dicom_file(str(file_path))
        if dicom_dataset is None:
            raise ValueError("Не удалось прочитать DICOM файл")

        metadata = extract_dicom_metadata(dicom_dataset)

        is_series = (
            metadata.series_instance_uid is not None
            and metadata.slice_location is not None
        )

        return metadata, is_series

    except Exception as e:
        print(f"Ошибка обработки DICOM файла {file_path}: {e}")
        return DicomMetadata(), False


@app.get("/", response_model=dict)
async def root():
    return {
        "message": "Liver CT Segmentation API",
        "version": "0.2.0",
        "status": "development",
        "features": [
            "Поддержка DICOM файлов",
            "Предпросмотр срезов",
            "Заглушка сегментации",
        ],
        "endpoints": {
            "health": "GET /health",
            "upload": "POST /upload",
            "file_info": "GET /files/{file_id}",
            "list_files": "GET /files",
            "dicom_metadata": "GET /dicom/{file_id}/metadata",
            "dicom_preview": "GET /dicom/{file_id}/preview",
            "dicom_slice": "GET /dicom/{file_id}/slice",
            "segment": "GET /segment",
        },
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Проверка здоровья сервера"""
    return HealthCheck()


@app.post(
    "/upload/folder",
    responses={400: {"model": ErrorResponse}},
    name="upload_folder"
)
async def upload_dicom_folder(request: Request):
    """
    Загрузка нескольких DICOM файлов (папки/серии).

    Принимает multipart/form-data с несколькими файлами под ключом 'files'.
    """
    # Получаем все файлы из request
    try:
        form = await request.form()
        # Правильно получаем все файлы с ключом 'files'
        file_list = form.getlist('files')
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "detail": f"Ошибка при чтении формы: {str(e)}",
                "error_code": "FORM_PARSE_ERROR",
            },
        )

    print(f"Получено файлов: {len(file_list)}")

    if not file_list:
        raise HTTPException(
            status_code=400,
            detail={
                "detail": "Файлы не предоставлены",
                "error_code": "NO_FILES",
            },
        )

    # Генерируем общий ID для всей серии
    series_id = str(uuid.uuid4())
    uploaded_files = []
    errors = []
    total_size = 0

    # Загружаем каждый файл
    for idx, file in enumerate(file_list):
        if idx % 20 == 0:
            print(f"Обработка файла {idx + 1}/{len(file_list)}")
        try:
            file_type = get_file_type(file.filename)
            if file_type != FileType.DICOM:
                continue

            # Используем оригинальное имя для идентификации среза
            file_id = Path(file.filename).stem

            file_path = save_uploaded_file(file, file_id, file_type)
            file_size = os.path.getsize(file_path)
            total_size += file_size

            uploaded_files.append({
                "file_id": file_id,
                "filename": file.filename,
                "size_mb": format_file_size(file_size)
            })

        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")

    # Проверяем, что загружены файлы
    if not uploaded_files:
        raise HTTPException(
            status_code=400,
            detail={
                "detail": "Не удалось загрузить ни одного файла",
                "error_code": "UPLOAD_ERROR",
                "errors": errors,
            },
        )

    # Сортируем по имени (для правильного порядка срезов)
    uploaded_files.sort(key=lambda x: x["file_id"])

    return {
        "message": f"Загружено {len(uploaded_files)} DICOM файлов",
        "series_id": series_id,
        "total_files": len(uploaded_files),
        "total_size_mb": format_file_size(total_size),
        "files": uploaded_files,  # Возвращаем все файлы
        "errors": errors[:10] if errors else [],
        "status": "completed"
    }


@app.get(
    "/dicom/{file_id}/metadata",
    response_model=DicomMetadata,
    responses={404: {"model": ErrorResponse}},
)
async def get_dicom_metadata(file_id: str):
    """
    Получение метаданных DICOM файла.
    """
    file_path = DICOM_DIR / f"{file_id}.dcm"
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "detail": f"DICOM файл с ID {file_id} не найден",
                "error_code": "DICOM_FILE_NOT_FOUND",
            },
        )

    # Читаем и обрабатываем DICOM файл
    metadata, _ = process_dicom_file(file_path)

    return metadata


@app.get("/dicom/{file_id}/preview", responses={404: {"model": ErrorResponse}})
async def get_dicom_preview(
    file_id: str,
    window_center: float = Query(40.0, description="Центр окна (window center) для CT"),
    window_width: float = Query(400.0, description="Ширина окна (window width) для CT"),
):
    file_path = DICOM_DIR / f"{file_id}.dcm"
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "detail": f"DICOM файл с ID {file_id} не найден",
                "error_code": "DICOM_FILE_NOT_FOUND",
            },
        )

    try:
        dicom_dataset = read_dicom_file(str(file_path))
        if dicom_dataset is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "detail": "Не удалось прочитать DICOM файл",
                    "error_code": "DICOM_READ_ERROR",
                },
            )

        # Конвертируем в base64 PNG
        image_base64 = dicom_to_base64(
            dicom_dataset, window_center=window_center, window_width=window_width
        )

        if image_base64 is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "detail": "DICOM файл не содержит изображения",
                    "error_code": "NO_IMAGE_DATA",
                },
            )

        slice_info = get_slice_info(dicom_dataset)

        return {
            "file_id": file_id,
            "image": image_base64,
            "slice_info": slice_info,
            "window_settings": {"center": window_center, "width": window_width},
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "detail": f"Ошибка при обработке DICOM файла: {str(e)}",
                "error_code": "DICOM_PROCESSING_ERROR",
            },
        )


@app.get(
    "/dicom/{file_id}/slice",
    response_model=SliceInfo,
    responses={404: {"model": ErrorResponse}},
)
async def get_dicom_slice_info(file_id: str):
    """
    Получение информации о DICOM срезе.
    """
    file_path = DICOM_DIR / f"{file_id}.dcm"
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "detail": f"DICOM файл с ID {file_id} не найден",
                "error_code": "DICOM_FILE_NOT_FOUND",
            },
        )

    dicom_dataset = read_dicom_file(str(file_path))
    if dicom_dataset is None:
        raise HTTPException(
            status_code=400,
            detail={
                "detail": "Не удалось прочитать DICOM файл",
                "error_code": "DICOM_READ_ERROR",
            },
        )

    slice_info = get_slice_info(dicom_dataset)

    return slice_info


@app.get(
    "/files/{file_id}",
    response_model=FileInfoResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_file_info(file_id: str):

    file_path = None
    file_type = None
    dicom_metadata = None
    is_dicom_series = False

    dicom_path = DICOM_DIR / f"{file_id}.dcm"
    if dicom_path.exists():
        file_path = dicom_path
        file_type = "dicom"

        metadata, is_series = process_dicom_file(file_path)
        dicom_metadata = metadata
        is_dicom_series = is_series

    if file_path is None:
        nifti_path = NIFTI_DIR / f"{file_id}.nii"
        nifti_gz_path = NIFTI_DIR / f"{file_id}.nii.gz"

        if nifti_path.exists():
            file_path = nifti_path
            file_type = "nifti"
        elif nifti_gz_path.exists():
            file_path = nifti_gz_path
            file_type = "nifti"

    if not file_path:
        raise HTTPException(
            status_code=404,
            detail={
                "detail": f"Файл с ID {file_id} не найден",
                "error_code": "FILE_NOT_FOUND",
            },
        )

    if file_path.suffix == ".dcm":
        file_ext = ".dcm"
    elif file_path.suffixes == [".nii", ".gz"]:
        file_ext = ".nii.gz"
    else:
        file_ext = ".nii"

    file_size = os.path.getsize(file_path)
    upload_date = datetime.fromtimestamp(os.path.getmtime(file_path))

    return FileInfoResponse(
        file_id=file_id,
        filename=file_path.name,
        file_size_bytes=file_size,
        file_size_mb=format_file_size(file_size),
        file_extension=file_ext,
        file_type=file_type,
        upload_date=upload_date,
        exists=True,
        dicom_metadata=dicom_metadata,
        is_dicom_series=is_dicom_series,
    )


@app.get("/files", response_model=FileListResponse)
async def list_files():
    """
    Получение списка всех загруженных файлов (DICOM и NIfTI).
    """
    files = []
    total_size = 0
    file_type_count = {"dicom": 0, "nifti": 0}

    scan_dirs = [DICOM_DIR, NIFTI_DIR]

    for scan_dir in scan_dirs:
        if scan_dir == DICOM_DIR:
            file_type = "dicom"
            patterns = ["*.dcm"]
        else:
            file_type = "nifti"
            patterns = ["*.nii", "*.nii.gz"]

        for pattern in patterns:
            for file_path in scan_dir.glob(pattern):
                if file_path.is_file():
                    file_size = os.path.getsize(file_path)
                    file_size_mb = format_file_size(file_size)

                    file_id = file_path.stem  # Убираем основное расширение

                    if file_path.suffixes == [".nii", ".gz"]:
                        file_id = file_path.name[:-7]  # Убираем .nii.gz

                    upload_date = datetime.fromtimestamp(os.path.getmtime(file_path))

                    files.append(
                        FileListItem(
                            file_id=file_id,
                            filename=file_path.name,
                            upload_date=upload_date,
                            file_size_mb=file_size_mb,
                            file_type=file_type,
                        )
                    )

                    total_size += file_size_mb
                    file_type_count[file_type] += 1

    return FileListResponse(
        files=files,
        total_count=len(files),
        total_size_mb=round(total_size, 2),
        summary=file_type_count,
    )


@app.post(
    "/segment/nifti/{file_id}",
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def segment_nifti_file(
    file_id: str,
    target_size_d: int = Query(64, description="Целевая глубина для модели"),
    target_size_h: int = Query(128, description="Целевая высота для модели"),
    target_size_w: int = Query(128, description="Целевая ширина для модели"),
):
    """
    Сегментация печени на NIfTI файле.

    Загружает NIfTI файл, применяет 3D U-Net модель и возвращает маску сегментации.
    """
    # Поиск файла
    nifti_path = NIFTI_DIR / f"{file_id}.nii"
    nifti_gz_path = NIFTI_DIR / f"{file_id}.nii.gz"

    file_path = None
    if nifti_path.exists():
        file_path = nifti_path
    elif nifti_gz_path.exists():
        file_path = nifti_gz_path

    if not file_path:
        raise HTTPException(
            status_code=404,
            detail={
                "detail": f"NIfTI файл с ID {file_id} не найден",
                "error_code": "NIFTI_FILE_NOT_FOUND",
            },
        )

    # Получаем модель
    model = get_segmentation_model()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "detail": "Модель сегментации недоступна. Обучите модель.",
                "error_code": "MODEL_NOT_LOADED",
            },
        )

    try:
        # Применяем модель
        target_size = (target_size_d, target_size_h, target_size_w)

        # Загружаем объём
        img = nib.load(str(file_path))
        volume = img.get_fdata().astype(np.float32)

        # Сегментация
        mask = model.predict_volume(volume, target_size=target_size)

        # Сохраняем результат
        output_path = UPLOAD_DIR / "segmentation" / f"{file_id}_segmentation.nii.gz"
        output_path.parent.mkdir(exist_ok=True)

        result_img = nib.Nifti1Image(mask.astype(np.uint8), img.affine, img.header)
        nib.save(result_img, str(output_path))

        # Статистика
        liver_voxels = int(mask.sum())
        total_voxels = int(mask.size)
        liver_percentage = round(liver_voxels / total_voxels * 100, 2) if total_voxels > 0 else 0

        return {
            "file_id": file_id,
            "status": "completed",
            "output_file_id": file_id + "_segmentation",
            "statistics": {
                "liver_voxels": liver_voxels,
                "total_voxels": total_voxels,
                "liver_percentage": liver_percentage,
                "original_shape": list(volume.shape),
                "mask_shape": list(mask.shape)
            },
            "message": "Сегментация завершена успешно"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "detail": f"Ошибка при сегментации: {str(e)}",
                "error_code": "SEGMENTATION_ERROR",
            },
        )


@app.post(
    "/segment/dicom",
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def segment_dicom_series(
    series_id: str = Query(..., description="ID серии DICOM файлов"),
    target_size_d: int = Query(64, description="Целевая глубина для модели"),
    target_size_h: int = Query(128, description="Целевая высота для модели"),
    target_size_w: int = Query(128, description="Целевая ширина для модели"),
):
    """
    Сегментация печени на DICOM серии.

    Собирает все DICOM файлы серии в 3D объём,
    применяет 3D U-Net модель и возвращает маски срезов.
    """
    import pydicom

    # Получаем все DICOM файлы
    dicom_files = sorted(DICOM_DIR.glob("*.dcm"))

    if not dicom_files:
        raise HTTPException(
            status_code=404,
            detail={
                "detail": "DICOM файлы не найдены",
                "error_code": "NO_DICOM_FILES",
            },
        )

    print(f"Найдено DICOM файлов: {len(dicom_files)}")

    # Читаем первый файл для получения размеров
    first_dcm = pydicom.dcmread(str(dicom_files[0]))
    rows = first_dcm.Rows
    cols = first_dcm.Columns

    # Получаем Rescale параметры
    rescale_slope = getattr(first_dcm, 'RescaleSlope', 1)
    rescale_intercept = getattr(first_dcm, 'RescaleIntercept', 0)

    print(f"Размер среза: {rows}x{cols}, всего срезов: {len(dicom_files)}")

    # Собираем 3D объём
    volume = np.zeros((len(dicom_files), rows, cols), dtype=np.float32)

    for i, dcm_path in enumerate(dicom_files):
        try:
            dcm = pydicom.dcmread(str(dcm_path))
            slice_data = dcm.pixel_array.astype(np.float32)

            # Применяем Rescale
            slice_data = slice_data * rescale_slope + rescale_intercept

            volume[i] = slice_data
        except Exception as e:
            print(f"Ошибка чтения {dcm_path}: {e}")
            continue

    # Проверяем, что объём не пустой
    if volume.size == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "detail": "Не удалось собрать объём из DICOM файлов",
                "error_code": "VOLUME_BUILD_ERROR",
            },
        )

    # Получаем модель
    model = get_segmentation_model()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "detail": "Модель сегментации недоступна",
                "error_code": "MODEL_NOT_LOADED",
            },
        )

    try:
        target_size = (target_size_d, target_size_h, target_size_w)

        # Сегментация
        mask = model.predict_volume(volume, target_size=target_size)

        print(f"Маска получена, форма: {mask.shape}")

        # Статистика
        liver_voxels = int(mask.sum())
        total_voxels = int(mask.size)
        liver_percentage = round(liver_voxels / total_voxels * 100, 2) if total_voxels > 0 else 0

        # Сохраняем результат как NIfTI
        output_path = UPLOAD_DIR / "segmentation" / f"{series_id}_segmentation.nii.gz"
        output_path.parent.mkdir(exist_ok=True)

        result_img = nib.Nifti1Image(mask.astype(np.uint8), np.eye(4))
        nib.save(result_img, str(output_path))

        # Конвертируем маски срезов в base64 для фронтенда
        slice_masks = []
        num_slices = min(mask.shape[0], 20)  # Ограничиваем для производительности

        for i in range(0, mask.shape[0], mask.shape[0] // num_slices):
            if i < mask.shape[0]:
                slice_mask = mask[i]

                # Масштабируем для отображения если нужно
                if slice_mask.shape != (rows, cols):
                    factors = (rows / slice_mask.shape[0], cols / slice_mask.shape[1])
                    slice_mask = zoom(slice_mask, factors, order=0)

                # Конвертируем в image
                slice_img = (slice_mask * 255).astype(np.uint8)
                pil_img = Image.fromarray(slice_img)

                buffered = io.BytesIO()
                pil_img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("ascii")

                slice_masks.append({
                    "slice_index": i,
                    "image": f"data:image/png;base64,{img_base64}"
                })

        return {
            "series_id": series_id,
            "status": "completed",
            "statistics": {
                "liver_voxels": liver_voxels,
                "total_voxels": total_voxels,
                "liver_percentage": liver_percentage,
                "original_shape": list(volume.shape),
                "mask_shape": list(mask.shape)
            },
            "slices": slice_masks,
            "message": "Сегментация завершена успешно"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "detail": f"Ошибка при сегментации: {str(e)}",
                "error_code": "SEGMENTATION_ERROR",
            },
        )


@app.get("/debug/model-path")
async def debug_model_path():
    """Отладочный эндпоинт для проверки пути к модели"""
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "best_model.pth"
    return {
        "checkpoint_path": str(checkpoint_path),
        "exists": checkpoint_path.exists(),
        "cwd": os.getcwd(),
    }


@app.get(
    "/segment/status",
)
async def get_segmentation_status():
    """
    Проверка статуса модели сегментации.
    """
    model = get_segmentation_model()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is not None:
        return {
            "status": "ready",
            "model_loaded": True,
            "device": device,
            "checkpoint": MODEL_CHECKPOINT,
            "message": "Модель готова к использованию"
        }
    else:
        return {
            "status": "not_loaded",
            "model_loaded": False,
            "device": device,
            "checkpoint": MODEL_CHECKPOINT,
            "message": "Модель не загружена. Обучите модель: python -m app.models.train"
        }

