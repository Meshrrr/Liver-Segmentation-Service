from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import os
from pathlib import Path
import uuid
from datetime import datetime
import shutil

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

app = FastAPI(
    title="Liver CT Segmentation API",
    description="API для сегментации печени на КТ-снимках с поддержкой DICOM",
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
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
    "/upload", response_model=UploadResponse, responses={400: {"model": ErrorResponse}}
)
async def upload_medical_file(file: UploadFile = File(...)):

    try:
        try:
            file_type = get_file_type(file.filename)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "detail": str(e),
                    "error_code": "INVALID_FILE_TYPE",
                    "supported_formats": [
                        "DICOM (.dcm, .dicom)",
                        "NIfTI (.nii, .nii.gz)",
                    ],
                },
            )

        file_id = str(uuid.uuid4())

        file_path = save_uploaded_file(file, file_id, file_type)

        file_size = os.path.getsize(file_path)

        dicom_metadata = None
        is_dicom_series = False

        if file_type == FileType.DICOM:
            metadata, is_series = process_dicom_file(file_path)
            dicom_metadata = metadata
            is_dicom_series = is_series

        return UploadResponse(
            message=f"{file_type.value.upper()} файл успешно загружен",
            file_id=file_id,
            filename=file.filename,
            file_type=file_type,
            file_size_bytes=file_size,
            file_size_mb=format_file_size(file_size),
            dicom_metadata=dicom_metadata,
            is_dicom_series=is_dicom_series,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "detail": f"Ошибка при загрузке файла: {str(e)}",
                "error_code": "UPLOAD_ERROR",
            },
        )


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
