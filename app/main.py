from fastapi import FastAPI, HTTPException, File, UploadFile
import os
import uuid
from pathlib import Path

from multipart import file_path

app = FastAPI(
    title="Liver Segmentation Service",
    description="API для сегментации КТ печени",
    version="1.0.0",
)

UPLOAD_DIRECTION = Path("uploads")
UPLOAD_DIRECTION.mkdir(exist_ok=True)


@app.get("/")
def start_service():
    return {"message": "service is running"}


@app.post("/upload")
async def upload_files(file: UploadFile = File()):

    valid_ext = [".nii", ".nii.gz"]
    file_ext = None

    if file.filename.endswith(".nii.gz"):
        file_ext = ".nii.gz"
    elif file.filename.endswith(".nii"):
        file_ext = ".nii"

    if file_ext not in valid_ext:
        return HTTPException(
            status_code=400,
            detail=f"Неверный формат файла, возможны: {valid_ext}",
        )

    file_id = str(uuid.uuid4())

    # Сохраняем файл в папку uploads (решение на время)
    if file_ext == ".nii.gz":
        file_path = UPLOAD_DIRECTION / f"{file_id}.nii.gz"
    elif file_ext == ".nii":
        file_path = UPLOAD_DIRECTION / f"{file_id}.nii"

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    file_size = os.path.getsize(file_path)

    return {
        "message": "Файл загружен!",
        "file_id": file_id,
    }
