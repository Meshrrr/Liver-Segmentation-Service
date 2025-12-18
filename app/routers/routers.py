from fastapi import APIRouter, File, UploadFile, HTTPException
import os
import uuid
from pathlib import Path

router = APIRouter()
