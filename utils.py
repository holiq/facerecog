import cv2
import numpy as np
import os
from PIL import Image, ImageOps
from io import BytesIO
from fastapi import HTTPException, UploadFile
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_face_app = None


def get_positive_float_env(name: str, default: float) -> float:
    value = os.getenv(name, str(default))
    try:
        parsed_value = float(value)
    except ValueError:
        logger.warning(f"Invalid {name} value '{value}', using {default}")
        return default

    if parsed_value <= 0:
        logger.warning(f"Invalid {name} value '{value}', using {default}")
        return default

    return parsed_value


FACE_MIN_IMAGE_SIZE = int(os.getenv('FACE_MIN_IMAGE_SIZE', '100'))
FACE_BLUR_THRESHOLD = float(os.getenv('FACE_BLUR_THRESHOLD', '50'))
FACE_BRIGHTNESS_MIN = float(os.getenv('FACE_BRIGHTNESS_MIN', '40'))
FACE_BRIGHTNESS_MAX = float(os.getenv('FACE_BRIGHTNESS_MAX', '210'))
FACE_MAX_IMAGE_SIZE = int(os.getenv('FACE_MAX_IMAGE_SIZE', '2048'))
FACE_MAX_UPLOAD_SIZE_MB = get_positive_float_env('FACE_MAX_UPLOAD_SIZE_MB', 2)
FACE_MAX_UPLOAD_SIZE_BYTES = int(FACE_MAX_UPLOAD_SIZE_MB * 1024 * 1024)
DEFAULT_ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/jpg', 'image/png', 'image/webp'}
FACE_ALLOWED_IMAGE_TYPES = {
    content_type.strip().lower()
    for content_type in os.getenv(
        'FACE_ALLOWED_IMAGE_TYPES',
        'image/jpeg,image/jpg,image/png,image/webp'
    ).split(',')
    if content_type.strip()
} or DEFAULT_ALLOWED_IMAGE_TYPES


def init_face_model(model_name: str = "buffalo_l", det_size: int = 640):
    global _face_app
    from insightface.app import FaceAnalysis
    _face_app = FaceAnalysis(
        name=model_name,
        providers=['CPUExecutionProvider'],
        allowed_modules=['detection', 'recognition']
    )
    _face_app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    logger.info(f"InsightFace model '{model_name}' initialized (det_size={det_size})")


def get_face_encoding(image: UploadFile):
    if _face_app is None:
        raise RuntimeError("Face model not initialized. Call init_face_model() first.")

    image_bytes = _read_upload_bytes(image)

    image_rgb = _load_image(image_bytes)
    _validate_quality(image_rgb)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    faces = _face_app.get(image_bgr)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in the image.")
    if len(faces) > 1:
        raise HTTPException(status_code=400, detail="Image must contain exactly one face.")

    embedding = np.asarray(faces[0].embedding, dtype=np.float32)
    return normalize_embedding(embedding)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(embedding))
    if norm < 1e-10:
        raise HTTPException(status_code=400, detail="Invalid embedding returned by model.")
    return embedding / norm


def _read_upload_bytes(image: UploadFile) -> bytes:
    _validate_upload_content_type(image)

    image.file.seek(0)
    image_bytes = image.file.read(FACE_MAX_UPLOAD_SIZE_BYTES + 1)
    image.file.seek(0)

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image payload is empty.")

    if len(image_bytes) > FACE_MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large, maximum is {FACE_MAX_UPLOAD_SIZE_MB:g} MB.",
        )

    return image_bytes


def _validate_upload_content_type(image: UploadFile):
    content_type = (image.content_type or "").split(";")[0].strip().lower()

    if not content_type:
        raise HTTPException(status_code=415, detail="Image content type is required.")

    if content_type not in FACE_ALLOWED_IMAGE_TYPES:
        allowed_types = ", ".join(sorted(FACE_ALLOWED_IMAGE_TYPES))
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image content type '{content_type}'. Allowed: {allowed_types}.",
        )


def _load_image(image_bytes: bytes) -> np.ndarray:
    try:
        pil_image = Image.open(BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Image payload is not a valid image.")

    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert("RGB")

    if max(pil_image.size) > FACE_MAX_IMAGE_SIZE:
        pil_image.thumbnail((FACE_MAX_IMAGE_SIZE, FACE_MAX_IMAGE_SIZE))

    if min(pil_image.size) < FACE_MIN_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Image is too small ({min(pil_image.size)}px), minimum is {FACE_MIN_IMAGE_SIZE}px.",
        )

    return np.asarray(pil_image, dtype=np.uint8)


def _validate_quality(image_rgb: np.ndarray):
    grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
    brightness = float(grayscale.mean())

    if blur_score < FACE_BLUR_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail=f"Image is too blurry (score: {blur_score:.2f}), minimum is {FACE_BLUR_THRESHOLD:.2f}.",
        )

    if brightness < FACE_BRIGHTNESS_MIN or brightness > FACE_BRIGHTNESS_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"Image lighting is out of acceptable range ({brightness:.2f}).",
        )
