import face_recognition
import numpy as np
from PIL import Image
import io
from fastapi import HTTPException
from fastapi import UploadFile

def get_face_encoding(image: UploadFile):
    image_bytes = image.file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = pil_image.convert("RGB")
    image_np = np.array(pil_image)
    
    face_encodings = face_recognition.face_encodings(image_np)
    
    if not face_encodings:
        raise HTTPException(status_code=400, detail="No face detected in the image.")
    
    return face_encodings[0]
