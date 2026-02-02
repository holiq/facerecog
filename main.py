from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from database import get_db, FaceEntity, engine
from utils import get_face_encoding
import numpy as np
import json
import logging
import os
from dotenv import load_dotenv
from typing import Optional
from datetime import datetime

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FACE_MATCH_THRESHOLD = float(os.getenv('FACE_MATCH_THRESHOLD', '0.6'))

app = FastAPI(title="Face Recognition API", version="1.0.0")

class FaceCache:
    def __init__(self):
        self.faces: Optional[list] = None
        self.encodings_matrix: Optional[np.ndarray] = None
        self.names_map: Optional[list] = None
        self.last_updated: Optional[datetime] = None
    
    def load_from_db(self, db: Session):
        """Load semua faces dari database ke cache"""
        logger.info("Loading faces from database to cache...")
        faces = db.query(FaceEntity).all()
        
        if not faces:
            self.faces = []
            self.encodings_matrix = None
            self.names_map = []
            self.last_updated = datetime.now()
            return
        
        all_encodings = []
        names_map = []
        
        for face in faces:
            descriptors = json.loads(face.descriptor)
            for encoding in descriptors:
                all_encodings.append(encoding)
                names_map.append(face.name)
        
        # Pre-convert ke numpy array untuk vectorized operations
        self.encodings_matrix = np.array(all_encodings)
        self.names_map = names_map
        self.faces = faces
        self.last_updated = datetime.now()
        
        logger.info(f"Cache loaded: {len(faces)} people, {len(all_encodings)} encodings")
    
    def invalidate(self):
        """Hapus cache (panggil setelah register/update)"""
        self.faces = None
        self.encodings_matrix = None
        self.names_map = None
        self.last_updated = None
        logger.info("Cache invalidated")
    
    def is_valid(self) -> bool:
        return self.encodings_matrix is not None
    
    def get_stats(self) -> dict:
        if not self.is_valid():
            return {"cached": False}
        return {
            "cached": True,
            "total_people": len(set(self.names_map)) if self.names_map else 0,
            "total_encodings": len(self.names_map) if self.names_map else 0,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }

face_cache = FaceCache()

@app.get("/health")
async def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        raise HTTPException(status_code=503)
    return Response(status_code=200)

@app.post("/face-recognition/register")
async def register_faces(images: list[UploadFile] = File(...), name: str = Form(...), db: Session = Depends(get_db)):
    name = name.strip()
    if not name:
        logger.warning("Registration attempted with empty name")
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    if len(name) > 255:
        logger.warning(f"Registration attempted with name exceeding 255 characters: {len(name)}")
        raise HTTPException(status_code=400, detail="Name cannot exceed 255 characters")
    
    descriptors = []

    for image in images:
        try:
            encoding = get_face_encoding(image)
            descriptors.append(encoding.tolist())
            logger.info(f"Face encoding extracted from {image.filename}")
        except HTTPException as e:
            logger.error(f"Face not detected in {image.filename}")
            raise HTTPException(status_code=400, detail=f"Face not detected in {image.filename}")

    existing_face = db.query(FaceEntity).filter(FaceEntity.name == name).first()

    if existing_face:
        existing_face.descriptor = json.dumps(descriptors)
        db.commit()
        face_cache.invalidate()  # Invalidate cache setelah update
        logger.info(f"Face updated for {name} with {len(descriptors)} images")
        return {"message": "Face updated successfully!", "name": name, "images_count": len(descriptors)}
    else:
        new_face = FaceEntity(name=name, descriptor=json.dumps(descriptors))
        db.add(new_face)
        db.commit()
        face_cache.invalidate()  # Invalidate cache setelah register baru
        logger.info(f"New face registered for {name} with {len(descriptors)} images")
        return {"message": "Face registered successfully!", "name": name, "images_count": len(descriptors)}

@app.post("/face-recognition/predict")
async def recognize_face(image: UploadFile = File(...), db: Session = Depends(get_db)):
    logger.info(f"Face recognition request received for {image.filename}")
    
    try:
        encoding = get_face_encoding(image)
    except HTTPException as e:
        logger.error(f"Face not detected in uploaded image: {image.filename}")
        raise HTTPException(status_code=400, detail="No face detected in the uploaded image")
    
    if not face_cache.is_valid():
        face_cache.load_from_db(db)
    
    if face_cache.encodings_matrix is None or len(face_cache.names_map) == 0:
        logger.warning("No registered faces in the system")
        return {"message": "No registered faces in the system."}
    
    distances = np.linalg.norm(face_cache.encodings_matrix - encoding, axis=1)
    
    min_index = np.argmin(distances)
    best_match_distance = float(distances[min_index])
    best_match_name = face_cache.names_map[min_index]
    
    if best_match_distance < FACE_MATCH_THRESHOLD:
        logger.info(f"Face recognized as {best_match_name} with distance {best_match_distance}")
        return {"message": "Face recognized", "match": best_match_name, "distance": best_match_distance}
    else:
        logger.info(f"No match found. Best distance was {best_match_distance}")
        return {"message": "No match found!"}
