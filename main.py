from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile as UF, Form, HTTPException, Depends, Response
from sqlalchemy import text
from sqlalchemy.orm import Session
from database import get_db, FaceEntity
from utils import get_face_encoding, init_face_model, normalize_embedding
import numpy as np
import json
import logging
import os
from dotenv import load_dotenv
from typing import Optional
from datetime import datetime
import redis
from abc import ABC, abstractmethod

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
FACE_MATCH_THRESHOLD = float(os.getenv('FACE_MATCH_THRESHOLD', '0.35'))
CACHE_STRATEGY = os.getenv('CACHE_STRATEGY', 'memory')  # 'redis', 'memory', 'disabled'
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

# Insightface Configuration
INSIGHTFACE_MODEL_NAME = os.getenv('INSIGHTFACE_MODEL_NAME', 'buffalo_l')
INSIGHTFACE_DET_SIZE = int(os.getenv('INSIGHTFACE_DET_SIZE', '640'))

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_face_model(model_name=INSIGHTFACE_MODEL_NAME, det_size=INSIGHTFACE_DET_SIZE)
    yield

app = FastAPI(title="Face Recognition API", version="2.0.0", lifespan=lifespan)

# TODO: will be remove when fix issue upload file schema OAS 3.0 released, ref: https://github.com/fastapi/fastapi/pull/15069
from typing import Annotated
from pydantic import WithJsonSchema
UploadFile = Annotated[UF, WithJsonSchema({"type": "string", "format": "binary"})]

class BaseFaceCache(ABC):
    @abstractmethod
    def load_from_db(self, db: Session) -> dict:
        """Load faces from database and return cache data"""
        pass
    
    @abstractmethod
    def get_cache_data(self) -> Optional[dict]:
        """Get cached data if available"""
        pass
    
    @abstractmethod
    def invalidate(self):
        """Invalidate cache"""
        pass
    
    @abstractmethod
    def get_stats(self) -> dict:
        """Get cache statistics"""
        pass

class InMemoryFaceCache(BaseFaceCache):
    def __init__(self):
        self.cache_data: Optional[dict] = None
        self.last_updated: Optional[datetime] = None
    
    def load_from_db(self, db: Session) -> dict:
        logger.info("[MEMORY] Loading faces from database to cache...")
        faces = db.query(FaceEntity).all()
        
        if not faces:
            cache_data = {
                "encodings_matrix": None,
                "names_map": [],
                "faces_count": 0
            }
        else:
            all_encodings = []
            names_map = []
            
            for face in faces:
                descriptors = json.loads(face.descriptor)
                for encoding in descriptors:
                    all_encodings.append(encoding)
                    names_map.append(face.name)
            
            cache_data = {
                "encodings_matrix": np.array(all_encodings),
                "names_map": names_map,
                "faces_count": len(faces)
            }
        
        self.cache_data = cache_data
        self.last_updated = datetime.now()
        logger.info(f"[MEMORY] Cache loaded: {cache_data['faces_count']} people, {len(cache_data['names_map'])} encodings")
        return cache_data
    
    def get_cache_data(self) -> Optional[dict]:
        return self.cache_data
    
    def invalidate(self):
        self.cache_data = None
        self.last_updated = None
        logger.info("[MEMORY] Cache invalidated")
    
    def get_stats(self) -> dict:
        if not self.cache_data:
            return {"strategy": "memory", "cached": False}
        return {
            "strategy": "memory",
            "cached": True,
            "total_people": self.cache_data["faces_count"],
            "total_encodings": len(self.cache_data["names_map"]),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }

class RedisFaceCache(BaseFaceCache):
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client = None
        self.cache_key = "face_recognition:cache"
        self._connect()
    
    def _connect(self):
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
            # Test connection
            self.redis_client.ping()
            logger.info(f"[REDIS] Connected to {self.redis_url}")
        except Exception as e:
            logger.error(f"[REDIS] Failed to connect: {e}")
            self.redis_client = None
    
    def load_from_db(self, db: Session) -> dict:
        if not self.redis_client:
            logger.warning("[REDIS] Redis not available, loading fresh from DB")
            return self._load_fresh_from_db(db)
        
        logger.info("[REDIS] Loading faces from database to Redis cache...")
        faces = db.query(FaceEntity).all()
        
        if not faces:
            cache_data = {
                "encodings_matrix": None,
                "names_map": [],
                "faces_count": 0,
                "last_updated": datetime.now().isoformat()
            }
        else:
            all_encodings = []
            names_map = []
            
            for face in faces:
                descriptors = json.loads(face.descriptor)
                for encoding in descriptors:
                    all_encodings.append(encoding)
                    names_map.append(face.name)
            
            cache_data = {
                "encodings_list": all_encodings,  # Store as list for JSON serialization
                "names_map": names_map,
                "faces_count": len(faces),
                "last_updated": datetime.now().isoformat()
            }
        
        try:
            # Store to Redis
            self.redis_client.set(self.cache_key, json.dumps(cache_data), ex=3600)  # 1 hour TTL
            logger.info(f"[REDIS] Cache stored: {cache_data['faces_count']} people, {len(cache_data['names_map'])} encodings")
        except Exception as e:
            logger.error(f"[REDIS] Failed to store cache: {e}")
        
        # Convert encodings_list to numpy array for return
        if cache_data.get("encodings_list"):
            cache_data["encodings_matrix"] = np.array(cache_data["encodings_list"])
        else:
            cache_data["encodings_matrix"] = None
        if "encodings_list" in cache_data:
            del cache_data["encodings_list"]
        
        return cache_data
    
    def _load_fresh_from_db(self, db: Session) -> dict:
        """Fallback: load fresh from DB when Redis unavailable"""
        faces = db.query(FaceEntity).all()
        if not faces:
            return {"encodings_matrix": None, "names_map": [], "faces_count": 0}
        
        all_encodings = []
        names_map = []
        for face in faces:
            descriptors = json.loads(face.descriptor)
            for encoding in descriptors:
                all_encodings.append(encoding)
                names_map.append(face.name)
        
        return {
            "encodings_matrix": np.array(all_encodings),
            "names_map": names_map,
            "faces_count": len(faces)
        }
    
    def get_cache_data(self) -> Optional[dict]:
        if not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(self.cache_key)
            if not cached:
                return None
            
            cache_data = json.loads(cached)
            # Convert encodings_list back to numpy array
            if cache_data.get("encodings_list"):
                cache_data["encodings_matrix"] = np.array(cache_data["encodings_list"])
                del cache_data["encodings_list"]
            else:
                cache_data["encodings_matrix"] = None
            
            return cache_data
        except Exception as e:
            logger.error(f"[REDIS] Failed to get cache: {e}")
            return None
    
    def invalidate(self):
        if not self.redis_client:
            return
        
        try:
            self.redis_client.delete(self.cache_key)
            logger.info("[REDIS] Cache invalidated")
        except Exception as e:
            logger.error(f"[REDIS] Failed to invalidate cache: {e}")
    
    def get_stats(self) -> dict:
        if not self.redis_client:
            return {"strategy": "redis", "cached": False, "redis_available": False}
        
        try:
            cached = self.redis_client.get(self.cache_key)
            if not cached:
                return {"strategy": "redis", "cached": False, "redis_available": True}
            
            cache_data = json.loads(cached)
            return {
                "strategy": "redis",
                "cached": True,
                "redis_available": True,
                "total_people": cache_data.get("faces_count", 0),
                "total_encodings": len(cache_data.get("names_map", [])),
                "last_updated": cache_data.get("last_updated")
            }
        except Exception as e:
            return {"strategy": "redis", "cached": False, "redis_available": False, "error": str(e)}

class NoCache(BaseFaceCache):
    def load_from_db(self, db: Session) -> dict:
        logger.info("[NO_CACHE] Loading faces directly from database...")
        faces = db.query(FaceEntity).all()
        
        if not faces:
            return {"encodings_matrix": None, "names_map": [], "faces_count": 0}
        
        all_encodings = []
        names_map = []
        
        for face in faces:
            descriptors = json.loads(face.descriptor)
            for encoding in descriptors:
                all_encodings.append(encoding)
                names_map.append(face.name)
        
        cache_data = {
            "encodings_matrix": np.array(all_encodings),
            "names_map": names_map,
            "faces_count": len(faces)
        }
        
        logger.info(f"[NO_CACHE] Loaded: {cache_data['faces_count']} people, {len(cache_data['names_map'])} encodings")
        return cache_data
    
    def get_cache_data(self) -> Optional[dict]:
        return None  # Always return None to force fresh load
    
    def invalidate(self):
        logger.info("[NO_CACHE] Nothing to invalidate")
    
    def get_stats(self) -> dict:
        return {"strategy": "disabled", "cached": False}


def create_cache() -> BaseFaceCache:
    if CACHE_STRATEGY.lower() == 'redis':
        return RedisFaceCache(REDIS_URL)
    elif CACHE_STRATEGY.lower() == 'memory':
        return InMemoryFaceCache()
    elif CACHE_STRATEGY.lower() == 'disabled':
        return NoCache()
    else:
        logger.warning(f"Unknown cache strategy '{CACHE_STRATEGY}', defaulting to memory")
        return InMemoryFaceCache()

# Global cache instance
face_cache = create_cache()
logger.info(f"Cache strategy initialized: {CACHE_STRATEGY.upper()}")

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
    
    # Get cache data from the selected cache strategy
    cache_data = face_cache.get_cache_data()
    
    # Load from DB if cache is empty
    if cache_data is None:
        cache_data = face_cache.load_from_db(db)
    
    # Check if there are any registered faces
    if cache_data["encodings_matrix"] is None or len(cache_data["names_map"]) == 0:
        logger.warning("No registered faces in the system")
        return {"message": "No registered faces in the system."}
    
    # VECTORIZED NUMPY OPERATION — Cosine similarity
    # Normalize query
    encoding = normalize_embedding(encoding)
    # Normalize matrix rows (handles unnormalized old data safely)
    matrix = cache_data["encodings_matrix"]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    matrix = matrix / norms

    scores = np.dot(matrix, encoding)
    max_index = np.argmax(scores)
    best_score = float(scores[max_index])
    best_match_name = cache_data["names_map"][max_index]
    
    if best_score > FACE_MATCH_THRESHOLD:
        logger.info(f"Face recognized as {best_match_name} with similarity {best_score:.4f}")
        return {"message": "Face recognized", "match": best_match_name, "similarity": best_score}
    else:
        logger.info(f"No match found. Best similarity was {best_score:.4f}")
        return {"message": "No match found!"}
