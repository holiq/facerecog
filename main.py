from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from database import get_db, FaceEntity
from utils import get_face_encoding
import numpy as np

app = FastAPI()

@app.post("/face-recognition/register")
async def register_faces(images: list[UploadFile] = File(...), name: str = Form(...), db: Session = Depends(get_db)):
    descriptors = []

    for image in images:
        try:
            encoding = get_face_encoding(image)
            descriptors.append(encoding.tolist())
        except HTTPException:
            return {"error": f"Face not detected in {image.filename}"}

    existing_face = db.query(FaceEntity).filter(FaceEntity.name == name).first()

    if existing_face:
        existing_face.descriptor = str(descriptors)
        db.commit()
        return {"message": "Face updated successfully!", "name": name, "images_count": len(descriptors)}
    else:
        new_face = FaceEntity(name=name, descriptor=str(descriptors))
        db.add(new_face)
        db.commit()
        return {"message": "Face registered successfully!", "name": name, "images_count": len(descriptors)}

@app.post("/face-recognition/predict")
async def recognize_face(image: UploadFile = File(...), db: Session = Depends(get_db)):
    encoding = get_face_encoding(image)

    faces = db.query(FaceEntity).all()
    if not faces:
        return {"message": "No registered faces in the system."}

    best_match_name = None
    best_match_distance = float("inf")

    for face in faces:
        stored_descriptors = eval(face.descriptor)  
        for stored_encoding in stored_descriptors:
            distance = np.linalg.norm(np.array(stored_encoding) - encoding)
            if distance < best_match_distance:
                best_match_distance = distance
                best_match_name = face.name

    if best_match_distance < 0.6:
        return {"message": "Face recognized", "match": best_match_name, "distance": best_match_distance}
    else:
        return {"message": "No match found!"}

