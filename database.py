from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import uuid
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(os.getenv('DATABASE_URL', 'mysql+pymysql://root@localhost/db_name'))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class FaceEntity(Base):
    __tablename__ = os.getenv('TABLE_NAME', 'face_entity')
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), index=True)
    descriptor = Column(Text) 

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
