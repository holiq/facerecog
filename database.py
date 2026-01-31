from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import uuid
from dotenv import load_dotenv
import os
import pymysql

pymysql.install_as_MySQLdb()

load_dotenv()

engine = create_engine(
    os.getenv('DATABASE_URL', 'mysql+pymysql://root@localhost/db_name'),
    pool_size=10,          # Jumlah koneksi tetap di pool
    max_overflow=20,       # Koneksi tambahan saat traffic tinggi
    pool_timeout=30,       # Timeout menunggu koneksi (detik)
    pool_recycle=1800,     # Recycle koneksi setiap 30 menit
    pool_pre_ping=True     # Test koneksi sebelum digunakan
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class FaceEntity(Base):
    __tablename__ = os.getenv('TABLE_NAME', 'face_entity')
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), unique=True, index=True)
    descriptor = Column(Text) 

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
