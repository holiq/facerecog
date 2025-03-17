# Face Recognition API

This is a FastAPI-based face recognition API that allows users to register and recognize faces using uploaded images.

## Features
- Register multiple images for a user
- Update registered faces
- Predict the most similar face from the database

## Requirements
- Docker & Docker Compose
- MySQL

## Setup & Run
### 1. Configure Environment Variables
Create a `.env` file in the project root with the following content:

```
DATABASE_URL=mysql+mysqldb://<user>:<password>@<host>[:<port>]/<dbname>
TABLE_NAME="face_entities"
```

### 2. Build & Run the Container
Run the following command to start the application:

```sh
docker compose up --build
```

The FastAPI server will be available at `http://localhost:8000`

### 3. API Endpoints
#### Register Face
```http
POST /face-recognition/register
```
**Request Body (multipart/form-data):**
- `images`: List of image files
- `name`: User's name

#### Recognize Face
```http
POST /face-recognition/predict
```
**Request Body (multipart/form-data):**
- `image`: Single image file

## Notes
- Ensure your PostgreSQL database is running and accessible.
- The API will return an error if no face is detected in the uploaded images.

