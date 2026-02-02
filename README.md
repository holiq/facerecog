# Face Recognition API

A production-ready FastAPI-based face recognition API that allows users to register and recognize faces using uploaded images.

## 🚀 Quick Start

### Production Deployment (Recommended)

Pull the pre-built image from GitHub Container Registry:

```bash
# Pull the latest image
docker pull ghcr.io/holiq/facerecog/app:latest

# Simple run
docker run -d \
  --name facerecog-api \
  -p 8000:8000 \
  -e DATABASE_URL="mysql+pymysql://user:password@host:port/dbname" \
  ghcr.io/holiq/facerecog/app:latest
```

### Production Docker Compose

Use the production compose file:

```bash
# Download the production compose file
wget https://raw.githubusercontent.com/holiq/facerecog/main/docker-compose.prod.yml

# Configure your environment
cp .env.example .env
# Edit .env with your settings

# Start the application
docker compose -f docker-compose.prod.yml up -d
```

### Development Setup

For local development and contributions:

```bash
# Clone the repository
git clone https://github.com/holiq/facerecog.git
cd facerecog

# Copy environment file
cp .env.example .env
# Edit .env with your settings

# Start development environment
docker compose up --build
```

## 🔧 Configuration

### Environment Variables

| Variable               | Default           | Description                                               |
| ---------------------- | ----------------- | --------------------------------------------------------- |
| `PORT`                 | `8000`            | Server port                                               |
| `HOST`                 | `0.0.0.0`         | Server host                                               |
| `WORKERS`              | `1`               | Number of uvicorn workers                                 |
| `LOG_LEVEL`            | `info`            | Logging level (debug, info, warning, error)               |
| `FACE_MATCH_THRESHOLD` | `0.6`             | Face recognition threshold (0.0-1.0, lower = more strict) |
| `DATABASE_URL`         | -                 | Database connection string (required)                     |
| `TABLE_NAME`           | `"face_entities"` | Database table name                                       |

### Database URL Examples

```bash
# MySQL
DATABASE_URL=mysql+pymysql://username:password@hostname:3306/database_name

# SQLite (for development only)
DATABASE_URL=sqlite:///./facerecog.db
```

## 📚 API Documentation

The API documentation is automatically generated and available at:

- **Swagger UI**: `http://localhost:8000/docs`

### Key Endpoints

#### Health Check

```http
GET /health
```

#### Register Face

```http
POST /face-recognition/register
```

**Body (multipart/form-data):**

- `images`: List of image files
- `name`: User's name

#### Recognize Face

```http
POST /face-recognition/predict
```

**Body (multipart/form-data):**

- `image`: Single image file

## 🏗️ Development

### Local Development

```bash
# Clone repository
git clone https://github.com/holiq/facerecog.git
cd facerecog

# Build and run locally
docker compose up --build

# Or run with Python
pip install -r requirements.txt
uvicorn main:app --reload
```

## 🔒 Security Features

- ✅ Non-root user in container
- ✅ Minimal base image (python-slim)
- ✅ Health checks included
- ✅ Production-ready configuration
- ✅ Environment-based configuration

## 📝 Notes

- Ensure your database is running and accessible
- The API will return an error if no face is detected in uploaded images
- Face recognition threshold can be adjusted via `FACE_MATCH_THRESHOLD` environment variable
- For production use, always use the pre-built images from GitHub Container Registry
