# Face Recognition API

FastAPI service for registering and recognizing faces from uploaded images. The
service extracts face embeddings with InsightFace, stores them in MySQL, and
matches incoming faces with cosine similarity.

## Stack

- FastAPI and Uvicorn
- InsightFace with ONNX Runtime CPU provider
- MySQL through SQLAlchemy and PyMySQL
- Optional Redis cache
- Docker image built with `uv`

## How It Works

1. `POST /face-recognition/register` accepts a `name` and one or more face
   images.
2. Each image is validated for size, blur, lighting, and must contain exactly
   one face.
3. The service stores normalized face embeddings as JSON in the configured
   MySQL table.
4. `POST /face-recognition/predict` extracts one embedding from the uploaded
   image, compares it against registered embeddings, and returns a match when
   the similarity is above `FACE_MATCH_THRESHOLD`.

Registering an existing `name` replaces that person's stored descriptors.

## Quick Start

The application requires an accessible MySQL database. Docker Compose in this
repository starts only the API container, not MySQL or Redis.

### Run the Published Docker Image

```bash
docker run -d \
  --name facerecog-api \
  -p 8000:8000 \
  -e DATABASE_URL="mysql+pymysql://user:password@host:3306/database_name" \
  -e TABLE_NAME="face_entities" \
  ghcr.io/holiq/facerecog/app:latest
```

### Run with Docker Compose

```bash
cp .env.example .env
# Edit DATABASE_URL and other settings in .env

docker compose up --build
```

For production with the published image:

```bash
docker compose -f docker-compose.prod.yml up -d
```

### Run Locally

Use Python 3.12 or newer.

```bash
cp .env.example .env
# Edit DATABASE_URL and other settings in .env

uv sync --locked
uv run uvicorn main:app --reload
```

If you are not using `uv`, install the dependencies from `pyproject.toml` with
your preferred Python package manager, then run:

```bash
uvicorn main:app --reload
```

## Configuration

Configuration is loaded from environment variables. See `.env.example` for a
ready-to-edit template.

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | `mysql+pymysql://root@localhost/db_name` | SQLAlchemy MySQL connection URL. Set this explicitly outside local experiments. |
| `TABLE_NAME` | `face_entity` | Database table used for registered faces. `.env.example` uses `face_entities`. |
| `PORT` | `8000` | Host port used by Docker Compose. |
| `WORKERS` | `1` | Number of Uvicorn workers used by `entrypoint.sh`. |
| `LOG_LEVEL` | `info` | Uvicorn log level. |
| `FACE_MATCH_THRESHOLD` | `0.35` | Minimum cosine similarity required for a match. Higher is stricter. |
| `FACE_MIN_IMAGE_SIZE` | `100` | Minimum image side length in pixels. |
| `FACE_MAX_IMAGE_SIZE` | `2048` | Maximum image side length before resizing. |
| `FACE_BLUR_THRESHOLD` | `50` | Minimum Laplacian blur score. Higher rejects more blurry images. |
| `FACE_BRIGHTNESS_MIN` | `40` | Minimum grayscale brightness average. |
| `FACE_BRIGHTNESS_MAX` | `210` | Maximum grayscale brightness average. |
| `FACE_MAX_UPLOAD_SIZE_MB` | `2` | Maximum size for each uploaded image. |
| `FACE_MAX_REQUEST_SIZE_MB` | `10` | Maximum request body size for upload endpoints, based on `Content-Length`. |
| `FACE_ALLOWED_IMAGE_TYPES` | `image/jpeg,image/jpg,image/png,image/webp` | Comma-separated allowed upload MIME types. |
| `INSIGHTFACE_MODEL_NAME` | `buffalo_l` | InsightFace model name. Common options: `buffalo_l`, `buffalo_m`, `buffalo_s`. |
| `INSIGHTFACE_DET_SIZE` | `640` | Detection input size used as `(value, value)`. |
| `CACHE_STRATEGY` | `memory` | Cache mode: `memory`, `redis`, or `disabled`. |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL when `CACHE_STRATEGY=redis`. |

## API

Interactive OpenAPI documentation is available at:

```text
http://localhost:8000/docs
```

### Health Check

```http
GET /health
```

Returns HTTP `200` when the API can query the database. Returns HTTP `503` when
the database check fails.

### Register Face

```http
POST /face-recognition/register
Content-Type: multipart/form-data
```

Form fields:

| Field | Type | Description |
| --- | --- | --- |
| `name` | string | Person name. Must be non-empty and at most 255 characters. |
| `images` | file[] | One or more images. Each image must contain exactly one face. |

Example:

```bash
curl -X POST "http://localhost:8000/face-recognition/register" \
  -F "name=Alice" \
  -F "images=@alice-1.jpg;type=image/jpeg" \
  -F "images=@alice-2.jpg;type=image/jpeg"
```

Success response:

```json
{
  "message": "Face registered successfully!",
  "name": "Alice",
  "images_count": 2
}
```

### Recognize Face

```http
POST /face-recognition/predict
Content-Type: multipart/form-data
```

Form fields:

| Field | Type | Description |
| --- | --- | --- |
| `image` | file | Image containing exactly one face. |

Example:

```bash
curl -X POST "http://localhost:8000/face-recognition/predict" \
  -F "image=@unknown.jpg;type=image/jpeg"
```

Matched response:

```json
{
  "message": "Face recognized",
  "match": "Alice",
  "similarity": 0.72
}
```

No-match response:

```json
{
  "message": "No match found!"
}
```

## Cache Behavior

- `memory`: embeddings are loaded from MySQL on first prediction and kept in the
  process memory until a registration updates or invalidates the cache.
- `redis`: embeddings are stored in Redis with a one-hour TTL and invalidated
  after registration updates.
- `disabled`: every prediction loads embeddings directly from MySQL.

For deployments with `WORKERS > 1`, prefer `CACHE_STRATEGY=redis` so all worker
processes share the same cache state.

Example multi-worker cache configuration:

```bash
WORKERS=4
CACHE_STRATEGY=redis
REDIS_URL=redis://redis-host:6379
```

## Image Requirements

Uploaded images are normalized with EXIF orientation support and converted to
RGB. An image is rejected when it:

- is not a valid image payload
- has a MIME type outside `FACE_ALLOWED_IMAGE_TYPES`
- is larger than `FACE_MAX_UPLOAD_SIZE_MB`
- is part of a request body larger than `FACE_MAX_REQUEST_SIZE_MB`
- is smaller than `FACE_MIN_IMAGE_SIZE`
- is too blurry
- is too dark or too bright
- contains no face
- contains more than one face

## Database Schema

The service creates the configured table automatically at startup.

| Column | Type | Description |
| --- | --- | --- |
| `id` | string UUID | Primary key. |
| `name` | string | Unique person name. |
| `descriptor` | text | JSON array of normalized embedding vectors. |

## Notes

- The default runtime uses the CPU provider for ONNX Runtime.
- The Docker image does not include MySQL or Redis.
- Keep `FACE_MATCH_THRESHOLD` tuned for your dataset. Higher values reduce false
  positives but may increase false negatives.
- There is currently no project-level automated test suite in this repository.
