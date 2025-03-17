FROM python:3.13.2

WORKDIR /app

RUN apt-get update && apt-get install -y cmake

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
