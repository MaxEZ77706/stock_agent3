# ==== Base image ====
FROM python:3.12-slim

# ==== System deps ====
RUN apt-get update && apt-get install -y git && apt-get clean

# ==== Workdir ====
WORKDIR /app

# ==== Copy code ====
COPY . .

# ==== Install deps ====
RUN pip install --no-cache-dir -r requirements.txt

# ==== DVC setup ====
RUN pip install "dvc[dagshub]" "dvc[gdrive]" "dvc[s3]"

# ==== Entrypoint ====
CMD ["bash"]
