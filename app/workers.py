from celery import Celery
from redisconfig import settings
import pandas as pd
import uuid
import os
from typing import Tuple
import logging
import vector_db
import zipfile

# Initialize Celery
app = Celery(
    "tasks",
    broker=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
    backend=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB + 1}",
)

# Celery configuration
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    broker_connection_retry_on_startup=True,
    task_track_started=True,
)

logger = logging.getLogger(__name__)

@app.task(bind=True, max_retries=3)
def process_csv_task(self, file_path: str, filename: str) -> Tuple[bool, str]:
    try:
        df = pd.read_csv(file_path)

        if "clause_text" not in df.columns:
            raise ValueError("Missing 'clause_text' column")

        # Check existing data
        existing = vector_db.get(
            where={"source_file": filename},
            limit=1
        )
        if existing and existing['ids']:
            logger.info(f"Embeddings from '{filename}' already exist. Skipping.")
            if os.path.exists(file_path):
                os.remove(file_path)
            return False, f"⚠️ Embeddings from '{filename}' already exist. Skipping."

        df = df.dropna(subset=["clause_text"])
        texts = df["clause_text"].tolist()
        types = df.get("clause_type", ["Unknown"] * len(texts)).tolist()

        batch_size = 100
        total = len(texts)

        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_types = types[i:i + batch_size]

            vector_db.add(
                ids=[str(uuid.uuid4()) for _ in batch_texts],
                documents=batch_texts,
                metadatas=[{"type": t, "source_file": filename} for t in batch_types]
            )

            self.update_state(
                state="PROGRESS",
                meta={
                    "current": min(i + batch_size, total),
                    "total": total,
                    "filename": filename
                }
            )

        os.remove(file_path)
        return True, f"✅ {len(texts)} clauses from '{filename}' added successfully"

    except Exception as e:
        logger.error(f"Failed processing {filename}: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise self.retry(exc=e, countdown=60)

@app.task(bind=True)
def process_zip_task(self, file_path: str, filename: str):
    try:
        results = []
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]

            if not csv_files:
                raise ValueError("No CSV files found in ZIP")

            temp_dir = os.path.join("uploads", "extracted_csvs")
            os.makedirs(temp_dir, exist_ok=True)
            zip_ref.extractall(temp_dir)

            for csv_file in csv_files:
                csv_path = os.path.join(temp_dir, csv_file)
                csv_task = process_csv_task.delay(csv_path, os.path.basename(csv_file))
                results.append(csv_task.id)

        os.remove(file_path)
        return {
            "status": "started",
            "task_ids": results,
            "message": f"Processing {len(results)} CSV files from {filename}"
        }

    except Exception as e:
        logger.error(f"Failed processing ZIP {filename}: {str(e)}")
        raise self.retry(exc=e, countdown=60)

