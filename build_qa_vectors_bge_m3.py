import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from tqdm import tqdm
import time
from typing import List
import logging
import os
from dotenv import load_dotenv
load_dotenv()  # load env variables from .env file

print("API Key:", os.getenv("QDRANT_API_KEY"))
# Configuration
csv_file = "combined_medical_QAs.csv"
chunk_size = 100
QDRANT_COLLECTION_NAME = "medical_qa"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Set Qdrant Cloud API key and URL
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Set this in your environment or .env file
QDRANT_URL = os.getenv("QDRANT_URL")  # Replace with your Qdrant cloud URL

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_qdrant():
    """Initialize Qdrant client and collection"""
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    # Create collection if it doesn't exist
    if QDRANT_COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
    return qdrant

def initialize_embeddings():
    """Initialize BGE-M3 embedding model"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'batch_size': 32}
    )

def embed_texts(embeddings_model, texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts using HuggingFace embeddings with query prefix"""
    texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
    try:
        return embeddings_model.embed_documents(texts)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise

def upsert_chunk(qdrant, embeddings_model, df_chunk: pd.DataFrame, offset: int):
    """Embed and upsert one chunk of the dataframe"""
    try:
        questions = df_chunk["question"].tolist()
        embeddings = embed_texts(embeddings_model, questions)

        points = []
        for i, (embedding, row) in enumerate(zip(embeddings, df_chunk.itertuples())):
            point = PointStruct(
                id=offset + i,
                vector=embedding,
                payload={
                    "question": str(row.question),
                    "answer": str(row.answer),
                    "tags": str(row.tags)
                }
            )
            points.append(point)

        operation_info = qdrant.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=points,
            wait=True
        )
        return operation_info
    except Exception as e:
        logger.error(f"Failed to process chunk: {e}")
        raise

def main():
    qdrant = initialize_qdrant()
    embeddings_model = initialize_embeddings()

    offset = 0
    total_processed = 0

    try:
        for chunk in tqdm(pd.read_csv(csv_file, chunksize=chunk_size), desc="Uploading chunks"):
            start_time = time.time()

            try:
                upsert_chunk(qdrant, embeddings_model, chunk, offset)
                processed = len(chunk)
                offset += processed
                total_processed += processed
                elapsed = time.time() - start_time
                logger.info(f"Chunk processed in {elapsed:.2f} seconds, total records: {offset}")

            except Exception as e:
                logger.error(f"Error processing chunk at offset {offset}: {e}")
                continue

    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        logger.info(f"Successfully uploaded {total_processed} records to Qdrant!")

if __name__ == "__main__":
    main()
