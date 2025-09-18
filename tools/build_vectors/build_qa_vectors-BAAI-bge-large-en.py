import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client.http.models import VectorParams, Distance
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from tqdm import tqdm
import time
from typing import List
import logging
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, "..", "..", "dataset", "combined_medical_QAs.csv")
chunk_size = 100
QDRANT_COLLECTION_NAME = "medical_qa_bge_large_en"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_qdrant():
    """Initialize Qdrant client and collection"""
    qdrant = QdrantClient(host="localhost", port=6333, timeout=60)

    if QDRANT_COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    return qdrant


def initialize_embeddings():
    """Initialize embedding model with batching support"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={
            'batch_size': 8,
            'normalize_embeddings': True,
        }
    )


def embed_texts(embeddings_model, texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts using HuggingFace embeddings"""
    try:
        return embeddings_model.embed_documents(texts)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise


def upsert_chunk(qdrant, embeddings_model, df_chunk: pd.DataFrame, offset: int):
    """Embed and upsert one chunk of the dataframe"""
    try:
        docs = [
            Document(
                page_content=str(row.question),
                metadata={
                    "answer": str(row.answer),
                    "tags": [tag.strip() for tag in str(row.tags).split(",")],
                }
            )
            for row in df_chunk.itertuples()
        ]

        embeddings = embed_texts(embeddings_model, [doc.page_content for doc in docs])

        points = []
        for i, (embedding, doc) in enumerate(zip(embeddings, docs)):
            point = PointStruct(
                id=offset + i,
                vector=embedding,
                payload={
                    "page_content": doc.page_content,
                    "metadata": {
                        "answer": doc.metadata["answer"],
                        "tags": doc.metadata["tags"],
                    }
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
        logger.info(f"Upload finished. Total records processed: {total_processed}")


if __name__ == "__main__":
    main()
