import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from tqdm import tqdm
import time
from typing import List
import logging

# Configuration
csv_file = "combined_medical_QAs.csv"
chunk_size = 100
QDRANT_COLLECTION_NAME = "medical_qa_specific"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en" 

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_qdrant():
    """Initialize Qdrant client and collection"""
    qdrant = QdrantClient(host="localhost", port=6333, timeout=60)
    
    # Create collection if it doesn't exist
    if QDRANT_COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config={"size": 1024, "distance": "Cosine"},
    )
    return qdrant

def initialize_embeddings():
    """Initialize embedding model with batching support"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs = {
    'batch_size': 8,
    'normalize_embeddings': True
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
        # Read CSV in chunks
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
                # Optionally: add retry logic here
                continue
                
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        logger.info(f"Successfully uploaded {total_processed} records to Qdrant!")
        # Close resources if needed

if __name__ == "__main__":
    main()