import uuid
import chromadb
from config.config_env import CHROMA_DB_HOST, CHROMA_DB_PORT, CHROMA_DB_COLLECTION_NAME, VECTOR_QUERY_SIZE

try:
    client = chromadb.HttpClient(host=CHROMA_DB_HOST, port=CHROMA_DB_PORT)
    client.heartbeat()
except:
    raise RuntimeError("ChromaDB is not running")

try:
    collection = client.get_collection(CHROMA_DB_COLLECTION_NAME)
except:
    collection = client.create_collection(CHROMA_DB_COLLECTION_NAME)


def add_document(chanks: list[str], priority: int) -> dict:
    """
    Add a document to the ChromaDB database.
    Automatically generates embeddings for the chunks.
    [Using default model: 'sentence-transformers/paraphrase-MiniLM-L6-v2']
    """
    doc_id = uuid.uuid4().hex
    collection.add(
        ids=[f"Document_{doc_id}_{i}" for i in range(len(chanks))],
        documents=chanks,
        metadatas=[
            {"doc_id": doc_id, "chunk_id": i, "priority": priority}
            for i in range(len(chanks))
        ],
    )
    return {"doc_id": doc_id, "chunks_count": len(chanks)}


def query_db(query: str) -> list[str]:
    """
    Query the ChromaDB database for similar vectors.
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=VECTOR_QUERY_SIZE,
        )
        if not results:
            return []
        # return [
        #     result.payload["content"]
        #     for result in sorted(
        #         results,
        #         key=lambda x: (
        #             x.payload.get("priority", 1),
        #             -x.score,
        #         ),  # Ascending priority, descending score
        #         # reverse=True,
        #     )
        # ]
        # TODO: Implement sorting by priority and distance
        return [result for result in results["documents"][0]]
    except Exception as e:
        raise RuntimeError(f"Error querying the database: {e}")


def delete_document(doc_id):
    """
    Delete all chunks of a document from the ChromaDB database.
    """
    try:
        collection.delete(where={"doc_id": doc_id})

        return {"doc_id": doc_id}
    except Exception as e:
        raise RuntimeError(f"Error querying the database: {e}")
