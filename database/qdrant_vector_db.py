import uuid
from qdrant_client.http.models import PointStruct,Filter, FieldCondition, Match
from config.config_qdrant import qdrant
from qdrant_client.models import Distance, VectorParams
from config.config_env import QDRANT_COLLECTION_NAME, VECTOR_QUERY_SIZE
from services.embedding import generate_embedding

# check if the collection exists otherwise create it
try:
    qdrant.get_collection(collection_name=QDRANT_COLLECTION_NAME)
except Exception:
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=generate_embedding(["test"]).shape[-1], distance=Distance.COSINE
        ),
    )


def add_document(chanks: list[str], vectors: list[list[float]], priority: int) -> dict:
    """
    Add a document to the Qdrant database.
    """
    if len(chanks) != len(vectors):
        raise ValueError("Number of chunks and vectors should be the same.")

    points = []
    doc_id = uuid.uuid4().hex
    chunks_ids = []
    for idx, vector in enumerate(vectors):
        id = uuid.uuid4().hex
        chunks_ids.append(id)
        points.append(
            PointStruct(
                id=id,
                vector=vector,
                payload={
                    "content": chanks[idx],
                    "doc_id": doc_id,
                    "chunk_id": idx,
                    "priority": priority,
                },
            )
        )
    qdrant.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)
    return {"doc_id": doc_id, "chunks_ids": chunks_ids}


def query_db(query_embedding: list[float]) -> list[str]:
    """
    Query the Qdrant database for similar vectors.
    """
    try:
        results = qdrant.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=VECTOR_QUERY_SIZE,
        )
        if not results:
            return []

        return [
            result.payload["content"]
            for result in sorted(
                results,
                key=lambda x: (
                    x.payload.get("priority", 1),
                    -x.score,
                ),  # Ascending priority, descending score
                # reverse=True,
            )
        ]
    except Exception as e:
        raise RuntimeError(f"Error querying the database: {e}")


def delete_document(doc_id):
    """
    Delete all chunks of a document from the Qdrant database.
    """
    try:
        # const = 100
        # results = qdrant.search(
        #     collection_name=QDRANT_COLLECTION_NAME,
        #     query_vector=[0.0] * generate_embedding(["test"]).shape[-1],  # Dummy vector
        #     limit=const,
        #     query_filter={
        #         "must": [
        #             {
        #                 "key": "doc_id",  # Metadata field for filtering
        #                 "match": {"value": doc_id},
        #             }
        #         ]
        #     },
        # )
        # print(results)


        results = qdrant.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=Match(value=doc_id))]
            ),
        )

        return {
            "doc_id": doc_id,
            "chunks_deleted": len(results),
            "chunks": results,
        }
    except Exception as e:
        raise RuntimeError(f"Error querying the database: {e}")
