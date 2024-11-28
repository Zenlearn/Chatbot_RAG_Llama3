from qdrant_client import QdrantClient
from config.config_env import QDRANT_HOST, QDRANT_API_KEY

qdrant = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY,
)
