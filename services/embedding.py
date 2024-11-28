from sentence_transformers import SentenceTransformer
from config.config_env import EMBEDDING_MODEL
import numpy

model = SentenceTransformer(EMBEDDING_MODEL)


def generate_embedding(chunks: list[str]) -> numpy.ndarray:
    """
    Generate embeddings for a list of text chunks.
    """
    return model.encode(chunks, convert_to_numpy=True)
