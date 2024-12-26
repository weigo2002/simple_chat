from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

def get_vector_store(client: QdrantClient, collection_name: str, embedding_model: str) -> Qdrant:
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=hf_embeddings,
    )
    return vector_store