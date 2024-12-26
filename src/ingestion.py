import os.path
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import SpacyTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from config import CONFIG
from vector_store import get_vector_store


def create_collection(collection_name: str, client: QdrantClient):
    if not client.collection_exists(collection_name):
        vector_config = VectorParams(
            size=512,
            distance=Distance.COSINE
        )
        client.create_collection(collection_name, vector_config)

def process_pdf(file_path: str) -> list[str]:
    pdf_loader = PyPDFLoader(file_path, extract_images=False)
    text_splitter = SpacyTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        pipeline="zh_core_web_sm",
    )
    pdf_contents = pdf_loader.load()
    pdf_text = "\n".join([page.page_content for page in pdf_contents])
    print(f"PDF文档的总字符数: {len(pdf_text)}")

    chunks = text_splitter.split_text(pdf_text)
    print(f"分割的文本chunk数量: {len(chunks)}")

    return chunks

if __name__ == "__main__":
    model_name = CONFIG["llm"]
    embedding_model = CONFIG["embedding_model"]
    collection_name = CONFIG["collection_name"]

    client = QdrantClient(
        url=CONFIG["qdrant_host"],
        api_key=CONFIG["qdrant_api_key"],
    )

    create_collection(collection_name, client)
    print(f"Collection created: {collection_name}")

    vector_store = get_vector_store(client, collection_name, embedding_model)
    abs_path = os.path.abspath("海尔智家股份有限公司2024年第三季度报告.PDF")
    chunks = process_pdf(abs_path)
    print("start adding vectors")
    vector_store.add_texts(chunks)
    print("Vectors added")
