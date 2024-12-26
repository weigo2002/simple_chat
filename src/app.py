import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from qdrant_client import QdrantClient
from config import CONFIG
from llm import get_llm, get_prompt_template
from vector_store import get_vector_store

st.title("Simple Chat App")

def get_retriever() -> VectorStoreRetriever:
    client = QdrantClient(
        url=CONFIG["qdrant_host"],
        api_key=CONFIG["qdrant_api_key"],
    )

    vector_store = get_vector_store(client, CONFIG["collection_name"], CONFIG["embedding_model"])
    return vector_store.as_retriever()

def generate_response(input_text: str, chain):
    print(input_text)
    result = chain.invoke(input_text)
    if result and result != "":
        st.info(result)
    else:
        st.info("未找到需要的信息")

with st.form("Chat Dialog"):
    text = st.text_area(
        "Enter a message:"
    )
    submitted = st.form_submit_button("Submit")
    llm = get_llm(CONFIG["llm"])
    prompt = get_prompt_template()
    retriever = get_retriever()

    rag_chain = ({
                     "context": retriever,
                     "question": RunnablePassthrough()
                 } | prompt
                 | llm
                 | StrOutputParser())
    if submitted and text and text != "":
        generate_response(text, rag_chain)