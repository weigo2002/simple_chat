from langchain import hub
from langchain_ollama import OllamaLLM


def get_llm(model_name: str) -> OllamaLLM:
    return OllamaLLM(model=model_name)

def get_prompt_template() -> str:
    return hub.pull("jclemens24/rag-prompt")