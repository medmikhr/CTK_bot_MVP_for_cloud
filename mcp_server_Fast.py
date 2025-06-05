import os
from typing import List
from langchain_chroma import Chroma
from dotenv import load_dotenv
from document_processor_langchain import PERSIST_DIR, embeddings
import logging
from fastmcp import FastMCP

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Получение токена из переменных окружения
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("Не найден токен HF_TOKEN в переменных окружения")

# Инициализация эмбеддингов
# embeddings = HuggingFaceEndpointEmbeddings(
#     repo_id=EMBEDDING_MODEL,
#     huggingfacehub_api_token=HF_TOKEN
# )

# Инициализация векторных хранилищ
vector_stores = {
    "dama": Chroma(collection_name="dama", persist_directory=PERSIST_DIR, embedding_function=embeddings),
    "ctk": Chroma(collection_name="ctk", persist_directory=PERSIST_DIR, embedding_function=embeddings)
}

def search_documents(store: Chroma, query: str, k: int = 5) -> tuple[str, List]:
    """Общая функция для поиска документов в векторном хранилище."""
    retrieved_docs = store.similarity_search(query, k=k)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Создание FastMCP сервера
app = FastMCP()

@app.tool("dama_search")
async def dama_search(query: str) -> dict:
    """Поиск информации о методологии управления данными, стандартах DAMA."""
    if not query:
        return {"error": "Query parameter is required"}
    
    try:
        content, docs = search_documents(vector_stores["dama"], query)
        artifacts = [{"type": "document", "data": doc.dict()} for doc in docs]
        return {"content": content, "artifacts": artifacts}
    except Exception as e:
        logger.error(f"Error in dama_search: {str(e)}")
        return {"error": str(e)}

@app.tool("ctk_search")
async def ctk_search(query: str) -> dict:
    """Поиск информации о технологических решениях и архитектуре систем ЦТК."""
    if not query:
        return {"error": "Query parameter is required"}
    
    try:
        content, docs = search_documents(vector_stores["ctk"], query)
        artifacts = [{"type": "document", "data": doc.dict()} for doc in docs]
        return {"content": content, "artifacts": artifacts}
    except Exception as e:
        logger.error(f"Error in ctk_search: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    app.run() 