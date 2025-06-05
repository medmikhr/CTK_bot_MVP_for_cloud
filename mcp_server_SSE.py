import os
from typing import List
from langchain_chroma import Chroma
from dotenv import load_dotenv
from document_processor_langchain import PERSIST_DIR, embeddings
import logging
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel import NotificationOptions
from mcp.server.streamable_http import streamablehttp_server

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

# Создание MCP сервера
server = Server("vector-search-server")

@server.call_tool()
async def dama_search(name: str, arguments: dict) -> dict:
    """Поиск информации о методологии управления данными, стандартах DAMA."""
    query = arguments.get("query", "")
    if not query:
        return {"error": "Query parameter is required"}
    
    try:
        content, docs = search_documents(vector_stores["dama"], query)
        artifacts = [{"type": "document", "data": doc.dict()} for doc in docs]
        return {"content": content, "artifacts": artifacts}
    except Exception as e:
        logger.error(f"Error in dama_search: {str(e)}")
        return {"error": str(e)}

@server.call_tool()
async def ctk_search(name: str, arguments: dict) -> dict:
    """Поиск информации о технологических решениях и архитектуре систем ЦТК."""
    query = arguments.get("query", "")
    if not query:
        return {"error": "Query parameter is required"}
    
    try:
        content, docs = search_documents(vector_stores["ctk"], query)
        artifacts = [{"type": "document", "data": doc.dict()} for doc in docs]
        return {"content": content, "artifacts": artifacts}
    except Exception as e:
        logger.error(f"Error in ctk_search: {str(e)}")
        return {"error": str(e)}

async def run():
    async with streamablehttp_server(host="0.0.0.0", port=8000) as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="vector-search",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(run()) 