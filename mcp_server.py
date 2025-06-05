import os
import asyncio
import socket
import json
import logging
from typing import List, Dict, Set, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from datetime import datetime, timedelta
import hashlib
from langchain.schema import Document
import traceback
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from mcp.server.fastmcp import FastMCPServer
from mcp.tool import tool

# Конфигурация
PERSIST_DIR = "chroma_db"  # Директория для хранения базы данных Chroma
EMBEDDING_MODEL = "cointegrated/rubert-tiny2"  # Модель для эмбеддингов
CHUNK_SIZE = 1000  # Размер чанка при разбиении текста
CHUNK_OVERLAP = 200  # Перекрытие между чанками
MCP_HOST = "0.0.0.0"
MCP_PORT = 8000

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(title="Document Processor MCP Server")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'}
)

# Создаем словарь для хранения векторных хранилищ для разных коллекций
vectorstores = {}

# Pydantic модели для запросов и ответов
class SearchRequest(BaseModel):
    query: str
    collection: str
    n_results: Optional[int] = 5

class DocumentInfoRequest(BaseModel):
    collection: Optional[str] = None

class DeleteDocumentRequest(BaseModel):
    document_id: str
    collection: str

class MCPResponse(BaseModel):
    status: str
    data: Optional[Dict] = None
    error: Optional[str] = None

def get_vectorstore(collection: str) -> Chroma:
    """Получает или создает векторное хранилище для указанной коллекции"""
    if collection not in vectorstores:
        vectorstores[collection] = Chroma(
            collection_name=collection,
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    return vectorstores[collection]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

def filter_duplicates(docs: List[Document], collection: str) -> List[Document]:
    """Фильтрует документы, уже существующие в базе"""
    existing_hashes = set()
    vectorstore = get_vectorstore(collection)

    # Получаем хеши существующих документов
    if os.path.exists(PERSIST_DIR):
        existing_data = vectorstore.get()  # Получаем все данные из базы
        for metadata in existing_data["metadatas"]:
            if "doc_hash" in metadata:
                existing_hashes.add(metadata["doc_hash"])

    # Фильтрация новых документов
    unique_docs = []
    for doc in docs:
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        if content_hash not in existing_hashes:
            doc.metadata["doc_hash"] = content_hash  # Добавляем хеш в метаданные
            unique_docs.append(doc)

    return unique_docs

def load_document(file_path: str) -> List[Dict]:
    """Загрузка документа в зависимости от его типа."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension in ['.doc', '.docx']:
            loader = Docx2txtLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            logger.error(f"Неподдерживаемый формат файла: {file_extension}")
            return []
        
        # Загрузка и разбиение документа
        documents = loader.load()
        splits = text_splitter.split_documents(documents)
        
        # Добавление метаданных
        for split in splits:
            split.metadata.update({
                'source': file_path,
                'file_type': file_extension[1:],
                'file_name': os.path.basename(file_path)
            })
        
        return splits
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке документа {file_path}: {str(e)}")
        return []

def process_document(file_path: str, collection: str) -> bool:
    """Обработка документа и сохранение в ChromaDB."""
    try:        
        # Загрузка и разбиение документа
        splits = load_document(file_path)
        if not splits:
            return False
        
        # Фильтрация дубликатов
        unique_splits = filter_duplicates(splits, collection)
        if not unique_splits:
            return False
        
        # Добавление уникальных документов в векторное хранилище
        vectorstore = get_vectorstore(collection)
        vectorstore.add_documents(unique_splits)
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при обработке документа: {str(e)}")
        return False

def search_documents(query: str, collection: str, n_results: int = 5) -> List[Dict]:
    """Поиск по документам."""
    try:
        # Получаем векторное хранилище для указанной коллекции
        vectorstore = get_vectorstore(collection)
        
        # Поиск в векторном хранилище
        results = vectorstore.similarity_search_with_score(
            query,
            k=n_results
        )
        
        # Форматирование результатов
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            })
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Ошибка при поиске документов: {e}")
        return []

def delete_document(document_id: str, collection: str) -> bool:
    """Удаление документа из базы данных."""
    try:
        vectorstore = get_vectorstore(collection)
        # Удаление документов по метаданным
        vectorstore.delete(
            filter={"source": document_id}
        )
        
        logger.info(f"Документ успешно удален: {document_id}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при удалении документа: {e}")
        return False

def get_document_info(collection: str = None) -> Dict:
    """Получение информации о сохраненных документах."""
    try:
        if collection:
            # Получаем информацию для конкретной коллекции
            vectorstore = get_vectorstore(collection)
            collection_data = vectorstore.get()
        else:
            # Получаем информацию по всем коллекциям
            all_documents = []
            total_documents = 0
            for coll in vectorstores.keys():
                vectorstore = get_vectorstore(coll)
                collection_data = vectorstore.get()
                if collection_data and collection_data['documents']:
                    sources = set()
                    for metadata in collection_data['metadatas']:
                        sources.add(metadata['source'])
                    total_documents += len(sources)
                    all_documents.extend([
                        {
                            "source": source,
                            "collection": coll,
                            "chunks": len([m for m in collection_data['metadatas'] if m['source'] == source])
                        }
                        for source in sources
                    ])
            return {
                "total_documents": total_documents,
                "documents": all_documents
            }

        if not collection_data or not collection_data['documents']:
            return {"total_documents": 0, "documents": []}
        
        # Подсчет уникальных документов по источнику
        sources = set()
        for metadata in collection_data['metadatas']:
            sources.add(metadata['source'])
        
        return {
            "total_documents": len(sources),
            "documents": [
                {
                    "source": source,
                    "chunks": len([m for m in collection_data['metadatas'] if m['source'] == source])
                }
                for source in sources
            ]
        }
        
    except Exception as e:
        logger.error(f"Ошибка при получении информации о документах: {e}")
        return {"total_documents": 0, "documents": []}

@app.post("/process_document")
async def process_document_endpoint(
    file: UploadFile = File(...),
    collection: str = "default"
) -> MCPResponse:
    """Эндпоинт для обработки документа"""
    try:
        # Сохраняем файл временно
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Обрабатываем документ
        success = process_document(temp_path, collection)
        
        # Удаляем временный файл
        os.remove(temp_path)
        
        if success:
            return MCPResponse(
                status="success",
                data={"message": f"Документ успешно обработан и добавлен в коллекцию {collection}"}
            )
        else:
            return MCPResponse(
                status="error",
                error="Ошибка при обработке документа"
            )
    except Exception as e:
        logger.error(f"Ошибка при обработке документа: {str(e)}")
        return MCPResponse(
            status="error",
            error=str(e)
        )

@app.post("/search")
async def search_endpoint(request: SearchRequest) -> MCPResponse:
    """Эндпоинт для поиска по документам"""
    try:
        results = search_documents(request.query, request.collection, request.n_results)
        return MCPResponse(
            status="success",
            data={"results": results}
        )
    except Exception as e:
        logger.error(f"Ошибка при поиске документов: {str(e)}")
        return MCPResponse(
            status="error",
            error=str(e)
        )

@app.post("/delete_document")
async def delete_document_endpoint(request: DeleteDocumentRequest) -> MCPResponse:
    """Эндпоинт для удаления документа"""
    try:
        success = delete_document(request.document_id, request.collection)
        if success:
            return MCPResponse(
                status="success",
                data={"message": f"Документ успешно удален из коллекции {request.collection}"}
            )
        else:
            return MCPResponse(
                status="error",
                error="Ошибка при удалении документа"
            )
    except Exception as e:
        logger.error(f"Ошибка при удалении документа: {str(e)}")
        return MCPResponse(
            status="error",
            error=str(e)
        )

@app.post("/get_document_info")
async def get_document_info_endpoint(request: DocumentInfoRequest) -> MCPResponse:
    """Эндпоинт для получения информации о документах"""
    try:
        info = get_document_info(request.collection)
        return MCPResponse(
            status="success",
            data=info
        )
    except Exception as e:
        logger.error(f"Ошибка при получении информации о документах: {str(e)}")
        return MCPResponse(
            status="error",
            error=str(e)
        )

class DocumentProcessorServer(FastMCPServer):
    def __init__(self, host: str, port: int):
        super().__init__(host, port)
        self.register_tools()

    def register_tools(self):
        """Регистрация инструментов MCP"""
        self.register_tool(self.process_document_tool)
        self.register_tool(self.search_tool)
        self.register_tool(self.delete_document_tool)
        self.register_tool(self.get_document_info_tool)

    @tool(
        name="process_document",
        description="Обработка и индексация документа в указанную коллекцию",
        parameters={
            "file_path": {"type": "string", "description": "Путь к файлу для обработки"},
            "collection": {"type": "string", "description": "Имя коллекции для сохранения документа"}
        }
    )
    async def process_document_tool(self, file_path: str, collection: str = "default") -> Dict:
        """Обработчик команды process_document"""
        try:
            success = process_document(file_path, collection)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Документ успешно обработан и добавлен в коллекцию {collection}"
                }
            else:
                return {
                    "status": "error",
                    "error": "Ошибка при обработке документа"
                }
        except Exception as e:
            logger.error(f"Ошибка при обработке документа: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    @tool(
        name="search",
        description="Поиск по документам в указанной коллекции",
        parameters={
            "query": {"type": "string", "description": "Поисковый запрос"},
            "collection": {"type": "string", "description": "Имя коллекции для поиска"},
            "n_results": {"type": "integer", "description": "Количество результатов", "default": 5}
        }
    )
    async def search_tool(self, query: str, collection: str, n_results: int = 5) -> Dict:
        """Обработчик команды search"""
        try:
            results = search_documents(query, collection, n_results)
            return {
                "status": "success",
                "results": results
            }
        except Exception as e:
            logger.error(f"Ошибка при поиске документов: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    @tool(
        name="delete_document",
        description="Удаление документа из коллекции",
        parameters={
            "document_id": {"type": "string", "description": "ID документа для удаления"},
            "collection": {"type": "string", "description": "Имя коллекции"}
        }
    )
    async def delete_document_tool(self, document_id: str, collection: str) -> Dict:
        """Обработчик команды delete_document"""
        try:
            success = delete_document(document_id, collection)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Документ успешно удален из коллекции {collection}"
                }
            else:
                return {
                    "status": "error",
                    "error": "Ошибка при удалении документа"
                }
        except Exception as e:
            logger.error(f"Ошибка при удалении документа: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    @tool(
        name="get_document_info",
        description="Получение информации о документах в коллекции",
        parameters={
            "collection": {"type": "string", "description": "Имя коллекции", "optional": True}
        }
    )
    async def get_document_info_tool(self, collection: Optional[str] = None) -> Dict:
        """Обработчик команды get_document_info"""
        try:
            info = get_document_info(collection)
            return {
                "status": "success",
                "info": info
            }
        except Exception as e:
            logger.error(f"Ошибка при получении информации о документах: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

async def main():
    """Основная функция запуска сервера"""
    server = DocumentProcessorServer(MCP_HOST, MCP_PORT)
    await server.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Сервер остановлен") 