import os
from typing import List, Dict, Set
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
import logging
from datetime import datetime, timedelta
import hashlib
from langchain.schema import Document
import traceback

# Конфигурация
PERSIST_DIR = "chroma_db"  # Директория для хранения базы данных Chroma
EMBEDDING_MODEL = "cointegrated/rubert-tiny2"  # Модель для эмбеддингов
CHUNK_SIZE = 1000  # Размер чанка при разбиении текста
CHUNK_OVERLAP = 200  # Перекрытие между чанками

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Глобальные переменные
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'}
)

# Создаем словарь для хранения векторных хранилищ для разных коллекций
vectorstores = {}

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

if __name__ == "__main__":
    # Пример использования process_document
    file_path = "C:/Users/Administrator/Documents/LangFlow/ctk_pdf_for_Giga/DAMA-DMBOK (2020).pdf"  # Путь к вашему документу
    collection_name = "dama"  # Имя коллекции для хранения документов
    
    success = process_document(file_path, collection_name)
    if success:
        print(f"Документ {file_path} успешно обработан и добавлен в коллекцию {collection_name}")
    else:
        print(f"Ошибка при обработке документа {file_path}") 