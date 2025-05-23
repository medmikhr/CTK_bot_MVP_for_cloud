import os
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация глобальных компонентов
embeddings = HuggingFaceEmbeddings(
    model_name="sberbank-ai/sbert_large_nlu_ru",
    model_kwargs={'device': 'cpu'}
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# Инициализация базы данных Chroma
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

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
        logger.error(f"Ошибка при загрузке документа {file_path}: {e}")
        return []

def process_document(file_path: str) -> bool:
    """Обработка документа и сохранение в ChromaDB."""
    try:
        # Загрузка и разбиение документа
        splits = load_document(file_path)
        if not splits:
            return False
        
        # Добавление документов в векторное хранилище
        vectorstore.add_documents(splits)
        vectorstore.persist()
        
        logger.info(f"Документ успешно обработан и сохранен: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при обработке документа: {e}")
        return False

def search_documents(query: str, n_results: int = 5) -> List[Dict]:
    """Поиск по документам."""
    try:
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

def delete_document(document_id: str) -> bool:
    """Удаление документа из базы данных."""
    try:
        # Удаление документов по метаданным
        vectorstore.delete(
            filter={"source": document_id}
        )
        vectorstore.persist()
        
        logger.info(f"Документ успешно удален: {document_id}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при удалении документа: {e}")
        return False

def get_document_info() -> Dict:
    """Получение информации о сохраненных документах."""
    try:
        collection = vectorstore.get()
        if not collection or not collection['documents']:
            return {"total_documents": 0, "documents": []}
        
        # Подсчет уникальных документов по источнику
        sources = set()
        for metadata in collection['metadatas']:
            sources.add(metadata['source'])
        
        return {
            "total_documents": len(sources),
            "documents": [
                {
                    "source": source,
                    "chunks": len([m for m in collection['metadatas'] if m['source'] == source])
                }
                for source in sources
            ]
        }
        
    except Exception as e:
        logger.error(f"Ошибка при получении информации о документах: {e}")
        return {"total_documents": 0, "documents": []} 