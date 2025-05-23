import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langdetect import detect
import PyPDF2
import docx
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, persist_directory: str = "chroma_db"):
        """Инициализация процессора документов."""
        # Инициализация модели для эмбеддингов (специально для русского языка)
        self.embedding_model = SentenceTransformer('sberbank-ai/sbert_large_nlu_ru')
        
        # Инициализация ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Создание или получение коллекции
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Извлечение текста из PDF файла."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Ошибка при чтении PDF файла: {e}")
            return ""

    def extract_text_from_docx(self, file_path: str) -> str:
        """Извлечение текста из DOCX файла."""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Ошибка при чтении DOCX файла: {e}")
            return ""

    def extract_text_from_txt(self, file_path: str) -> str:
        """Извлечение текста из TXT файла."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Ошибка при чтении TXT файла: {e}")
            return ""

    def process_document(self, file_path: str, metadata: Dict = None) -> bool:
        """Обработка документа и сохранение в ChromaDB."""
        try:
            # Определение типа файла и извлечение текста
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_extension == '.docx':
                text = self.extract_text_from_docx(file_path)
            elif file_extension == '.txt':
                text = self.extract_text_from_txt(file_path)
            else:
                logger.error(f"Неподдерживаемый формат файла: {file_extension}")
                return False

            if not text:
                logger.error("Не удалось извлечь текст из документа")
                return False

            # Проверка языка текста
            try:
                lang = detect(text)
                if lang != 'ru':
                    logger.warning(f"Документ не на русском языке (определен язык: {lang})")
            except:
                logger.warning("Не удалось определить язык документа")

            # Разбиение текста на чанки
            chunks = self._split_text(text)
            
            # Подготовка метаданных
            if metadata is None:
                metadata = {}
            metadata['source'] = file_path
            metadata['file_type'] = file_extension[1:]

            # Создание эмбеддингов и сохранение в ChromaDB
            for i, chunk in enumerate(chunks):
                embedding = self.embedding_model.encode(chunk).tolist()
                self.collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{**metadata, 'chunk_id': i}],
                    ids=[f"{os.path.basename(file_path)}_{i}"]
                )

            logger.info(f"Документ успешно обработан и сохранен: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Ошибка при обработке документа: {e}")
            return False

    def _split_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Разбиение текста на чанки."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 для пробела

            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Поиск по документам."""
        try:
            # Создание эмбеддинга для запроса
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Поиск в ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            # Форматирование результатов
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Ошибка при поиске документов: {e}")
            return []

    def delete_document(self, document_id: str) -> bool:
        """Удаление документа из базы данных."""
        try:
            self.collection.delete(
                where={"source": document_id}
            )
            logger.info(f"Документ успешно удален: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при удалении документа: {e}")
            return False 