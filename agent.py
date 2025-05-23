import os
from typing import List, Dict
from langchain_gigachat import GigaChat
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain.agents import tool
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from document_processor_langchain import PERSIST_DIR, EMBEDDING_MODEL
import logging

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Получение токенов из переменных окружения
GC_AUTH = os.getenv('GIGACHAT_TOKEN')
HF_TOKEN = os.getenv('HF_TOKEN')
if not GC_AUTH or not HF_TOKEN:
    raise ValueError("Не найдены необходимые токены в переменных окружения")

# Инициализация GigaChat
llm = GigaChat(
    credentials=GC_AUTH,
    model='GigaChat:latest',
    verify_ssl_certs=False,
    profanity_check=False
)

# Инициализация эмбеддингов
embeddings = HuggingFaceEndpointEmbeddings(
    repo_id=EMBEDDING_MODEL,
    huggingfacehub_api_token=HF_TOKEN
)

# Инициализация векторных хранилищ
vector_stores = {
    "dama": Chroma(collection_name="dama", persist_directory=PERSIST_DIR, embedding_function=embeddings),
    "ctk": Chroma(collection_name="ctk", persist_directory=PERSIST_DIR, embedding_function=embeddings),
    "sbf": Chroma(collection_name="sbf", persist_directory=PERSIST_DIR, embedding_function=embeddings)
}

def search_documents(store: Chroma, query: str, k: int = 5) -> tuple[str, List]:
    """Общая функция для поиска документов в векторном хранилище."""
    retrieved_docs = store.similarity_search(query, k=k)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@tool(response_format="content_and_artifact")
def dama_retrieve_tool(query: str):
    """Используй этот инструмент для поиска информации о методологии управления данными, 
    стандартах DAMA, процессах управления данными, ролях и ответственности в области управления данными.
    Этот инструмент содержит информацию из Data Management Body Of Knowledge (DMBOK)."""
    return search_documents(vector_stores["dama"], query)

@tool(response_format="content_and_artifact")
def ctk_retrieve_tool(query: str):
    """Используй этот инструмент для поиска информации о технологических решениях, 
    архитектуре систем, методологиях разработки, стандартах и практиках ЦТК.
    Этот инструмент содержит документацию Центра Технологического Консалтинга."""
    return search_documents(vector_stores["ctk"], query)

@tool(response_format="content_and_artifact")
def sbf_retrieve_tool(query: str):
    """Используй этот инструмент для поиска информации о факторинговых операциях, 
    продуктах и услугах СберБанк Факторинга, процессах работы с клиентами.
    Этот инструмент содержит документацию СберБанк Факторинга."""
    return search_documents(vector_stores["sbf"], query)

tools_dict = [dama_retrieve_tool, ctk_retrieve_tool, sbf_retrieve_tool]
tools = ToolNode(tools_dict)
memory = MemorySaver()
agent_executor = create_react_agent(llm, tools_dict, checkpointer=memory)

def agent_ask(user_id, input_message):
    config = {"configurable": {"thread_id": user_id}}
    for event in agent_executor.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        answer_message = event["messages"][-1]
        answer_message.pretty_print()
    return answer_message

if __name__ == '__main__':
    while True:
        user_input = input("Спрашивай: ")
        agent_ask(1, user_input)

# какие tools тебе доступны
# используй ctk_retrieve_tool и назови слои информационной архитектуры
