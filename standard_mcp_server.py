#!/usr/bin/env python3
"""
Стандартный MCP сервер для тестирования подключения
"""

import asyncio
import logging
from typing import Any, Sequence
from mcp.server import Server
from mcp.types import Tool, TextContent

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создание MCP сервера
server = Server("test-vector-search")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Список доступных инструментов"""
    return [
        Tool(
            name="search_documents",
            description="Поиск по документам в векторной базе данных",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Максимальное количество результатов",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_server_info",
            description="Получить информацию о сервере",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> Sequence[TextContent]:
    """Выполнение инструмента"""
    
    if name == "search_documents":
        query = arguments.get("query", "")
        limit = arguments.get("limit", 5)
        
        logger.info(f"Поиск по запросу: {query}, лимит: {limit}")
        
        # Заглушка для демонстрации
        mock_results = [
            {
                "content": f"Найденный документ 1 по запросу '{query}'",
                "score": 0.95,
                "metadata": {"source": "doc1.pdf", "page": 1}
            },
            {
                "content": f"Найденный документ 2 по запросу '{query}'",
                "score": 0.87,
                "metadata": {"source": "doc2.pdf", "page": 3}
            }
        ]
        
        results = mock_results[:limit]
        return [TextContent(type="text", text=str(results))]
        
    elif name == "get_server_info":
        info = {
            "name": "Standard MCP Vector Search Server",
            "version": "1.0.0",
            "status": "running",
            "capabilities": ["search", "info"]
        }
        return [TextContent(type="text", text=str(info))]
    
    else:
        raise ValueError(f"Неизвестный инструмент: {name}")

async def main():
    """Главная функция"""
    # Импорт зависимостей для запуска
    from mcp.server.stdio import stdio_server
    
    logger.info("Запуск стандартного MCP сервера через stdio")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
