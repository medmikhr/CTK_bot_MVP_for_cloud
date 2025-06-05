#!/usr/bin/env python3
"""
Minimal MCP Tools Client
Минимальный клиент для работы с MCP серверами - только функции, без классов.
"""

import asyncio
import json
import logging
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Настройка логирования
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_tool_for_display(tool_data) -> str:
    """Форматирует инструмент для отображения."""
    result = f"🛠️  {tool_data.name}\n"
    result += f"   📝 Описание: {tool_data.description or 'Описание отсутствует'}\n"
    
    # Обработка параметров
    if hasattr(tool_data, 'inputSchema') and tool_data.inputSchema:
        schema = tool_data.inputSchema
        if isinstance(schema, dict):
            properties = schema.get('properties', {})
            required = schema.get('required', [])
            
            if properties:
                result += "   📊 Параметры:\n"
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'unknown')
                    param_desc = param_info.get('description', '')
                    is_required = '(обязательный)' if param_name in required else '(опциональный)'
                    
                    result += f"      • {param_name}: {param_type} {is_required}\n"
                    if param_desc:
                        result += f"        └─ {param_desc}\n"
    
    return result


async def connect_to_server(config):
    """Подключается к MCP серверу и возвращает сессию."""
    logger.info("🔗 Подключение к MCP серверу...")
    
    exit_stack = AsyncExitStack()
    
    try:
        server_params = StdioServerParameters(
            command=config["command"],
            args=config["args"],
            env=config.get("env")
        )
        
        # Устанавливаем подключение
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        
        # Создаем сессию
        session = await exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        
        # Инициализируем сессию
        await session.initialize()
        
        logger.info("✅ Подключение установлено")
        return session, exit_stack
        
    except Exception as e:
        logger.error(f"❌ Ошибка подключения: {e}")
        await exit_stack.aclose()
        raise


async def get_tools(session):
    """Получает список инструментов от сервера."""
    try:
        logger.info("📋 Запрос списка инструментов...")
        tools_response = await session.list_tools()
        
        if hasattr(tools_response, 'tools') and tools_response.tools:
            logger.info(f"✅ Найдено {len(tools_response.tools)} инструментов")
            return tools_response.tools
        else:
            logger.warning("⚠️ Инструменты не найдены")
            return []
            
    except Exception as e:
        logger.error(f"❌ Ошибка получения инструментов: {e}")
        raise


def display_tools(tools):
    """Отображает список инструментов."""
    print("\n" + "="*80)
    print("📋 СПИСОК ИНСТРУМЕНТОВ MCP СЕРВЕРА")
    print("="*80)
    
    if not tools:
        print("❌ Инструменты не найдены")
        return
    
    print(f"🔢 Всего инструментов: {len(tools)}\n")
    
    for i, tool in enumerate(tools, 1):
        print(f"{i}. {format_tool_for_display(tool)}")
    
    print("="*80)


def load_config(config_file="servers_config.json"):
    """Загружает конфигурацию из файла."""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"📋 Конфигурация загружена из {config_file}")
        return config
    except FileNotFoundError:
        logger.error(f"❌ Файл конфигурации {config_file} не найден")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"❌ Ошибка парсинга JSON: {e}")
        raise


async def main():
    """Главная функция."""
    session = None
    exit_stack = None
    
    try:
        # Загружаем конфигурацию
        config = load_config()
        
        # Берем первый сервер из конфигурации
        server_configs = config.get("mcpServers", {})
        if not server_configs:
            logger.error("❌ Нет серверов в конфигурации")
            return
        
        # Используем первый сервер
        server_name, server_config = next(iter(server_configs.items()))
        logger.info(f"🎯 Используем сервер: {server_name}")
        
        # Подключаемся к серверу
        session, exit_stack = await connect_to_server(server_config)
        
        # Получаем инструменты
        tools = await get_tools(session)
        
        # Отображаем инструменты
        display_tools(tools)
        
    except Exception as e:
        logger.error(f"❌ Ошибка выполнения: {e}")
    finally:
        if exit_stack:
            logger.info("🧹 Очистка ресурсов...")
            await exit_stack.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Операция прервана пользователем")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
