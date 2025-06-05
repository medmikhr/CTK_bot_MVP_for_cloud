#!/usr/bin/env python3
"""
Правильный MCP клиент на основе официальной документации
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_connection(config_path: str, server_name: str):
    """Тестирование MCP подключения"""
    
    # Загрузка конфигурации
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    server_config = config['mcpServers'].get(server_name)
    if not server_config:
        raise ValueError(f"Сервер {server_name} не найден в конфигурации")
    
    print(f"📋 Конфигурация сервера: {server_config}")
    
    # Проверяем тип подключения
    if 'command' not in server_config:
        raise ValueError(f"Неподдерживаемый тип конфигурации для сервера {server_name}")
    
    # Создание параметров сервера для stdio подключения
    server_params = StdioServerParameters(
        command=server_config['command'],
        args=server_config.get('args', []),
        env=server_config.get('env', {})
    )
    
    print(f"🔗 Подключение к серверу {server_name}...")
    print(f"   Команда: {server_params.command}")
    print(f"   Аргументы: {server_params.args}")
    print(f"   Окружение: {server_params.env}")
    
    # Подключение к серверу
    async with stdio_client(server_params) as (read_stream, write_stream):
        print("✅ Stdio клиент создан успешно!")
        
        async with ClientSession(read_stream, write_stream) as session:
            print("✅ Сессия создана успешно!")
            
            # Инициализация соединения
            await session.initialize()
            print("✅ Соединение инициализировано!")
            
            # Тестирование инструментов
            print("\n🔧 Получение списка инструментов...")
            try:
                tools_response = await session.list_tools()
                if tools_response.tools:
                    print(f"✅ Найдено {len(tools_response.tools)} инструментов:")
                    for tool in tools_response.tools:
                        print(f"  - {tool.name}: {tool.description or 'Без описания'}")
                else:
                    print("⚠️  Инструменты не найдены")
            except Exception as e:
                print(f"❌ Ошибка получения инструментов: {str(e)}")
            
            # Тестирование ресурсов
            print("\n📚 Получение списка ресурсов...")
            try:
                resources_response = await session.list_resources()
                if resources_response.resources:
                    print(f"✅ Найдено {len(resources_response.resources)} ресурсов:")
                    for resource in resources_response.resources:
                        print(f"  - {resource.name}: {resource.description or 'Без описания'}")
                else:
                    print("⚠️  Ресурсы не найдены")
            except Exception as e:
                print(f"❌ Ошибка получения ресурсов: {str(e)}")
            
            # Тестирование промптов
            print("\n💬 Получение списка промптов...")
            try:
                prompts_response = await session.list_prompts()
                if prompts_response.prompts:
                    print(f"✅ Найдено {len(prompts_response.prompts)} промптов:")
                    for prompt in prompts_response.prompts:
                        print(f"  - {prompt.name}: {prompt.description or 'Без описания'}")
                else:
                    print("⚠️  Промпты не найдены")
            except Exception as e:
                print(f"❌ Ошибка получения промптов: {str(e)}")
            
            # Тестирование выполнения инструмента
            print("\n🧪 Тестирование выполнения инструментов...")
            try:
                tools_response = await session.list_tools()
                if tools_response.tools:
                    for tool in tools_response.tools:
                        print(f"\n   Тестирование инструмента: {tool.name}")
                        try:
                            if tool.name == "get_server_info":
                                result = await session.call_tool(tool.name, {})
                                print(f"   ✅ Результат: {result}")
                            elif tool.name == "search_documents":
                                result = await session.call_tool(tool.name, {
                                    "query": "тестовый запрос",
                                    "limit": 2
                                })
                                print(f"   ✅ Результат: {result}")
                            else:
                                print(f"   ⏭️  Пропуск неизвестного инструмента: {tool.name}")
                        except Exception as e:
                            print(f"   ❌ Ошибка выполнения инструмента {tool.name}: {str(e)}")
            except Exception as e:
                print(f"❌ Ошибка тестирования инструментов: {str(e)}")
            
            print("\n🎉 Все тесты MCP подключения завершены!")

async def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python correct_mcp_client.py <config_path> <server_name>")
        print("Example: python correct_mcp_client.py mcp_config.json vector-search-server")
        sys.exit(1)
    
    config_path = sys.argv[1]
    server_name = sys.argv[2]
    
    try:
        await test_mcp_connection(config_path, server_name)
    except Exception as e:
        print(f"❌ Критическая ошибка: {str(e)}")
        import traceback
        print("Трассировка ошибки:")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
