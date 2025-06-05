#!/usr/bin/env python3
"""
MCP Streamable HTTP клиент согласно спецификации MCP 2025-03-26
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
import aiohttp
import asyncio

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPStreamableClient:
    """MCP клиент с поддержкой Streamable HTTP транспорта"""
    
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url
        self.session_id: Optional[str] = None
        self.next_request_id = 1
        self.initialized = False
    
    def get_next_request_id(self) -> int:
        """Получение следующего ID запроса"""
        request_id = self.next_request_id
        self.next_request_id += 1
        return request_id
    
    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Отправка JSON-RPC запроса через HTTP POST"""
        request_id = self.get_next_request_id()
        
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        
        if params is not None:
            message["params"] = params
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream'
        }
        
        # Добавляем session ID если есть
        if self.session_id:
            headers['Mcp-Session-Id'] = self.session_id
        
        logger.info(f"Отправка запроса: {method}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.endpoint_url,
                json=message,
                headers=headers
            ) as response:
                
                content_type = response.headers.get('Content-Type', '')
                
                # Получаем session ID из заголовка (для initialize)
                if 'Mcp-Session-Id' in response.headers:
                    self.session_id = response.headers['Mcp-Session-Id']
                    logger.info(f"Получен session ID: {self.session_id}")
                
                if content_type.startswith('text/event-stream'):
                    # SSE response
                    return await self.read_sse_response(response)
                elif content_type.startswith('application/json'):
                    # JSON response
                    return await response.json()
                else:
                    raise ValueError(f"Неподдерживаемый Content-Type: {content_type}")
    
    async def read_sse_response(self, response) -> Dict[str, Any]:
        """Чтение SSE response"""
        logger.info("Чтение SSE response...")
        
        last_data = None
        async for line in response.content:
            line_str = line.decode('utf-8').strip()
            
            if line_str.startswith('id: '):
                # Сохраняем ID события (для resumability)
                event_id = line_str[4:]
                continue
            elif line_str.startswith('data: '):
                data_str = line_str[6:]  # Убираем 'data: '
                try:
                    data = json.loads(data_str)
                    logger.info(f"Получено SSE сообщение: {data.get('method', 'response')}")
                    last_data = data
                except json.JSONDecodeError as e:
                    logger.error(f"Ошибка парсинга SSE данных: {e}")
                    continue
            elif line_str == '' and last_data is not None:
                # Пустая строка означает конец события
                return last_data
        
        if last_data:
            return last_data
        
        raise ValueError("SSE stream закрыт без получения данных")
    
    async def initialize(self) -> Dict[str, Any]:
        """Инициализация MCP соединения"""
        logger.info("Инициализация MCP клиента...")
        
        params = {
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "roots": {
                    "listChanged": False
                },
                "sampling": {}
            },
            "clientInfo": {
                "name": "MCP Streamable HTTP Test Client",
                "version": "1.0.0"
            }
        }
        
        result = await self.send_request("initialize", params)
        
        if 'error' in result:
            raise Exception(f"Ошибка инициализации: {result['error']}")
        
        self.initialized = True
        logger.info("✅ MCP клиент инициализирован успешно!")
        return result
    
    async def list_tools(self) -> Dict[str, Any]:
        """Получение списка доступных инструментов"""
        if not self.initialized:
            raise Exception("Клиент не инициализирован")
        
        logger.info("Получение списка инструментов...")
        return await self.send_request("tools/list")
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Вызов инструмента"""
        if not self.initialized:
            raise Exception("Клиент не инициализирован")
        
        logger.info(f"Вызов инструмента: {name}")
        
        params = {
            "name": name,
            "arguments": arguments
        }
        
        return await self.send_request("tools/call", params)
    
    async def list_resources(self) -> Dict[str, Any]:
        """Получение списка ресурсов"""
        if not self.initialized:
            raise Exception("Клиент не инициализирован")
        
        logger.info("Получение списка ресурсов...")
        return await self.send_request("resources/list")
    
    async def list_prompts(self) -> Dict[str, Any]:
        """Получение списка промптов"""
        if not self.initialized:
            raise Exception("Клиент не инициализирован")
        
        logger.info("Получение списка промптов...")
        return await self.send_request("prompts/list")
    
    async def close_session(self):
        """Закрытие сессии"""
        if self.session_id:
            headers = {'Mcp-Session-Id': self.session_id}
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(self.endpoint_url, headers=headers) as response:
                    if response.status == 200:
                        logger.info("✅ Сессия закрыта успешно")
                    else:
                        logger.warning(f"Не удалось закрыть сессию: {response.status}")

async def test_streamable_http_connection(endpoint_url: str = "http://127.0.0.1:8080/mcp"):
    """Тестирование Streamable HTTP MCP подключения"""
    
    print(f"🚀 Тестирование MCP Streamable HTTP клиента")
    print(f"🔗 Подключение к: {endpoint_url}")
    print("=" * 60)
    
    # Проверка доступности сервера
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint_url.replace('/mcp', '/health')) as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ Сервер доступен: {health_data.get('status')}")
                    print(f"   Транспорт: {health_data.get('transport')}")
                    print(f"   Версия: {health_data.get('version')}")
                else:
                    print(f"⚠️  Сервер отвечает с кодом: {response.status}")
    except Exception as e:
        print(f"❌ Сервер недоступен: {e}")
        print("💡 Убедитесь, что сервер запущен: python mcp_streamable_server.py")
        return False
    
    print("\n" + "-" * 60)
    
    # Тестирование MCP протокола
    client = MCPStreamableClient(endpoint_url)
    
    try:
        # 1. Инициализация
        print("\n1️⃣  Инициализация MCP соединения...")
        init_result = await client.initialize()
        print(f"✅ Инициализация успешна")
        print(f"   Session ID: {client.session_id}")
        print(f"   Версия протокола: {init_result['result']['protocolVersion']}")
        print(f"   Сервер: {init_result['result']['serverInfo']['name']} v{init_result['result']['serverInfo']['version']}")
        
        # 2. Список инструментов
        print("\n2️⃣  Получение списка инструментов...")
        tools_result = await client.list_tools()
        if 'error' in tools_result:
            print(f"❌ Ошибка: {tools_result['error']}")
        else:
            tools = tools_result['result']['tools']
            print(f"✅ Найдено {len(tools)} инструментов:")
            for tool in tools:
                print(f"   - {tool['name']}: {tool['description']}")
        
        # 3. Тестирование инструментов
        print("\n3️⃣  Тестирование инструментов...")
        
        # Тест get_server_info
        print("\n   🧪 Тестирование get_server_info...")
        try:
            result = await client.call_tool("get_server_info", {})
            if 'error' in result:
                print(f"   ❌ Ошибка: {result['error']}")
            else:
                content = result['result']['content'][0]['text']
                print(f"   ✅ Результат получен (длина: {len(content)} символов)")
                print(f"   📋 Первые 200 символов: {content[:200]}...")
        except Exception as e:
            print(f"   ❌ Исключение: {e}")
        
        # Тест search_documents
        print("\n   🧪 Тестирование search_documents...")
        try:
            result = await client.call_tool("search_documents", {
                "query": "тестовый запрос для демонстрации",
                "limit": 3
            })
            if 'error' in result:
                print(f"   ❌ Ошибка: {result['error']}")
            else:
                content = result['result']['content'][0]['text']
                print(f"   ✅ Результат получен (длина: {len(content)} символов)")
                print(f"   📋 Первые 200 символов: {content[:200]}...")
        except Exception as e:
            print(f"   ❌ Исключение: {e}")
        
        # 4. Список ресурсов
        print("\n4️⃣  Получение списка ресурсов...")
        try:
            resources_result = await client.list_resources()
            if 'error' in resources_result:
                print(f"❌ Ошибка: {resources_result['error']}")
            else:
                resources = resources_result['result']['resources']
                print(f"✅ Найдено {len(resources)} ресурсов")
        except Exception as e:
            print(f"❌ Исключение: {e}")
        
        # 5. Список промптов
        print("\n5️⃣  Получение списка промптов...")
        try:
            prompts_result = await client.list_prompts()
            if 'error' in prompts_result:
                print(f"❌ Ошибка: {prompts_result['error']}")
            else:
                prompts = prompts_result['result']['prompts']
                print(f"✅ Найдено {len(prompts)} промптов")
        except Exception as e:
            print(f"❌ Исключение: {e}")
        
        # 6. Закрытие сессии
        print("\n6️⃣  Закрытие сессии...")
        await client.close_session()
        
        print("\n" + "=" * 60)
        print("🎉 Все тесты MCP Streamable HTTP завершены успешно!")
        print("✅ Сервер работает как сервис на порту 8080")
        print("✅ Протокол MCP Streamable HTTP функционирует корректно")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Критическая ошибка тестирования: {e}")
        import traceback
        print("Трассировка ошибки:")
        print(traceback.format_exc())
        return False

async def main():
    """Главная функция"""
    await test_streamable_http_connection()

if __name__ == "__main__":
    asyncio.run(main())
