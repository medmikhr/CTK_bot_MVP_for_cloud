#!/usr/bin/env python3
"""
MCP Streamable HTTP сервер согласно спецификации MCP 2025-03-26
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional
from aiohttp import web, WSMsgType

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPStreamableServer:
    """MCP сервер с поддержкой Streamable HTTP транспорта"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.next_request_id = 1
        
    def create_session_id(self) -> str:
        """Создание уникального ID сессии"""
        return str(uuid.uuid4())
    
    def is_safe_origin(self, origin: str) -> bool:
        """Проверка безопасности Origin заголовка"""
        # Разрешаем localhost, 127.0.0.1 и null origin (для тестирования)
        safe_origins = [
            'http://localhost:8080',
            'http://127.0.0.1:8080',
            'http://localhost',
            'http://127.0.0.1',
            'null'  # Для файловых запросов и некоторых инструментов
        ]
        return origin in safe_origins or origin.startswith('http://localhost:') or origin.startswith('http://127.0.0.1:')
    
    async def handle_mcp_endpoint(self, request):
        """Основной MCP endpoint для POST и GET запросов"""
        if request.method == 'OPTIONS':
            return await self.handle_options(request)
        elif request.method == 'POST':
            return await self.handle_post(request)
        elif request.method == 'GET':
            return await self.handle_get(request)
        elif request.method == 'DELETE':
            return await self.handle_delete(request)
        else:
            return web.json_response(
                {"error": "Method not allowed"}, 
                status=405
            )
    
    async def handle_options(self, request):
        """Обработка OPTIONS запросов для CORS"""
        return web.Response(
            status=200,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Accept, Content-Type, Mcp-Session-Id, Last-Event-ID',
                'Access-Control-Max-Age': '86400',
            }
        )
    
    async def handle_post(self, request):
        """Обработка POST запросов (отправка сообщений на сервер)"""
        try:
            # Проверка Origin заголовка для безопасности (защита от DNS rebinding)
            origin = request.headers.get('Origin')
            if origin and not self.is_safe_origin(origin):
                logger.warning(f"Отклонен запрос с небезопасным Origin: {origin}")
                return web.json_response(
                    {"error": "Unsafe origin"}, 
                    status=403
                )
            
            # Проверка Content-Type
            if request.content_type != 'application/json':
                return web.json_response(
                    {"error": "Content-Type must be application/json"}, 
                    status=400
                )
            
            # Проверка заголовка Accept
            accept_header = request.headers.get('Accept', '')
            if 'application/json' not in accept_header and 'text/event-stream' not in accept_header:
                return web.json_response(
                    {"error": "Accept header must include application/json or text/event-stream"}, 
                    status=400
                )
            
            # Получение session ID
            session_id = request.headers.get('Mcp-Session-Id')
            
            # Чтение тела запроса
            body = await request.json()
            
            # Обработка JSON-RPC сообщения
            if isinstance(body, list):
                # Batch request
                results = []
                for message in body:
                    result = await self.process_jsonrpc_message(message, session_id)
                    if result:
                        results.append(result)
                
                if any('method' in msg for msg in body if isinstance(msg, dict)):
                    # Есть requests - возвращаем SSE stream
                    return await self.create_sse_response(results, session_id, request=request)
                else:
                    # Только responses/notifications - возвращаем 202
                    return web.Response(status=202)
            else:
                # Single message
                result = await self.process_jsonrpc_message(body, session_id)
                
                if 'method' in body:
                    # Request - возвращаем SSE stream или JSON
                    if 'text/event-stream' in accept_header:
                        return await self.create_sse_response([result] if result else [], session_id, request=request)
                    else:
                        # Добавляем session ID в заголовок для initialize
                        if result and result.get('_session_id'):
                            session_id = result['_session_id']
                            del result['_session_id']
                            response = web.json_response(result)
                            response.headers['Mcp-Session-Id'] = session_id
                            return response
                        return web.json_response(result)
                else:
                    # Response/notification - возвращаем 202
                    return web.Response(status=202)
                    
        except Exception as e:
            logger.error(f"Ошибка обработки POST запроса: {e}")
            return web.json_response(
                {"error": str(e)}, 
                status=400
            )
    
    async def handle_get(self, request):
        """Обработка GET запросов (получение сообщений от сервера)"""
        # Проверка заголовка Accept
        accept_header = request.headers.get('Accept', '')
        if 'text/event-stream' not in accept_header:
            return web.Response(status=405, text="Method Not Allowed")
        
        # Получение session ID
        session_id = request.headers.get('Mcp-Session-Id')
        
        # Создание SSE stream
        return await self.create_sse_response([], session_id, listen_mode=True, request=request)
    
    async def handle_delete(self, request):
        """Обработка DELETE запросов (завершение сессии)"""
        session_id = request.headers.get('Mcp-Session-Id')
        
        if session_id and session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Сессия {session_id} завершена")
            return web.Response(status=200)
        else:
            return web.Response(status=404, text="Session not found")
    
    async def process_jsonrpc_message(self, message: Dict[str, Any], session_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Обработка JSON-RPC сообщения"""
        try:
            if 'method' in message:
                # Это request
                method = message['method']
                params = message.get('params', {})
                request_id = message.get('id')
                
                logger.info(f"Обработка метода: {method}")
                
                if method == 'initialize':
                    return await self.handle_initialize(params, request_id, session_id)
                elif method == 'tools/list':
                    return await self.handle_list_tools(request_id)
                elif method == 'tools/call':
                    return await self.handle_call_tool(params, request_id)
                elif method == 'resources/list':
                    return await self.handle_list_resources(request_id)
                elif method == 'prompts/list':
                    return await self.handle_list_prompts(request_id)
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}"
                        }
                    }
            else:
                # Это response или notification
                logger.info(f"Получен response/notification: {message}")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения: {e}")
            return {
                "jsonrpc": "2.0",
                "id": message.get('id'),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    async def handle_initialize(self, params: Dict[str, Any], request_id: Any, session_id: Optional[str]) -> Dict[str, Any]:
        """Обработка initialize запроса"""
        # Создаем новую сессию если её нет
        if not session_id:
            session_id = self.create_session_id()
            
        self.sessions[session_id] = {
            "protocol_version": params.get("protocolVersion", "2025-03-26"),
            "client_info": params.get("clientInfo", {}),
            "capabilities": params.get("capabilities", {}),
            "created_at": asyncio.get_event_loop().time()
        }
        
        logger.info(f"Инициализирована сессия: {session_id}")
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {
                    "tools": {
                        "listChanged": False
                    },
                    "resources": {
                        "subscribe": False,
                        "listChanged": False
                    },
                    "prompts": {
                        "listChanged": False
                    }
                },
                "serverInfo": {
                    "name": "MCP Vector Search Server",
                    "version": "1.0.0"
                }
            },
            "_session_id": session_id  # Будет использовано для заголовка
        }
    
    async def handle_list_tools(self, request_id: Any) -> Dict[str, Any]:
        """Обработка tools/list запроса"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": "search_documents",
                        "description": "Поиск по документам в векторной базе данных",
                        "inputSchema": {
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
                    },
                    {
                        "name": "get_server_info",
                        "description": "Получить информацию о сервере",
                        "inputSchema": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                ]
            }
        }
    
    async def handle_call_tool(self, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """Обработка tools/call запроса"""
        tool_name = params.get('name')
        arguments = params.get('arguments', {})
        
        if tool_name == "search_documents":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 5)
            
            logger.info(f"Поиск по запросу: {query}, лимит: {limit}")
            
            # Mock результаты
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
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(results, ensure_ascii=False, indent=2)
                        }
                    ]
                }
            }
            
        elif tool_name == "get_server_info":
            info = {
                "name": "MCP Streamable HTTP Vector Search Server",
                "version": "1.0.0",
                "status": "running",
                "transport": "streamable_http",
                "port": 8080,
                "capabilities": ["search", "info"]
            }
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(info, ensure_ascii=False, indent=2)
                        }
                    ]
                }
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}"
                }
            }
    
    async def handle_list_resources(self, request_id: Any) -> Dict[str, Any]:
        """Обработка resources/list запроса"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "resources": []
            }
        }
    
    async def handle_list_prompts(self, request_id: Any) -> Dict[str, Any]:
        """Обработка prompts/list запроса"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "prompts": []
            }
        }
    
    async def create_sse_response(self, messages: List[Dict], session_id: Optional[str], listen_mode: bool = False, request=None):
        """Создание SSE response"""
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'text/event-stream; charset=utf-8',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Accept, Content-Type, Mcp-Session-Id, Last-Event-ID',
                'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
            }
        )
        
        # Добавляем session ID в заголовок если нужно
        session_id_to_send = None
        for msg in messages:
            if msg.get('_session_id'):
                session_id_to_send = msg['_session_id']
                break
        
        if session_id_to_send:
            response.headers['Mcp-Session-Id'] = session_id_to_send
        
        await response.prepare(request)
        
        try:
            # Отправляем сообщения
            for i, message in enumerate(messages):
                # Удаляем служебное поле
                if '_session_id' in message:
                    del message['_session_id']
                
                # Формируем SSE событие с ID согласно спецификации
                event_id = f"msg-{int(asyncio.get_event_loop().time() * 1000)}-{i}"
                event_data = f"id: {event_id}\ndata: {json.dumps(message, ensure_ascii=False)}\n\n"
                await response.write(event_data.encode('utf-8'))
            
            if listen_mode:
                # В режиме прослушивания держим соединение открытым
                # В реальной реализации здесь можно отправлять уведомления от сервера
                await asyncio.sleep(0.1)  # Минимальная задержка
            
        except Exception as e:
            logger.error(f"Ошибка отправки SSE: {e}")
        finally:
            await response.write_eof()
        
        return response

async def health_check(request):
    """Проверка состояния сервера"""
    return web.json_response({
        "status": "healthy",
        "service": "MCP Streamable HTTP Vector Search Server",
        "version": "1.0.0",
        "transport": "streamable_http",
        "port": 8080,
        "mcp_endpoint": "/mcp"
    })

async def create_app():
    """Создание веб-приложения"""
    app = web.Application()
    
    # Создание MCP сервера
    mcp_server = MCPStreamableServer()
    
    # Простые маршруты без сложного CORS
    app.router.add_post('/mcp', mcp_server.handle_mcp_endpoint)
    app.router.add_get('/mcp', mcp_server.handle_mcp_endpoint)
    app.router.add_delete('/mcp', mcp_server.handle_mcp_endpoint)
    app.router.add_options('/mcp', mcp_server.handle_mcp_endpoint)
    app.router.add_get('/health', health_check)
    
    return app

async def main():
    """Главная функция"""
    logger.info("Запуск MCP Streamable HTTP сервера на порту 8080")
    
    # Создание приложения
    app = await create_app()
    
    # Запуск сервера
    runner = web.AppRunner(app)
    await runner.setup()
    
    # Привязываемся только к localhost для безопасности (согласно спецификации MCP)
    site = web.TCPSite(runner, '127.0.0.1', 8080)
    await site.start()
    
    logger.info("🚀 MCP Streamable HTTP сервер запущен на http://127.0.0.1:8080")
    logger.info("🔗 MCP endpoint: http://127.0.0.1:8080/mcp")
    logger.info("🏥 Health check: http://127.0.0.1:8080/health")
    logger.info("📋 Спецификация: MCP Streamable HTTP Transport 2025-03-26")
    
    # Держим сервер запущенным
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Остановка сервера...")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
