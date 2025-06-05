import asyncio
from typing import Dict, Any, List
import json
import traceback
import httpx
from datetime import timedelta
import requests
import sseclient

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client, StreamableHTTPTransport

def sse_test_via_requests(server_url):
    headers = {
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "User-Agent": "curl/7.79.1"
    }
    print(f"\n=== SSE TEST VIA REQUESTS + SSECLIENT ===")
    try:
        resp = requests.get(server_url, headers=headers, stream=True, verify=False)
        client = sseclient.SSEClient(resp)
        print("SSE stream opened. Waiting for events...")
        for i, event in enumerate(client.events()):
            print(f"[SSE EVENT {i}] {event.data}")
            if i > 10:
                print("Received 10 events, stopping test.")
                break
    except Exception as sse_e:
        print(f"SSE test error: {sse_e}")

async def connect_to_server(config_path: str, server_name: str) -> ClientSession:
    """Connect to an MCP server using configuration"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    server_config = config['mcpServers'][server_name]
    server_url = server_config['args'][-1]  # Получаем URL из конфигурации
    
    print(f"Connecting to URL: {server_url}")
    
    try:
        # Создаем HTTP клиент без прокси
        async with httpx.AsyncClient(
            verify=False,  # Отключаем проверку SSL для локального соединения
            timeout=30.0
        ) as client:
            # Отправляем GET запрос для SSE соединения
            response = await client.get(
                server_url,
                headers={
                    'Accept': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'User-Agent': 'curl/7.79.1'
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to connect: {response.status_code} {response.text}")
            
            # Создаем сессию
            async with ClientSession(response.aiter_bytes(), None) as session:
                # Инициализируем соединение
                await session.initialize()
                return session
    except Exception as e:
        print(f"Connection error: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        sse_test_via_requests(server_url)
        raise

async def get_available_tools(session: ClientSession) -> List[str]:
    """Get list of available tools from the server"""
    response = await session.list_tools()
    return [tool.name for tool in response.tools]

async def execute_tool(session: ClientSession, tool_name: str, parameters: Dict[str, Any]) -> Any:
    """Execute a tool with given parameters"""
    return await session.call_tool(tool_name, parameters)

async def main():
    import sys
    config_path = sys.argv[1]
    server_name = sys.argv[2]

    try:
        print(f"Connecting to server {server_name}...")
        session = await connect_to_server(config_path, server_name)
        print("Connected successfully!")

        print("\nGetting available tools...")
        tools = await get_available_tools(session)
        print("Available tools:", tools)

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 