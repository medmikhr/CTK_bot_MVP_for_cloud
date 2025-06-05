import asyncio
from typing import Dict, Any, List
import json
import traceback

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def connect_to_server(config_path: str, server_name: str) -> ClientSession:
    """Connect to an MCP server using configuration"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    server_config = config['mcpServers'][server_name]
    server_url = server_config['args'][-1]  # Получаем URL из конфигурации
    
    print(f"Connecting to URL: {server_url}")
    
    try:
        # Подключаемся к серверу через streamable HTTP
        async with streamablehttp_client(server_url) as (read_stream, write_stream, _):
            # Создаем сессию
            async with ClientSession(read_stream, write_stream) as session:
                # Инициализируем соединение
                await session.initialize()
                return session
    except Exception as e:
        print(f"Connection error: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        raise

async def get_available_tools(session: ClientSession) -> List[str]:
    """Get list of available tools from the server"""
    try:
        response = await session.list_tools()
        return [tool.name for tool in response.tools]
    except Exception as e:
        print(f"Error getting tools: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        raise

async def execute_tool(session: ClientSession, tool_name: str, parameters: Dict[str, Any]) -> Any:
    """Execute a tool with given parameters"""
    try:
        return await session.call_tool(tool_name, parameters)
    except Exception as e:
        print(f"Error executing tool: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        raise

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