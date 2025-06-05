import asyncio
from typing import Dict, Any, List, Tuple
import json
import traceback

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

class MCPClient:
    """MCP Client wrapper to properly manage connections"""
    
    def __init__(self, config_path: str, server_name: str):
        self.config_path = config_path
        self.server_name = server_name
        self.session = None
        self._context_manager = None
        
    async def connect(self) -> ClientSession:
        """Connect to an MCP server using configuration"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        server_config = config['mcpServers'][self.server_name]
        
        # Правильно извлекаем URL в зависимости от типа конфигурации
        if 'command' in server_config:
            # Для серверов с командой (например, uvx mcp-proxy)
            server_url = server_config['args'][-1]
        else:
            # Для прямых HTTP серверов
            server_url = server_config['args'][0]
        
        print(f"Connecting to URL: {server_url}")
        
        try:
            # Сохраняем контекст-менеджер для правильного управления жизненным циклом
            self._context_manager = streamablehttp_client(server_url)
            read_stream, write_stream, _ = await self._context_manager.__aenter__()
            
            # Создаем сессию
            self.session = ClientSession(read_stream, write_stream)
            await self.session.__aenter__()
            
            # Инициализируем соединение
            await self.session.initialize()
            
            print("Successfully connected to MCP server!")
            return self.session
            
        except Exception as e:
            print(f"Connection error: {str(e)}")
            print("Traceback:")
            print(traceback.format_exc())
            await self.close()
            raise
    
    async def close(self):
        """Properly close all connections"""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
                self.session = None
            
            if self._context_manager:
                await self._context_manager.__aexit__(None, None, None)
                self._context_manager = None
                
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

async def connect_to_server(config_path: str, server_name: str) -> MCPClient:
    """Connect to an MCP server using configuration - returns managed client"""
    client = MCPClient(config_path, server_name)
    await client.connect()
    return client

async def get_available_tools(client: MCPClient) -> List[str]:
    """Get list of available tools from the server"""
    try:
        if not client.session:
            raise Exception("Not connected to server")
            
        response = await client.session.list_tools()
        return [tool.name for tool in response.tools]
    except Exception as e:
        print(f"Error getting tools: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        raise

async def execute_tool(client: MCPClient, tool_name: str, parameters: Dict[str, Any]) -> Any:
    """Execute a tool with given parameters"""
    try:
        if not client.session:
            raise Exception("Not connected to server")
            
        return await client.session.call_tool(tool_name, parameters)
    except Exception as e:
        print(f"Error executing tool: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        raise

async def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python mcp_client_SSE.py <config_path> <server_name>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    server_name = sys.argv[2]
    
    client = None

    try:
        print(f"Connecting to server {server_name}...")
        client = await connect_to_server(config_path, server_name)
        print("Connected successfully!")

        print("\nGetting available tools...")
        tools = await get_available_tools(client)
        print("Available tools:", tools)
        
        # Пример использования: получаем ресурсы если доступны
        try:
            print("\nGetting available resources...")
            resources = await client.session.list_resources()
            print("Available resources:", [res.name for res in resources.resources])
        except Exception as e:
            print(f"No resources available or error: {str(e)}")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
    finally:
        if client:
            await client.close()
            print("Connection closed.")

if __name__ == "__main__":
    asyncio.run(main())