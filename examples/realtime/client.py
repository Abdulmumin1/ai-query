"""WebSocket + SSE client for the real-time chat room.

Usage:
    uv run examples/realtime/client.py --username Alice

Commands:
    - Type a message and press Enter to send
    - Include @ai to get an AI response (streamed via SSE)
    - Press Ctrl+C to quit
"""

import asyncio
import argparse
import aiohttp


async def main(username: str, host: str = "localhost", port: int = 8080):
    ws_url = f"ws://{host}:{port}/ws?username={username}"
    sse_url = f"http://{host}:{port}/events"
    
    print(f"Connecting to {ws_url}...")
    print(f"SSE: {sse_url}")
    
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
            print("Connected! Type messages and press Enter. Use @ai to talk to the AI.\n")
            
            # Task to receive WebSocket messages
            async def receive_ws():
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        # Skip state broadcasts for cleaner output
                        if not msg.data.startswith('{"type": "state"'):
                            print(f"\r{msg.data}")
                            print("> ", end="", flush=True)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"Error: {ws.exception()}")
                        break
            
            # Task to receive SSE events (AI streaming)
            async def receive_sse():
                try:
                    async with session.get(sse_url) as resp:
                        async for line in resp.content:
                            text = line.decode().strip()
                            if text.startswith("event: ai_start"):
                                print("\r[AI] ", end="", flush=True)
                            elif text.startswith("data: ") and not text.startswith("data: event:"):
                                chunk = text[6:]  # Remove "data: " prefix
                                if chunk:
                                    print(chunk, end="", flush=True)
                            elif text.startswith("event: ai_end"):
                                print()  # Newline after AI response
                                print("> ", end="", flush=True)
                except Exception as e:
                    print(f"SSE error: {e}")
            
            # Task to send messages
            async def send():
                loop = asyncio.get_event_loop()
                while True:
                    try:
                        message = await loop.run_in_executor(None, input, "> ")
                        if message:
                            await ws.send_str(message)
                    except EOFError:
                        break
            
            # Run all tasks
            ws_task = asyncio.create_task(receive_ws())
            sse_task = asyncio.create_task(receive_sse())
            send_task = asyncio.create_task(send())
            
            try:
                await asyncio.gather(ws_task, sse_task, send_task)
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat room client")
    parser.add_argument("--username", "-u", default="Anonymous", help="Your username")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Server port")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.username, args.host, args.port))
    except KeyboardInterrupt:
        print("\nDisconnected.")
