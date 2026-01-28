import asyncio
import argparse
import aiohttp
import json
import sys
from ai_query import connect


async def monitor_websocket(url: str, agent_id: str):
    """Connects to the WebSocket and prints incoming messages."""
    ws_url = f"{url}/agent/{agent_id}"
    ws_url = ws_url.replace("https://", "wss://").replace("http://", "ws://")

    print(f"Connecting to WebSocket: {ws_url} ...")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                print("✓ Connected! Waiting for updates...")
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        print(f"\n[WebSocket] Received: {json.dumps(data, indent=2)}")
                        if data.get("type", "") == "error":
                            print("Error received, exiting.")
                            break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print("ws connection closed with exception %s", ws.exception())
    except Exception as e:
        print(f"WebSocket error: {e}")


async def interactive_loop(url: str, agent_id: str):
    """Sends HTTP requests to increment or chat."""
    agent = connect(f"{url}/agent/{agent_id}")

    print(f"\n[Commands] 'i': increment, 's': stream chat, 'q': quit, anything else: chat")

    while True:
        cmd = await asyncio.to_thread(input, "> ")
        if cmd.lower() == "q":
            break

        try:
            if cmd.lower() == "i":
                # Increment Action using RemoteAgent
                result = await agent.call().increment()
                print(f"[Count] New Value: {result}")
            elif cmd.lower() == "s":
                # Stream chat demo
                user_msg = await asyncio.to_thread(input, "Message to stream: ")
                print("[AI] ", end="", flush=True)
                async for chunk in agent.stream(user_msg):
                    print(chunk, end="", flush=True)
                print()  # newline after stream
            else:
                # Chat Message using RemoteAgent
                response = await agent.chat(cmd)
                print(f"[AI] {response}")

        except Exception as e:
            print(f"Request failed: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Cloudflare Counter Client")
    parser.add_argument(
        "url",
        help="The base URL of your worker (e.g. https://my-worker.user.workers.dev)",
    )
    parser.add_argument("--id", default="counter-1", help="The Agent ID to target")
    args = parser.parse_args()

    # Run WebSocket monitor in background
    ws_task = asyncio.create_task(monitor_websocket(args.url.rstrip("/"), args.id))

    # Run interactive loop
    await interactive_loop(args.url.rstrip("/"), args.id)

    # Cleanup
    ws_task.cancel()
    try:
        await ws_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("Usage: python client.py <WORKER_URL>")
            print(
                "Example: python client.py https://ai-query-counter.my-user.workers.dev"
            )
            sys.exit(1)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
