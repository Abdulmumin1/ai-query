
import asyncio
import aiohttp
import argparse
import uuid

async def main(user_id: str):
    """Connect to an agent's SSE endpoint and print events."""
    agent_id = "test-room"
    url = f"http://localhost:8080/agent/{agent_id}/events?last_event_id=0"
    
    print(f"Connecting to SSE event stream for agent '{agent_id}' as user '{user_id}'...")
    print(f"URL: {url}")

    try:
        async with aiohttp.ClientSession() as session:
            headers = {"X-User-Id": user_id, "Accept": "text/event-stream"}
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    print(f"Error connecting: {response.status}")
                    body = await response.text()
                    print(body)
                    return

                print("Connection successful. Listening for server events...")
                # The connection will stay open and receive events indefinitely
                while True:
                    line = await response.content.readline()
                    if not line:
                        break
                    line_str = line.decode('utf-8').strip()
                    if line_str:
                        print(f"<<< {line_str}")

    except aiohttp.ClientConnectorError as e:
        print(f"Connection error: {e}")
        print("Is the agent server running? Try: `python examples/rpc_agent.py`")
    except asyncio.CancelledError:
        print("\nClient shut down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent SSE Event Stream Client")
    parser.add_argument(
        "--user-id",
        type=str,
        default=f"sse-user-{uuid.uuid4().hex[:6]}",
        help="Unique ID for the client user.",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.user_id))
    except KeyboardInterrupt:
        print("\nClient shut down.")
