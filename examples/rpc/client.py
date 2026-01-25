
import asyncio
import argparse
import uuid
import json
from typing import Any, Dict
import aiohttp

# Import the specific agent class for type-hinting
from examples.rpc.server import ChatRoom
# Import the base Agent class to create our client-side dummy agent
from ai_query.agents import Agent
# Import the base Transport class to create a custom websocket transport
from ai_query.agents.transport import AgentTransport


def format_message(data: dict) -> str:
    """Format a message from the server into a user-friendly string."""
    msg_type = data.get("type")
    
    if msg_type == "status":
        text = data.get("text", "...")
        return f"*** {text} ***"
        
    elif msg_type == "message":
        user = data.get("user_id", "anonymous")
        text = data.get("text", "")
        return f"[{user}]: {text}"

    elif msg_type == "reaction":
        user = data.get("user_id", "anonymous")
        reaction = data.get("reaction", "")
        return f"<{user} reacted with {reaction}>"

    # Action results are handled by the transport now, but we keep this for other messages
    elif msg_type == "action_result":
        call_id = data.get("call_id")
        result = data.get("result")
        error = data.get("error")
        prefix = f"Result for '{call_id}'" if call_id else "Result"
        
        if error:
            return f"--> Error: {error}"
        
        # Pretty-print history results
        if isinstance(result, list):
            history_str = "\n".join([f"  - [{m.get('user_id')}]: {m.get('text')}" for m in result])
            return f"--> {prefix}:\n{history_str}"
            
        return f"--> {prefix}: {json.dumps(result, indent=2)}"

    return f"Received raw: {json.dumps(data)}"


class ClientWebsocketTransport(AgentTransport):
    """A custom transport to bridge the fluent API with our WebSocket client."""
    def __init__(self, ws: aiohttp.ClientWebSocketResponse):
        self.ws = ws
        self.futures: Dict[str, asyncio.Future] = {}
        self.request_counter = 0

    async def listen_for_messages(self):
        """Listens for all messages and routes them appropriately."""
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                # If it's a result for a call, resolve the future
                if data.get("type") == "action_result" and data.get("call_id") in self.futures:
                    future = self.futures.pop(data["call_id"])
                    # Return the full data dict so the AgentCallProxy can extract .get("result")
                    future.set_result(data)
                # Otherwise, it's a broadcast message, so format and print it
                else:
                    formatted = format_message(data)
                    print(f"\r{formatted}\n>>> ", end="", flush=True)

    async def invoke(self, agent_id: str, payload: dict, timeout: float = 30.0) -> Any:
        """Sends the RPC call over WebSocket and waits for the result."""
        self.request_counter += 1
        call_id = f"fluent-call-{self.request_counter}"
        
        await self.ws.send_json({
            "type": "action",
            "name": payload["method"],
            "params": payload["params"],
            "call_id": call_id,
        })
        
        future = asyncio.get_event_loop().create_future()
        self.futures[call_id] = future
        return await asyncio.wait_for(future, timeout=timeout)


async def main(user_id: str):
    """Run the client."""
    session = aiohttp.ClientSession()
    agent_id = "test-room"

    try:
        headers = {"X-User-Id": user_id}
        async with session.ws_connect(
            f"http://localhost:8080/agent/{agent_id}/ws", headers=headers
        ) as ws:
            print(f"Connected to agent '{agent_id}' as user '{user_id}'.")
            print("Commands: /history [user_id], /react <emoji>, /greet <name>, /quit")

            transport = ClientWebsocketTransport(ws)
            client_agent = Agent("interactive-client", transport=transport)
            
            listen_task = asyncio.create_task(transport.listen_for_messages())

            while not ws.closed:
                prompt = ">>> "
                try:
                    message = await asyncio.to_thread(input, prompt)
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting...")
                    break

                if message.strip() == "/quit":
                    break

                proxy = client_agent.call(agent_id, agent_cls=ChatRoom)
                
                try:
                    if message.startswith("/history"):
                        parts = message.split(" ", 2)
                        target_user = parts[1] if len(parts) > 1 else None
                        print(f"--> Calling get_history(user_id={target_user})")
                        res = await proxy.get_history(user_id=target_user)
                        print(f"\r{format_message({'type': 'action_result', 'result': res})}\n>>> ", end="", flush=True)

                    elif message.startswith("/react "):
                        parts = message.split(" ", 2)
                        reaction = parts[1] if len(parts) > 1 else "👍"
                        print(f"--> Calling add_reaction(reaction='{reaction}')")
                        res = await proxy.add_reaction(message_id="fluent-msg", reaction=reaction)
                        print(f"\r{format_message({'type': 'action_result', 'result': res})}\n>>> ", end="", flush=True)

                    elif message.startswith("/greet "):
                        name = message.split(" ", 1)[1]
                        print(f"--> Calling greet(name='{name}')")
                        res = await proxy.greet(name=name)
                        print(f"\r{format_message({'type': 'action_result', 'result': res})}\n>>> ", end="", flush=True)

                    elif message:
                        await ws.send_str(message)
                except Exception as e:
                    print(f"\r--> Error: {e}\n>>> ", end="", flush=True)

            if not listen_task.done():
                listen_task.cancel()
            await ws.close()

    except aiohttp.ClientConnectorError as e:
        print(f"\nConnection error: {e}")
    finally:
        await session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPC Agent Client")
    parser.add_argument(
        "--user-id", type=str, default=f"user-{uuid.uuid4().hex[:6]}"
    )
    args = parser.parse_args()
    try:
        asyncio.run(main(args.user_id))
    except KeyboardInterrupt:
        print("\nClient shut down.")
