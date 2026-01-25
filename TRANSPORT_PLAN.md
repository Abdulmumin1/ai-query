# Transport & Runtime Architecture Plan

This document outlines the design for the **Unified Transport Layer** and **Serverless Runtime Adapters** in `ai-query`.

## 1. Vision: Location-Transparent Composability

The goal is to decouple **Agent Logic** from **Infrastructure**.
- **Logic:** "I want to ask the `researcher` agent to find X."
- **Infrastructure:** "The `researcher` agent runs on AWS Lambda / Vercel / Local Process."

Developers should write logic once. The wiring happens at the edge.

## 2. Core Architecture

The system relies on a **Registry** to resolve Agent IDs to their execution logic (Local Class) or location (Remote URL).

```mermaid
graph TD
    Client[Client Code] --> Registry[Agent Registry]
    
    Registry -->|Match: "writer"| Local1[WriterAgent Class]
    Registry -->|Match: "chat-.*"| Local2[ChatAgent Class]
    Registry -->|Match: "remote"| Remote[HTTPTransport]
```

### 2.1 The Agent Registry
The Registry is the heart of the system. It replaces the single-class limitation of the current `AgentServer`.

```python
class AgentRegistry:
    def register(self, pattern: str, target: type[Agent] | AgentTransport):
        """Map an ID pattern to an Agent Class (local) or Transport (remote)."""
        ...
        
    def resolve(self, agent_id: str) -> type[Agent] | AgentTransport:
        """Find the matching target for an agent ID."""
        ...
```

## 3. The Transport Layer

The `AgentTransport` interface abstracts the "how" of communication.

### 3.1 Interface
```python
class AgentTransport(ABC):
    async def send(self, agent_id: str, request: AgentRequest, timeout: float = 30.0) -> AgentResponse:
        """Send a structured request to an agent."""
        ...

    def connect_events(self, agent_id: str) -> AsyncIterator[AgentEvent]:
        """Subscribe to the agent's event stream (SSE/WS)."""
        ...
```

### 3.2 Implementations
1.  **`LocalTransport`**: Calls `agent.handle_request()` directly in memory. Best for development and single-process apps.
2.  **`HTTPTransport`**: Serializes requests to JSON and sends via `httpx`. Handles authentication headers and error mapping.

## 4. Client Experience (DX)

### 4.1 Consuming Agents (`connect`)

The `connect()` function is the universal entry point. It returns a `RemoteAgent` proxy that mimics the local `Agent` API.

```python
from ai_query import connect

# Connect to a remote agent
researcher = connect("https://api.myapp.com/agents/researcher")

# 1. Chat (Request/Response)
# Transport sends: POST /chat { "message": "..." }
response = await researcher.chat("Analyze Q3 data")

# 2. Streaming (SSE)
# Transport connects: GET /events?stream=true
async for chunk in researcher.stream("Summarize"):
    print(chunk, end="")

# 3. RPC (Typed Actions)
# Transport sends: POST /invoke { "method": "get_data", "params": {...} }
data = await researcher.call().get_data(limit=10)
```

### 4.2 Wiring (The Registry)

For complex apps, use a Registry to map logical IDs to physical locations.

```python
# infrastructure.py
registry = AgentRegistry()
registry.register("writer", LocalTransport(WriterAgent()))
registry.register("researcher", HTTPTransport("https://my-lambda.com/agent"))

# app.py (Pure Logic)
class Orchestrator(Agent):
    async def run(self):
        # Doesn't know 'writer' is local and 'researcher' is remote
        res = await self.call("researcher").research()
        await self.call("writer").write(res)
```

## 5. Serverless Runtimes (Adapters)

Adapters eliminate boilerplate by bridging the platform's request object to the Agent's `handle_request`.

### 5.1 FastAPI / Starlette
```python
from fastapi import FastAPI
from ai_query.adapters.fastapi import AgentRouter

app = FastAPI()
agent = MyAgent("id")

# Mounts standard endpoints: /chat, /invoke, /events, /ws
app.include_router(AgentRouter(agent), prefix="/agent/my-id")
```

### 5.2 Vercel / Next.js
```python
# api/agent/[...slug].ts (concept) -> Python equivalent
from ai_query.adapters.vercel import handle_vercel
from my_agent import agent

# Handles generic POST/GET requests automatically
def handler(request):
    return handle_vercel(agent, request)
```

### 5.3 AWS Lambda
```python
from ai_query.adapters.aws import handle_lambda
from my_agent import agent

def lambda_handler(event, context):
    return handle_lambda(agent, event, context)
```

## 6. Request Flow Scenarios

### 6.1 Standard Chat Flow
1.  **Client:** `await agent.chat("Hello")`
2.  **Transport:** POST `/chat` Body: `{"message": "Hello"}`
3.  **Adapter:** Parses JSON, calls `agent.chat("Hello")`
4.  **Agent:** Runs logic, saves state.
5.  **Adapter:** Returns JSON `{"response": "Hi there"}`
6.  **Client:** Returns string `"Hi there"`

### 6.2 Streaming Flow (SSE via POST)
1.  **Client:** `async for chunk in agent.stream("Hello")`
2.  **Transport:** Request `POST /chat` 
    *   Header: `Accept: text/event-stream`
    *   Body: `{"message": "Hello", "stream": true}`
3.  **Adapter:** Detects stream request. Opens SSE stream.
4.  **Agent:** Yields chunks.
5.  **Adapter:** Formats as `data: "He"\n\n`, `data: "llo"\n\n`.
6.  **Client:** Decodes SSE, yields "He", "llo".

*Note: We use POST for streaming to allow complex bodies and unify endpoints. Native `EventSource` is not used; instead we stream the fetch response.*

### 6.3 Events (Emission Strategy)

The `emit()` method must work consistently across all environments.

#### A. In-Band Events (Streaming Response)
When a client requests a stream (POST `/chat` with `stream: true`), events emitted during execution are injected into the SSE stream.

1.  **Agent Logic:** `await self.emit("processing", {"step": 1})`
2.  **Runtime Adapter:**
    *   Hooks into `agent._emit_handler`.
    *   Captures the event.
    *   Yields an SSE event: `event: processing\ndata: {"step": 1}\n\n`
3.  **Client:** Receives `processing` event interleaved with text `chunk` events.

#### B. Out-of-Band Events (Persistence)
For non-streaming requests or background tasks (e.g., webhooks), events cannot be pushed immediately.

1.  **Agent Logic:** `await self.emit("done", ...)`
2.  **Runtime Adapter:**
    *   Ensures `enable_event_log=True` (if configured).
    *   Persists event to `Storage` (Redis/DB).
3.  **Client:**
    *   Polls `GET /events?last_event_id=100`.
    *   Or connects to a separate WebSocket server that reads from the same Storage/EventBus.

### 6.4 WebSockets
1.  **Client:** `await agent.connect()`
2.  **Transport:** Upgrade request `GET /ws`.
3.  **Adapter:** Accepts upgrade.
4.  **Flow:** Bidirectional JSON frames.
    *   Client sends `{"action": "chat", "message": "..."}`
    *   Server sends `{"type": "chunk", "content": "..."}`

## 7. Wire Protocol Specification

**Request (JSON):**
```json
{
  "action": "chat" | "invoke" | "state",
  "message": "Optional (for chat)",
  "payload": {
    "method": "method_name",
    "params": { ... }
  }
}
```

**Response (JSON):**
```json
{
  "agent_id": "string",
  "response": "string (for chat)",
  "result": "any (for invoke)",
  "error": "string (optional)"
}
```

**SSE/Stream Event:**
```
event: chunk | start | end | error
data: <json_string>
```

---
**Next Steps:**
1.  Implement `HTTPTransport`.
2.  Implement `RemoteAgent` proxy.
3.  Implement `AgentRegistry`.
4.  Create `ai_query.adapters` package.
