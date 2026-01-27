# Changelog
Note: version releases in the 0.x.y range may introduce breaking changes.

## 1.7.20

- patch: Fix Cloudflare adapter to parse URL path for action type in invoke requests

## 1.7.19

- patch: Add HTTP transport abstraction for Cloudflare Workers multi-runtime support

## 1.7.18

- patch: release

## 1.7.17

- patch: release

## 1.7.16

- patch: Add pyodide-http patching to Cloudflare example to enable correct networking
- patch: fix(cloudflare): resolve SSL support and WebSocket proxy errors in Cloudflare Workers

## 1.7.15

- patch: release

## 1.7.14

- patch: release

## 1.7.13

- patch: Fix Cloudflare headers extraction using js.Object.fromEntries() to avoid iteration issues

## 1.7.12

- patch: Prevent 'borrowed proxy destroyed' error by not storing raw request in ConnectionContext
- patch: Use ssl module import check for robust Cloudflare runtime detection

## 1.7.11

- patch: Fix ConnectionContext initialization in Cloudflare adapter to match the defined signature

## 1.7.10

- patch: Fix Cloudflare WebSocketPair unpacking using .object_values() for compatibility with Pyodide

## 1.7.9

- patch: Centralize aiohttp session creation in BaseProvider to handle Cloudflare SSL support across all providers
- patch: Fix Cloudflare Worker SSL error by disabling SSL verification in Google provider when running in Cloudflare
- patch: Improve Cloudflare runtime detection using sys.platform check and add error logging to Cloudflare adapter

## 1.7.8

- patch: Update Cloudflare emit handler to support SSE broadcasting alongside WebSockets

## 1.7.7

- patch: Fix Cloudflare adapter to correctly inherit class-level model definitions on Agent subclasses
- patch: Implement Cloudflare WebSocket emit handler to enable real-time event broadcasting

## 1.7.6

- patch: Fix Cloudflare Response initialization by explicitly converting options to JS objects

## 1.7.5

- patch: Fix Cloudflare Durable Object stub retrieval using getByName

## 1.7.4

- patch: Add Cloudflare Durable Objects support for stateful agents

## 1.7.3

- patch: Complete unified transport layer implementation
    
    - Implement AgentRegistry for multi-agent routing
    - Add HTTPTransport for remote agent communication
    - Create RemoteAgent client proxy for seamless remote calls
    - Add serverless adapters (FastAPI, Vercel, AWS Lambda)
    - Refactor AgentServer to use registry-based routing
    - Add comprehensive battle test demonstrating distributed workflow
    - Fix Python 3.9 compatibility issues

## 1.7.2

- patch: Refactored agent communication to a fluent, type-safe RPC API and fixed event broadcasting to WebSocket/SSE clients in AgentServer.

## 1.7.1

- patch: feat(agents): unify agents around the v2 architecture and reorganize the import structure for better modularity.

## 1.7.0

- minor: feat: add embeddings support with embed() and embed_many() functions for text vectorization

## 1.6.4

- patch: Added embed and embed_many functions

## 1.6.3

- patch: feat(transport): implemented a durable event replay mechanism

## 1.6.2

- patch: feat(transport): implemented a durable event replay mechanism

## 1.6.1

- patch: feat(transport): implemented a safer transport transport mechanism

## 1.6.0

- minor: refactor(agents): implement mailbox pattern and expand server endpoints

## 1.5.0

- minor: Added xai, groq, deepseek and openrouter as openai compat endpoints

## 1.4.0

- minor: feat: added multi-tenacy kinda logic for websocket and sse

## 1.3.0

- minor: Added support for Stateful agents. Agent base class support persistence, message history, and lifecycle hooks

## 1.2.0

- minor: Experimental MCP support, with stdio, sse and streamable http transport

## 1.1.0

- minor: Refactor tool definition to use `@tool` decorator and `Field` for parameters, updating examples and adding new tests and agent

## 1.0.0

- major: Added tool calling support

## 0.0.3

- patch: docs: added project urls

## 0.0.2

- patch: docs: add comprehensive usage guide and provider configuration details

## 0.0.1

- patch: Refactor providers to use base class fetching and fix OpenAI image fetching
