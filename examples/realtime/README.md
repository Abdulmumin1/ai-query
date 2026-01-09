# Real-time Chat Example

A WebSocket chat room with AI assistant.

## Start the Server

```bash
uv run examples/realtime/server.py
```

## Connect Clients

In separate terminals:

```bash
uv run examples/realtime/client.py --username Alice
uv run examples/realtime/client.py --username Bob
```

## Usage

- Type a message and press Enter to send
- Include `@ai` to get an AI response
- Press Ctrl+C to quit
