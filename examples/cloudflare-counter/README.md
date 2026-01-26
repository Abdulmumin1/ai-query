# Cloudflare Workers Counter Example

This example demonstrates a stateful "Counter" agent running on Cloudflare Workers using `ai-query` and Durable Objects.

## Features

- **Persistence**: The counter value persists even if the Worker restarts.
- **Real-time**: Connected WebSocket clients receive live updates when the counter increments.
- **Serverless**: Zero infrastructure management.

## Files

- `worker.py`: The Python code defining the Agent and the Worker.
- `wrangler.toml`: The configuration file for Cloudflare deployment.

## how to Run

### 1. Requirements

- `npm` (Node.js)
- `python` 3.12+

### 2. Install Wrangler

```bash
npm install -g wrangler
```

### 3. Deploy

1. **Set your API Key** (for the AI):

   ```bash
   npx wrangler secret put GEMINI_API_KEY
   # Or OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
   ```

   _Note: Using Gemini is recommended as it's the default._

2. **Deploy**:
   ```bash
   wrangler deploy
   ```

### 4. Interact

**Using the Python Client (Recommended):**

```bash
# Install dependencies
pip install aiohttp

# Run the interactive client
python client.py https://ai-query-counter.<your-subdomain>.workers.dev
```

**Or manually via curl:**

```bash
# Replace with your deployed URL
curl -X POST https://ai-query-counter.<your-subdomain>.workers.dev/agent/counter-1/invoke \
  -H "Content-Type: application/json" \
  -d '{"payload": {"method": "increment"}}'
```

**Connect via WebSocket:**

```javascript
// Open browser console
const ws = new WebSocket(
  "wss://ai-query-counter.<your-subdomain>.workers.dev/agent/counter-1",
);
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```
