
import json
from typing import Any, AsyncIterator, Dict, Optional, Union

try:
    import js
    from pyodide.ffi import to_js
except ImportError:
    js = None
    to_js = lambda x: x

from ai_query.agents.transport.base import AgentTransport


class DurableObjectTransport(AgentTransport):
    """Transport for communicating with agents running in Durable Objects.
    
    This transport uses the Cloudflare Workers `stub.fetch()` API to send
    requests directly to Durable Object instances without leaving the
    Cloudflare network.
    """

    def __init__(self, namespace: Any):
        """Initialize with a Durable Object binding.

        Args:
            namespace: The DO Namespace binding (e.g. env.MY_AGENT).
        """
        self.namespace = namespace

    def _get_stub(self, agent_id: str) -> Any:
        """Get the DO stub for a given agent ID."""
        # We use idFromName for named agents (stable IDs).
        # We could support hex tokens if needed, but names are standard
        # for our registry pattern.
        do_id = self.namespace.idFromName(agent_id)
        return do_id.getStub()

    async def invoke(
        self, 
        agent_id: str, 
        payload: Dict[str, Any], 
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Invoke a DO agent."""
        stub = self._get_stub(agent_id)
        
        # Prepare the standardized request body
        body_py = {
            "action": "invoke",
            "payload": payload
        }
        
        # We construct the JS Request options
        # Note: Pyodide creates a JS object wrapper for dicts automatically
        # but to_js is safer for complex types if needed.
        # Simple dicts work fine usually.
        
        # We assume the AgentDO adapter listens on /invoke (or generic /)
        url = "http://do/invoke"
        
        options = {
            "method": "POST",
            "body": json.dumps(body_py),
            "headers": {"Content-Type": "application/json"}
        }

        # Convert options to JS object for fetch
        # The 'to_js' conversion helps creating a proper JS object with dict=Map behavior check or generic Object
        # Actually in recent Pyodide, passing a dict to a JS function converts it to a Map or Object based on context.
        # But for 'fetch' init object, it expects a POJO.
        # We can use js.Object.fromEntries or just let the bridge handle it if configured.
        # Safer to use explicit conversion if possible, or just build it.
        # For simplicity in this environment, passing the dict usually works for kwargs like usage, 
        # but for the second arg of fetch, it expects an object.
        
        if js:
            js_options = js.Object.fromEntries(to_js(options))
        else:
            js_options = options

        response = await stub.fetch(url, js_options)
        
        if response.status != 200:
            text = await response.text()
            raise RuntimeError(f"Agent invoke failed: {response.status} {text}")
            
        data = await response.json()
        
        # Handle JS Proxy object return
        if hasattr(data, "to_py"):
             data = data.to_py()
        elif isinstance(data, (dict, list, str, int, float, bool, type(None))):
             pass
        else:
            # It's a JS proxy but to_py not found? Should generally work.
            pass

        # Wire Protocol Unpacking (same as HTTPTransport)
        if "error" in data:
            return {"error": data["error"]}
        if "result" in data:
            return {"result": data["result"]}
            
        return data

    async def chat(
        self,
        agent_id: str,
        message: str,
        timeout: float = 30.0
    ) -> str:
        """Send a chat message to a DO agent."""
        stub = self._get_stub(agent_id)
        url = "http://do/chat"
        
        body_py = {
            "action": "chat",
            "message": message
        }
        
        options = {
            "method": "POST",
            "body": json.dumps(body_py),
            "headers": {"Content-Type": "application/json"}
        }

        if js:
            js_options = js.Object.fromEntries(to_js(options))
        else:
            js_options = options

        response = await stub.fetch(url, js_options)
        response_data = await response.json()
        
        if hasattr(response_data, "to_py"):
            response_data = response_data.to_py()
            
        return response_data.get("response", "")

    # TODO: Stream implementation using JS ReadableStream reader
