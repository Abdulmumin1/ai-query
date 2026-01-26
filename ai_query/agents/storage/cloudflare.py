import json
from typing import Any, List, Optional

class RobustEncoder(json.JSONEncoder):
    """Encodes Python types that aren't natively JSON serializable."""
    def default(self, obj):
        if isinstance(obj, set):
            return {"__ai_query_type__": "set", "items": list(obj)}
        return super().default(obj)

def robust_object_hook(dct):
    """Decodes JSON back into Python types."""
    if "__ai_query_type__" in dct:
        type_name = dct["__ai_query_type__"]
        if type_name == "set":
            return set(dct["items"])
    return dct

try:
    import js
except ImportError:
    # Fallback for local testing/linting outside of Cloudflare
    js = None

from ai_query.agents.storage.base import Storage


class DurableObjectStorage(Storage):
    """Storage implementation for Cloudflare Durable Objects.
    
    This class wraps the `state.storage` API available in Durable Objects.
    It handles serialization/deserialization to JSON as the storage API
    accepts JS objects (which Python dicts bridge to).
    """

    def __init__(self, storage: Any):
        """Initialize with the Durable Object storage object (ctx.storage)."""
        self._storage = storage
        if js is None:
            # We are likely running in a test or non-workerd environment
            pass

    async def get(self, key: str) -> Any | None:
        """Get a value from DO storage."""
        # state.storage.get returns a Promise that resolves to the value
        # or undefined if not found.
        # Python bridge handles the Promise await automatically if we are in async?
        # Actually in Pyodide/workerd, we await the JsProxy Promise.
        val = await self._storage.get(key)
        if val is None or (hasattr(val, "undefined") and val.undefined):
             return None
        
        # We assume values are stored as JSON strings or JS objects.
        # Ideally we store as JSON strings to avoid JS proxy overhead/complexity
        # on complex Python objects.
        try:
            return json.loads(val, object_hook=robust_object_hook)
        except (TypeError, json.JSONDecodeError):
            # If it's already a JS primitive or dict that didn't need parsing
            return val

    async def set(self, key: str, value: Any) -> None:
        """Set a value in DO storage."""
        serialized = json.dumps(value, cls=RobustEncoder)
        await self._storage.put(key, serialized)

    async def delete(self, key: str) -> None:
        """Delete a key."""
        await self._storage.delete(key)

    async def keys(self, prefix: str = "") -> List[str]:
        """List keys, optionally filtering by prefix."""
        # storage.list() returns a Map. 
        # We can use list({ prefix: ... })
        options = js.Object.fromEntries([["prefix", prefix]]) if prefix else None
        
        # This returns a Map<string, any>
        result_map = await self._storage.list(options)
        
        # Convert JS Map keys to Python list
        # result_map.keys() is a JS iterator
        return list(result_map.keys())

    def transaction(self, closure: Any) -> Any:
        """Run a closure within a storage transaction.
        
        Usage:
            await self.storage.transaction(lambda txn: ...)
        """
        return self._storage.transaction(closure)
