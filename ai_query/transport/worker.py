"""HTTP transport using JS fetch API for Cloudflare Workers."""

from typing import Any, AsyncIterator

from .base import HTTPTransport


class WorkerFetchTransport(HTTPTransport):
    """HTTP transport using JS fetch API for Cloudflare Workers.

    This transport uses the browser's Fetch API (exposed via JS interop)
    to make HTTP requests without requiring sockets.
    """

    async def post(
        self,
        url: str,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Make a POST request and return JSON response."""
        import json as json_module

        from js import Object, fetch
        from pyodide.ffi import to_js

        fetch_headers = dict(headers) if headers else {}
        fetch_headers["Content-Type"] = "application/json"

        options = to_js(
            {
                "method": "POST",
                "headers": fetch_headers,
                "body": json_module.dumps(json),
            },
            dict_converter=Object.fromEntries,
        )

        resp = await fetch(url, options)
        if not resp.ok:
            text = await resp.text()
            raise Exception(f"HTTP {resp.status}: {text}")

        data = await resp.json()
        return data.to_py() if hasattr(data, "to_py") else dict(data)

    async def stream(
        self,
        url: str,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[bytes]:
        """Make a POST request and stream response bytes."""
        import json as json_module

        from js import Object, fetch
        from pyodide.ffi import to_js

        fetch_headers = dict(headers) if headers else {}
        fetch_headers["Content-Type"] = "application/json"

        options = to_js(
            {
                "method": "POST",
                "headers": fetch_headers,
                "body": json_module.dumps(json),
            },
            dict_converter=Object.fromEntries,
        )

        resp = await fetch(url, options)
        if not resp.ok:
            text = await resp.text()
            raise Exception(f"HTTP {resp.status}: {text}")

        reader = resp.body.getReader()
        while True:
            result = await reader.read()
            if result.done:
                break
            # Convert JS Uint8Array to Python bytes
            chunk_data = result.value
            if hasattr(chunk_data, "to_py"):
                yield bytes(chunk_data.to_py())
            else:
                yield bytes(chunk_data)

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> tuple[bytes, str]:
        """Make a GET request and return body bytes with content type."""
        from js import Object, fetch
        from pyodide.ffi import to_js

        options = to_js(
            {
                "method": "GET",
                "headers": headers or {},
            },
            dict_converter=Object.fromEntries,
        )

        resp = await fetch(url, options)
        if not resp.ok:
            text = await resp.text()
            raise Exception(f"HTTP {resp.status}: {text}")

        buffer = await resp.arrayBuffer()
        content_type = resp.headers.get("content-type") or "application/octet-stream"

        # Convert JS ArrayBuffer to Python bytes
        if hasattr(buffer, "to_py"):
            return bytes(buffer.to_py()), content_type
        return bytes(buffer), content_type
