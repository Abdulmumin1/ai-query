"""Vercel adapter for serving agents."""

from __future__ import annotations

import asyncio
import json
from http.server import BaseHTTPRequestHandler
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_query.agents.agent import Agent


def handle_vercel(agent: "Agent", handler: BaseHTTPRequestHandler) -> None:
    """Handle a Vercel/HTTP server request for an agent.
    
    This helper is designed for the standard Python runtime on Vercel which uses
    BaseHTTPRequestHandler. It bridges the sync handler to the async agent.

    Usage:
        # api/index.py
        from http.server import BaseHTTPRequestHandler
        from ai_query.adapters.vercel import handle_vercel
        from my_agent import agent

        class handler(BaseHTTPRequestHandler):
            def do_POST(self):
                handle_vercel(agent, self)
            
            def do_GET(self):
                handle_vercel(agent, self)
    """
    # Parse request body
    content_length = int(handler.headers.get("Content-Length", 0))
    if content_length > 0:
        body_bytes = handler.rfile.read(content_length)
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            handler.send_response(400)
            handler.end_headers()
            handler.wfile.write(b"Invalid JSON")
            return
    else:
        body = {}

    # Map GET query params to body (limited support)
    # TODO: Parse query string for simple GET requests if needed

    # Run agent logic
    try:
        # Create a new loop if needed (Vercel runtime might not have one active)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(agent.handle_request(body))

        # Send response
        response_body = json.dumps(result).encode("utf-8")
        
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(response_body)

    except Exception as e:
        handler.send_response(500)
        handler.end_headers()
        handler.wfile.write(str(e).encode("utf-8"))
