"""AWS Lambda adapter for serving agents."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_query.agents.agent import Agent


def handle_lambda(agent: "Agent", event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Handle an AWS Lambda event for an agent.
    
    Supports API Gateway Proxy Integration (v1 and v2).
    
    Usage:
        # lambda_function.py
        from ai_query.adapters.aws import handle_lambda
        from my_agent import agent
        
        def lambda_handler(event, context):
            return handle_lambda(agent, event, context)
    """
    # 1. Parse Request Body
    body = {}
    if "body" in event:
        raw_body = event["body"]
        if raw_body:
            try:
                body = json.loads(raw_body)
            except json.JSONDecodeError:
                return _error_response(400, "Invalid JSON body")
    
    # 2. Run Agent
    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        result = loop.run_until_complete(agent.handle_request(body))
        
        # 3. Format Response
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": json.dumps(result)
        }
        
    except Exception as e:
        return _error_response(500, str(e))


def _error_response(status: int, message: str) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({"error": message})
    }
