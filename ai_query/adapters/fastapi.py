"""FastAPI adapter for serving agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

try:
    from fastapi import APIRouter, Request, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
except ImportError:
    # Allow import without optional dependencies
    APIRouter = object # type: ignore

if TYPE_CHECKING:
    from ai_query.agents.agent import Agent
    from ai_query.agents.registry import AgentRegistry


class AgentRouter(APIRouter):
    """FastAPI Router for serving Agents.
    
    Supports serving a single Agent instance or an AgentRegistry for multi-agent routing.

    Endpoints (Single Agent):
    - POST /chat
    - POST /invoke
    - GET /state
    - POST /

    Endpoints (Registry):
    - POST /{agent_id}/chat
    - POST /{agent_id}/invoke
    - GET /{agent_id}/state
    - POST /{agent_id}
    
    Example:
        # Single Agent
        app.include_router(AgentRouter(my_agent), prefix="/agent/bot")

        # Registry (Multi-Agent)
        registry = AgentRegistry()
        registry.register("worker-.*", WorkerAgent)
        app.include_router(AgentRouter(registry), prefix="/agents")
    """

    def __init__(self, target: Union["Agent", "AgentRegistry"], **kwargs: Any):
        if APIRouter is object:
            raise ImportError("FastAPI is not installed. Run 'pip install fastapi'.")
            
        super().__init__(**kwargs)
        
        from ai_query.agents.registry import AgentRegistry
        
        self.is_registry = isinstance(target, AgentRegistry)
        self.target = target
        
        if self.is_registry:
            # Register routes with agent_id path param
            self.add_api_route("/{agent_id}/chat", self._handle_chat, methods=["POST"])
            self.add_api_route("/{agent_id}/invoke", self._handle_invoke, methods=["POST"])
            self.add_api_route("/{agent_id}/state", self._handle_get_state, methods=["GET"])
            self.add_api_route("/{agent_id}", self._handle_request, methods=["POST"])
        else:
            # Register routes at root
            self.add_api_route("/chat", self._handle_chat, methods=["POST"])
            self.add_api_route("/invoke", self._handle_invoke, methods=["POST"])
            self.add_api_route("/state", self._handle_get_state, methods=["GET"])
            self.add_api_route("/", self._handle_request, methods=["POST"])

    async def _get_agent(self, agent_id: Union[str, None] = None) -> "Agent":
        if self.is_registry:
            if not agent_id:
                raise HTTPException(status_code=400, detail="Missing agent_id")
            try:
                # Resolve returns class or transport
                target = self.target.resolve(agent_id)
                # Instantiate if class
                if isinstance(target, type):
                    return target(agent_id)
                # If transport, wrap in RemoteAgent? 
                # Ideally we shouldn't be routing to remote agents via this router 
                # unless acting as a gateway. For now assuming local execution.
                raise HTTPException(status_code=500, detail="Remote routing not supported in AgentRouter yet")
            except ValueError:
                raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        else:
            return self.target

    async def _ensure_started(self, agent: "Agent") -> None:
        if agent._state is None:
            await agent.start()

    async def _handle_chat(self, request: Request, agent_id: Union[str, None] = None) -> Any:
        agent = await self._get_agent(agent_id)
        await self._ensure_started(agent)
        
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")
            
        message = body.get("message", "")
        if not message:
            raise HTTPException(status_code=400, detail="Missing 'message'")
            
        # Check for streaming
        if request.query_params.get("stream", "").lower() == "true":
            return StreamingResponse(
                self._stream_generator(agent, message),
                media_type="text/event-stream"
            )
            
        result = await agent.handle_request({
            "action": "chat", 
            "message": message
        })
        return JSONResponse(result)

    async def _stream_generator(self, agent: "Agent", message: str) -> Any:
        stream_req = {"action": "chat", "message": message}
        async for chunk in agent.handle_request_stream(stream_req):
            yield chunk

    async def _handle_invoke(self, request: Request, agent_id: Union[str, None] = None) -> Any:
        agent = await self._get_agent(agent_id)
        await self._ensure_started(agent)
        
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")
            
        payload = body.get("payload", body)
        result = await agent.handle_request({
            "action": "invoke",
            "payload": payload
        })
        return JSONResponse(result)

    async def _handle_get_state(self, agent_id: Union[str, None] = None) -> Any:
        agent = await self._get_agent(agent_id)
        await self._ensure_started(agent)
        try:
            return JSONResponse(agent.state)
        except Exception:
            raise HTTPException(status_code=500, detail="State serialization failed")

    async def _handle_request(self, request: Request, agent_id: Union[str, None] = None) -> Any:
        agent = await self._get_agent(agent_id)
        await self._ensure_started(agent)
        
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")
            
        result = await agent.handle_request(body)
        return JSONResponse(result)
