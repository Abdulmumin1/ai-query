
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from ai_query import Agent
from ai_query.agents.storage import MemoryStorage

@pytest.mark.asyncio
async def test_agent_start_concurrency():
    """Verify that concurrent calls to start() only access storage once."""
    
    # Mock storage with delay to simulate race condition window
    mock_storage = AsyncMock(spec=MemoryStorage)
    get_count = 0
    
    async def slow_get(key):
        nonlocal get_count
        if "state" in key:
            get_count += 1
            print(f"GET called for {key} (Call #{get_count})")
            await asyncio.sleep(0.1) # Force context switch
        return None
        
    mock_storage.get.side_effect = slow_get
    
    # Create agent
    agent = Agent("test-agent", storage=mock_storage)
    
    # Run two start calls concurrently
    print("Launching concurrent starts...")
    task1 = asyncio.create_task(agent.start())
    task2 = asyncio.create_task(agent.start())
    
    await asyncio.gather(task1, task2)
    
    print(f"Total storage accesses: {get_count}")
    
    # Verify idempotency
    assert get_count == 1, f"Expected 1 storage access, got {get_count}. Race condition detected!"
    assert agent._state is not None

if __name__ == "__main__":
    asyncio.run(test_agent_start_concurrency())
