# tests/test_learning_loop.py
import asyncio
import pytest
from pathlib import Path
import os

from core.integration.systeme import SystemeFactory
from core.layer1_orchestration.async_router import Task

@pytest.fixture
def temp_db_path():
    """Create a temporary database for the test."""
    db_path = "test_learning_loop.db"
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)

async def mock_handler(task: Task):
    """A mock handler that returns a successful result with an insight."""
    return {"success": True, "insight": "New learning!"}

@pytest.mark.asyncio
async def test_learning_loop_adds_bullet_to_playbook(temp_db_path):
    """Verify that a successful task execution adds a new bullet to the ACE Playbook."""
    # 1. Setup: Create a system with a temporary database
    config = SystemeFactory.load_config()
    config.playbook_db_path = Path(temp_db_path)
    systeme = SystemeFactory.create(config)

    router = systeme["router"]
    playbook = systeme["components"]["ace_playbook"]

    # 2. Register a mock agent and handler
    router.register_agent("test_agent", ["test_capability"], mock_handler)

    # 3. Submit a task and wait for it to be processed
    task = Task(id="test_task", capability_required="test_capability", payload={})
    await router.submit_task(task)

    # Start the router in the background and wait for the task to complete
    processing_task = asyncio.create_task(router.start())
    await asyncio.sleep(1.5) # Allow time for the task to be processed
    router.stop()
    await processing_task

    # 4. Verification: Check if the new bullet was added to the playbook
    strategies = await playbook.get_section("strategies")
    assert len(strategies) > 0

    insights = [bullet.content for bullet in strategies]
    assert "New learning!" in insights
