# core/layer1_orchestration/async_router.py
"""
Asynchronous Task Router
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

logger = logging.getLogger("Système")

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    id: str
    capability_required: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: float = field(default_factory=time.time)
    source_agent: str = "unknown"
    correlation_id: Optional[str] = None
    retries: int = 0
    max_retries: int = 3

class AsyncTaskRouter:
    """Enhanced async task router with priority queuing and dead letter handling."""

    def __init__(self, max_queue_size: int = 1000):
        self.agents: Dict[str, Dict] = {}
        self._queues: Dict[TaskPriority, asyncio.PriorityQueue] = {
            p: asyncio.PriorityQueue(maxsize=max_queue_size) for p in TaskPriority
        }
        self._dead_letter_queue: List[Task] = []
        self._handlers: Dict[str, Callable] = {}
        self._running = False
        self._metrics = {"routed": 0, "failed": 0, "dlq": 0}
        logger.info("AsyncTaskRouter initialized")

    def register_agent(self, agent_id: str, capabilities: List[str], handler: Callable):
        """Register an agent with its capabilities and handler."""
        self.agents[agent_id] = {
            "capabilities": set(capabilities),
            "handler": handler,
            "status": "idle",
            "tasks_processed": 0
        }
        for cap in capabilities:
            self._handlers[cap] = handler
        logger.info(f"Agent '{agent_id}' registered with capabilities: {capabilities}")

    async def submit_task(self, task: Task) -> bool:
        """Submit a task to the appropriate priority queue."""
        try:
            # Priority queue uses (priority, timestamp, task) for ordering
            await self._queues[task.priority].put(
                (-task.priority.value, task.created_at, task)
            )
            logger.debug(f"Task {task.id} submitted with priority {task.priority.name}")
            return True
        except asyncio.QueueFull:
            logger.warning(f"Queue full, task {task.id} rejected")
            return False

    async def _process_queue(self, priority: TaskPriority):
        """Process tasks from a specific priority queue."""
        queue = self._queues[priority]
        while self._running:
            try:
                _, _, task = await asyncio.wait_for(queue.get(), timeout=1.0)
                await self._route_task(task)
                queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing queue {priority.name}: {e}")

    async def _route_task(self, task: Task):
        """Route a task to the appropriate handler."""
        handler = self._handlers.get(task.capability_required)

        if not handler:
            logger.warning(f"No handler for capability '{task.capability_required}'")
            self._send_to_dlq(task, "no_handler")
            return

        try:
            result = await handler(task)
            self._metrics["routed"] += 1
            logger.info(f"Task {task.id} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.retries += 1
            if task.retries < task.max_retries:
                await self.submit_task(task)
            else:
                self._send_to_dlq(task, str(e))

    def _send_to_dlq(self, task: Task, reason: str):
        """Send failed task to dead letter queue."""
        self._dead_letter_queue.append(task)
        self._metrics["dlq"] += 1
        self._metrics["failed"] += 1
        logger.warning(f"Task {task.id} sent to DLQ: {reason}")

    async def start(self):
        """Start processing all queues."""
        self._running = True
        tasks = [
            asyncio.create_task(self._process_queue(p))
            for p in TaskPriority
        ]
        logger.info("AsyncTaskRouter started")
        await asyncio.gather(*tasks)

    def stop(self):
        """Stop processing."""
        self._running = False
        logger.info("AsyncTaskRouter stopped")

    def get_metrics(self) -> Dict:
        return {**self._metrics, "dlq_size": len(self._dead_letter_queue)}
