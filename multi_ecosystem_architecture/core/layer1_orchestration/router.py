# core/layer1_orchestration/router.py

import json

class TaskRouter:
    """
    Message Coordination Protocol (MCP) and Agent-to-Agent (A2A) Task Router.
    Routes tasks to registered agents based on their capabilities.
    """
    def __init__(self):
        self.agents = {}
        print("TaskRouter initialized.")

    def register_agent(self, agent_id, capabilities):
        """
        Register an agent with its capabilities.

        :param agent_id: Unique identifier for the agent.
        :param capabilities: A list of capabilities (e.g., ['data_analysis', 'code_generation']).
        """
        self.agents[agent_id] = capabilities
        print(f"Agent {agent_id} registered with capabilities: {capabilities}")

    def route_task(self, task):
        """
        Route a task to the appropriate agent.

        :param task: A dictionary representing the task, including 'capability_required'.
        :return: The ID of the agent assigned to the task, or None if no suitable agent is found.
        """
        required_capability = task.get('capability_required')
        if not required_capability:
            print("Task is missing 'capability_required'.")
            return None

        for agent_id, capabilities in self.agents.items():
            if required_capability in capabilities:
                print(f"Task routed to agent {agent_id}.")
                return agent_id

        print(f"No agent found with capability '{required_capability}'.")
        return None

    def a2a_communicate(self, sender_id, recipient_id, message):
        """
        Simulate Agent-to-Agent (A2A) communication.

        :param sender_id: The ID of the sending agent.
        :param recipient_id: The ID of the receiving agent.
        :param message: The message to send.
        :return: A formatted JSON message.
        """
        if sender_id not in self.agents or recipient_id not in self.agents:
            print("A2A communication failed: sender or recipient not registered.")
            return None

        communication_packet = {
            'sender': sender_id,
            'recipient': recipient_id,
            'message': message,
            'protocol': 'A2A'
        }
        print(f"A2A message from {sender_id} to {recipient_id}: {message}")
        return json.dumps(communication_packet, indent=2)

if __name__ == '__main__':
    # Example Usage
    router = TaskRouter()

    # Register agents
    router.register_agent('agent_data_analyst', ['data_analysis', 'visualization'])
    router.register_agent('agent_code_generator', ['code_generation', 'debugging'])

    # Define tasks
    task1 = {'id': 1, 'capability_required': 'data_analysis', 'description': 'Analyze sales data.'}
    task2 = {'id': 2, 'capability_required': 'code_generation', 'description': 'Generate a Python script.'}
    task3 = {'id': 3, 'capability_required': 'unsupported_skill', 'description': 'This will fail.'}

    # Route tasks
    router.route_task(task1)
    router.route_task(task2)
    router.route_task(task3)

    # A2A communication
    router.a2a_communicate('agent_data_analyst', 'agent_code_generator', 'Please generate a script to visualize this data.')
