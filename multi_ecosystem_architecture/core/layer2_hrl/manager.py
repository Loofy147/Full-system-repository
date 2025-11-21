# core/layer2_hrl/manager.py

import json

class HRLManager:
    """
    Hierarchical Reinforcement Learning (HRL) Manager.
    Integrates with the ACE (Accumulated, Curated, Evolving) Playbook.
    """
    def __init__(self, playbook_path=None):
        self.playbook = self._load_playbook(playbook_path)
        self.workers = {}
        print("HRLManager initialized.")

    def _load_playbook(self, playbook_path):
        """
        Load the ACE Playbook.

        :param playbook_path: Path to the playbook JSON file.
        :return: A dictionary representing the playbook.
        """
        if playbook_path:
            try:
                with open(playbook_path, 'r') as f:
                    print(f"ACE Playbook loaded from {playbook_path}.")
                    return json.load(f)
            except FileNotFoundError:
                print(f"Warning: Playbook file not found at {playbook_path}.")
                return {}
        return {}

    def register_worker(self, worker_id, skills):
        """
        Register a worker agent.

        :param worker_id: The unique ID for the worker.
        :param skills: A list of skills the worker possesses.
        """
        self.workers[worker_id] = {'skills': skills, 'status': 'idle'}
        print(f"Worker {worker_id} registered with skills: {skills}")

    def delegate_task(self, task):
        """
        Delegate a high-level task to a suitable worker based on the playbook.

        :param task: A dictionary describing the task.
        :return: The ID of the worker assigned to the task.
        """
        strategy = self.playbook.get(task.get('type'))
        if not strategy:
            print(f"No strategy in playbook for task type: {task.get('type')}")
            return None

        required_skill = strategy.get('required_skill')
        for worker_id, worker_info in self.workers.items():
            if required_skill in worker_info['skills'] and worker_info['status'] == 'idle':
                worker_info['status'] = 'busy'
                print(f"Task '{task['description']}' delegated to worker {worker_id} using strategy '{strategy['name']}'.")
                return worker_id

        print(f"No available worker found for skill '{required_skill}'.")
        return None

if __name__ == '__main__':
    # Example Usage
    manager = HRLManager(playbook_path='dummy_playbook.json')
    manager.register_worker('worker_etl_1', ['etl', 'data_cleaning'])
    manager.register_worker('worker_ml_1', ['ml_training', 'data_visualization'])

    task1 = {'type': 'data_processing', 'description': 'Process customer data.'}
    task2 = {'type': 'model_training', 'description': 'Train a churn prediction model.'}

    manager.delegate_task(task1)
    manager.delegate_task(task2)
