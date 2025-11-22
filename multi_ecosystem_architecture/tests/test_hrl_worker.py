# tests/test_hrl_worker.py
import pytest
import random
from core.layer2_hrl.worker import HRLWorker

def test_worker_learns_and_updates_q_values():
    """
    Tests that the HRLWorker's Q-values change after a learning cycle.
    This verifies that the NeuralUVFA is integrated and the learning loop is active.
    """
    # Set a seed for reproducibility
    random.seed(0)

    # 1. Setup
    actions = ['clean_data', 'engineer_features', 'validate_model']
    worker = HRLWorker('test_worker', action_space=actions)

    initial_state = {"status": "raw_data", "quality": 0.3}
    goal = "model_is_validated"
    action_to_test = "clean_data"

    # 2. Get initial Q-value
    q_before = worker.uvfa.get_q_value(initial_state, goal, action_to_test)

    # 3. Simulate an episode to gather experience
    current_state = initial_state.copy()
    for _ in range(worker.uvfa.batch_size + 10):  # Ensure enough experiences for a batch
        achieved, success = worker.execute_task(current_state, goal)
        current_state['status'] = achieved
        if success:
            current_state = initial_state.copy() # Reset on success

    # 4. Trigger multiple learning cycles to ensure weight updates propagate
    loss = None
    for _ in range(10): # Run multiple learning steps
        current_loss = worker.learn()
        if current_loss is not None:
            loss = current_loss

    # 5. Get Q-value after learning
    q_after = worker.uvfa.get_q_value(initial_state, goal, action_to_test)

    # 6. Verification
    assert loss is not None, "Learning did not occur (loss is None)."
    assert q_before != q_after, "Q-values did not change after learning."
