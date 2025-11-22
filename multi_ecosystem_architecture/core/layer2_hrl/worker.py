# core/layer2_hrl/worker.py

import random
import logging
from .neural_uvfa import NeuralUVFA

logger = logging.getLogger("Système.HRLWorker")

class HRLWorker:
    """
    Hierarchical Reinforcement Learning (HRL) Worker.
    Executes low-level tasks and uses a Neural UVFA to learn.
    """
    def __init__(self, worker_id: str, action_space: list):
        self.worker_id = worker_id
        # The NeuralUVFA is the worker's "brain"
        self.uvfa = NeuralUVFA(action_space=action_space)
        self.episode_experiences = []
        logger.info(f"HRLWorker {worker_id} initialized with action space: {action_space}")

    def execute_task(self, state: dict, goal: str) -> tuple[str, bool]:
        """
        Execute a low-level sub-task based on the current state and goal.

        :param state: The current state of the environment.
        :param goal: The goal to achieve.
        :return: A tuple of (achieved_goal, success).
        """
        # 1. Select an action using the neural network
        action = self.uvfa.select_action(state, goal)
        logger.info(f"Worker {self.worker_id} selected action: '{action}' for goal '{goal}'")

        # 2. Simulate task execution
        success = random.choice([True, False, False]) # Skew towards failure to make learning more meaningful
        reward = 1.0 if success else -0.5 # Reward for success, penalty for failure

        # 3. Determine next state and achieved goal
        next_state = state.copy()
        if success:
            achieved_goal = goal
            # Simulate state change on success
            next_state['status'] = f"completed_{action}"
            next_state['steps'] = state.get('steps', 0) + 1
        else:
            achieved_goal = f"failed_to_{action}"
            next_state['status'] = "failed"

        # 4. Store the experience in the UVFA's replay buffer
        self.uvfa.store_experience(
            state=state,
            goal=goal,
            action=action,
            reward=reward,
            next_state=next_state,
            achieved_goal=achieved_goal,
            done=success
        )

        # Store experience for HER at the end of an episode
        self.episode_experiences.append(self.uvfa.replay_buffer[-1])

        if success:
            logger.info(f"  Action '{action}' completed successfully.")
        else:
            logger.info(f"  Action '{action}' failed.")

        return achieved_goal, success

    def learn(self) -> float | None:
        """
        Train the worker's neural network on a batch of experiences.
        :return: The training loss, or None if not enough experiences.
        """
        logger.info(f"Worker {self.worker_id} initiating learning cycle.")
        loss = self.uvfa.train_batch()
        if loss is not None:
            logger.info(f"  Training batch complete. Loss: {loss:.4f}")
        else:
            logger.warning("  Learning skipped: not enough experiences in buffer.")
        return loss

    def end_of_episode(self):
        """Called at the end of an episode to apply HER."""
        logger.info(f"Worker {self.worker_id} applying Hindsight Experience Replay.")
        self.uvfa.apply_her(self.episode_experiences)
        self.episode_experiences = []


if __name__ == '__main__':
    # Configure logging for the example
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example Usage
    actions = ['clean_data', 'engineer_features', 'validate_model']
    worker = HRLWorker('worker_1', action_space=actions)

    initial_state = {"status": "raw_data", "quality": 0.3}
    goal = "model_is_validated"

    # Get Q-value before learning
    q_before = worker.uvfa.get_q_value(initial_state, goal, 'clean_data')

    # Simulate an episode of 10 steps
    print("\n--- Simulating Episode ---")
    current_state = initial_state.copy()
    for i in range(35): # More steps to ensure buffer is filled
        achieved, success = worker.execute_task(current_state, goal)
        current_state['status'] = achieved
        # Learn every few steps
        if (i + 1) % 3 == 0:
            worker.learn()

    # Apply HER at the end of the episode
    worker.end_of_episode()

    # Run a few more learning cycles to ensure HER experiences are used
    print("\n--- Final Learning Cycles ---")
    for _ in range(5):
        worker.learn()

    # Get Q-value after learning
    q_after = worker.uvfa.get_q_value(initial_state, goal, 'clean_data')

    print("\n--- Verification ---")
    print(f"Q-value for ('clean_data' | initial_state, goal) before learning: {q_before:.6f}")
    print(f"Q-value for ('clean_data' | initial_state, goal) after learning:  {q_after:.6f}")

    if abs(q_before - q_after) > 1e-9:
        print("\n✅ SUCCESS: Worker's Q-values have changed. Learning is active.")
    else:
        print("\n❌ FAILURE: Worker's Q-values did not change. Learning is not working.")
