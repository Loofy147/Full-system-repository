# core/layer4_resilience/patterns.py

import time
import random

# --- Retry with Exponential Backoff ---

def retry_with_backoff(retries=3, initial_delay=1, backoff_factor=2):
    """
    A decorator for retrying a function with exponential backoff.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {i + 1}/{retries} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= backoff_factor
            print(f"Function failed after {retries} retries.")
            return None
        return wrapper
    return decorator

# --- Circuit Breaker ---

class CircuitBreaker:
    """
    A Circuit Breaker to prevent repeated calls to a failing service.
    """
    def __init__(self, failure_threshold=3, recovery_timeout=10):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = "CLOSED"  # Can be CLOSED, OPEN, or HALF_OPEN
        self.last_failure_time = None
        print("CircuitBreaker initialized.")

    def execute(self, func):
        """
        Execute a function protected by the circuit breaker.
        """
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                print("Circuit Breaker is now HALF_OPEN.")
            else:
                print("Circuit Breaker is OPEN. Call is blocked.")
                return None

        try:
            result = func()
            self._reset()
            return result
        except Exception as e:
            self._record_failure(e)
            return None

    def _record_failure(self, error):
        self.failures += 1
        print(f"Recorded failure. Total failures: {self.failures}. Error: {error}")
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            self.last_failure_time = time.time()
            print(f"Circuit Breaker is now OPEN for {self.recovery_timeout}s.")

    def _reset(self):
        self.failures = 0
        self.state = "CLOSED"
        print("Circuit Breaker is CLOSED.")

# --- Example Usage ---

@retry_with_backoff(retries=3, initial_delay=0.1, backoff_factor=2)
def potentially_failing_operation():
    """A function that might fail randomly."""
    if random.random() < 0.8:  # 80% chance of failure
        raise ValueError("Service is unavailable")
    print("Operation succeeded.")
    return "Success"

def another_failing_service():
    """Another function that fails."""
    raise ConnectionError("Could not connect to the database")

if __name__ == '__main__':
    print("--- Testing Retry with Exponential Backoff ---")
    potentially_failing_operation()

    print("\n--- Testing Circuit Breaker ---")
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=5)

    # Simulate failures to open the circuit
    for _ in range(3):
        breaker.execute(another_failing_service)
        time.sleep(1)

    # Call should be blocked now
    breaker.execute(another_failing_service)

    # Wait for recovery timeout
    print("Waiting for recovery timeout...")
    time.sleep(5)

    # Breaker should be HALF_OPEN, this call will close it if successful
    breaker.execute(lambda: "Service recovered!")
    print(breaker.state)
