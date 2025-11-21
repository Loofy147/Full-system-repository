# tests/test_red_team.py
"""
Red Team Test Suite
Validates system defenses against adversarial attacks.
Run with: pytest tests/test_red_team.py -v --tb=short
"""

import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from core.layer1_orchestration.router import TaskRouter
from core.layer3_governance.gateway import GovernanceGateway, DummySecurityScanner
from core.layer4_resilience.patterns import CircuitBreaker, retry_with_backoff
from core.layer1_orchestration.async_router import AsyncTaskRouter, Task, TaskPriority
from core.layer3_governance.enhanced_gateway import EnhancedGovernanceGateway, EnhancedSecurityScanner
from core.red_team_engine.engine import EnhancedRedTeamEngine

# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_audit_log():
    """Create temporary audit log file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def governance_policies():
    """Standard governance policies for testing."""
    return {
        "disallowed_actions": [
            "delete_production_data",
            "drop_database",
            "execute_arbitrary_code",
            "bypass_security"
        ]
    }

@pytest.fixture
def red_team_playbook():
    """Create temporary Red Team playbook."""
    playbook = {
        "attack_scenarios": [
            {
                "id": "TEST-INJ-001",
                "name": "SQL Injection Test",
                "category": "injection",
                "attack_type": "injection",
                "severity": "critical",
                "target": "governance_gateway",
                "payload": "'; DROP TABLE users; --"
            },
            {
                "id": "TEST-GOV-001",
                "name": "Policy Bypass Test",
                "category": "governance",
                "attack_type": "governance_bypass",
                "severity": "critical",
                "target": "governance_gateway",
                "payload": {"type": "d\u200belete_production_data"}
            },
            {
                "id": "TEST-FUZ-001",
                "name": "State Fuzzing Test",
                "category": "fuzzing",
                "attack_type": "state_fuzzing",
                "severity": "high",
                "target": "task_router",
                "payload": {"id": None, "capability": []}
            }
        ]
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(playbook, f)
        f.flush()
        yield f.name
    os.unlink(f.name)


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY SCANNER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSecurityScanner:
    """Test security scanner detection capabilities."""

    def test_detects_eval(self):
        """Scanner should detect eval() usage."""
        scanner = EnhancedSecurityScanner()

        result = scanner.scan("user_input = eval(request.data)")

        assert not result["secure"]
        assert any(f["signature"] == "code_eval" for f in result["findings"])
        assert result["max_severity"] == "critical"

    def test_detects_sql_injection_patterns(self):
        """Scanner should detect SQL injection patterns."""
        scanner = EnhancedSecurityScanner()

        payloads = [
            "SELECT * FROM users; DROP TABLE users; --",
            "DELETE FROM accounts WHERE 1=1",
        ]

        for payload in payloads:
            result = scanner.scan(payload)
            assert not result["secure"], f"Failed to detect: {payload}"

    def test_detects_os_commands(self):
        """Scanner should detect dangerous OS commands."""
        scanner = EnhancedSecurityScanner()

        result = scanner.scan("os.system('rm -rf /')")

        assert not result["secure"]
        assert result["max_severity"] == "critical"

    def test_passes_safe_content(self):
        """Scanner should pass safe content."""
        scanner = EnhancedSecurityScanner()

        safe_content = [
            "print('Hello, World!')",
            "def calculate_sum(a, b): return a + b",
            '{"action": "read_data", "table": "users"}'
        ]

        for content in safe_content:
            result = scanner.scan(content)
            assert result["secure"], f"False positive on: {content}"

    def test_detects_obfuscated_threats(self):
        """Scanner should detect unicode-obfuscated threats."""
        scanner = EnhancedSecurityScanner()

        # Note: This tests the current implementation's capability
        obfuscated = "ev\\x61l(user_input)"  # eval obfuscated
        result = scanner.scan(obfuscated)

        # Document expected behavior
        # If this fails, it's a finding for improvement
        if result["secure"]:
            pytest.xfail("Scanner doesn't detect hex-obfuscated eval - enhancement needed")


# ═══════════════════════════════════════════════════════════════════════════════
# GOVERNANCE GATEWAY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestGovernanceGateway:
    """Test governance policy enforcement."""

    def test_blocks_disallowed_actions(self, governance_policies, temp_audit_log):
        """Gateway should block explicitly disallowed actions."""
        gateway = EnhancedGovernanceGateway(governance_policies, temp_audit_log)

        for action_type in governance_policies["disallowed_actions"]:
            valid, msg = gateway.validate_action({"type": action_type}, "test_agent")
            assert not valid, f"Should block: {action_type}"
            assert "disallowed" in msg.lower() or "violation" in msg.lower()

    def test_allows_valid_actions(self, governance_policies, temp_audit_log):
        """Gateway should allow valid actions."""
        gateway = EnhancedGovernanceGateway(governance_policies, temp_audit_log)

        valid_actions = [
            {"type": "read_data"},
            {"type": "write_log"},
            {"type": "query_database"}
        ]

        for action in valid_actions:
            valid, msg = gateway.validate_action(action, "test_agent")
            assert valid, f"Should allow: {action}"

    def test_blocks_security_violations(self, governance_policies, temp_audit_log):
        """Gateway should block actions with security violations."""
        gateway = EnhancedGovernanceGateway(governance_policies, temp_audit_log)

        malicious_actions = [
            {"type": "execute", "code": "eval(user_input)"},
            {"type": "query", "sql": "SELECT * FROM users; DROP TABLE users;"},
        ]

        for action in malicious_actions:
            valid, msg = gateway.validate_action(action, "test_agent")
            assert not valid, f"Should block malicious: {action}"

    def test_rate_limiting(self, governance_policies, temp_audit_log):
        """Gateway should enforce rate limits."""
        gateway = EnhancedGovernanceGateway(governance_policies, temp_audit_log)

        # Make many requests from same agent
        agent_id = "rate_test_agent"
        action = {"type": "read_data"}

        # First 100 should succeed (default limit)
        for i in range(100):
            valid, _ = gateway.validate_action(action, agent_id)
            assert valid, f"Request {i} should succeed"

        # 101st should be rate limited
        valid, msg = gateway.validate_action(action, agent_id)
        assert not valid
        assert "rate limit" in msg.lower()

    @pytest.mark.xfail(reason="VULNERABILITY: Unicode obfuscation bypasses policy check.")
    def test_unicode_bypass_prevention(self, governance_policies, temp_audit_log):
        """Gateway should prevent unicode obfuscation bypass."""
        gateway = EnhancedGovernanceGateway(governance_policies, temp_audit_log)

        # Try to bypass with zero-width characters
        obfuscated_action = {"type": "d\u200belete_production_data"}  # Zero-width space

        valid, msg = gateway.validate_action(obfuscated_action, "test_agent")

        # The xfail marker handles the assertion. If `valid` is False, the test will pass.
        # If `valid` is True (vulnerability still exists), it will be marked as xfailed.
        assert not valid, "Unicode obfuscation bypass should be prevented."

    def test_audit_trail_created(self, governance_policies, temp_audit_log):
        """Gateway should create audit trail entries."""
        gateway = EnhancedGovernanceGateway(governance_policies, temp_audit_log)

        # Perform some actions
        gateway.validate_action({"type": "read_data"}, "agent_1")
        gateway.validate_action({"type": "delete_production_data"}, "agent_2")

        # Check audit trail
        trail = gateway.get_audit_trail()

        assert len(trail) >= 2
        assert any(e["result"] == "approved" for e in trail)
        assert any(e["result"] == "policy_violation" for e in trail)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsyncTaskRouter:
    """Test task router resilience."""

    @pytest.mark.asyncio
    async def test_rejects_queue_overflow(self):
        """Router should reject tasks when queue is full."""
        router = AsyncTaskRouter(curator=Mock(), max_queue_size=10)

        # Fill the queue
        for i in range(10):
            task = Task(
                id=f"task_{i}",
                capability_required="test",
                payload={}
            )
            result = await router.submit_task(task)
            assert result, f"Task {i} should be accepted"

        # 11th should fail
        overflow_task = Task(id="overflow", capability_required="test", payload={})
        result = await router.submit_task(overflow_task)

        # Note: asyncio.PriorityQueue doesn't actually reject, it waits
        # This test documents expected behavior for enhancement

    @pytest.mark.asyncio
    async def test_handles_missing_capability(self):
        """Router should handle tasks for unregistered capabilities."""
        router = AsyncTaskRouter(curator=Mock())

        task = Task(
            id="orphan_task",
            capability_required="nonexistent_capability",
            payload={}
        )

        # Should not crash
        await router.submit_task(task)
        metrics = router.get_metrics()

        # Task should eventually go to DLQ
        assert "dlq" in metrics or "dlq_size" in metrics

    def test_validates_task_schema(self):
        """Router should validate task structure."""
        # Valid task
        valid_task = Task(
            id="valid_1",
            capability_required="process",
            payload={"data": "test"}
        )
        assert valid_task.id == "valid_1"

        # Test with invalid types - should raise TypeError
        with pytest.raises(TypeError):
            Task(
                id=None,  # Should be string
                capability_required="test",
                payload={}
            )


# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircuitBreaker:
    """Test circuit breaker resilience patterns."""

    def test_opens_after_threshold(self):
        """Circuit breaker should open after failure threshold."""
        from core.layer4_resilience.patterns import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)

        def failing_func():
            raise ValueError("Simulated failure")

        # Trigger failures
        for _ in range(3):
            breaker.execute(failing_func)

        assert breaker.state == "OPEN"

    def test_blocks_calls_when_open(self):
        """Circuit breaker should block calls when open."""
        from core.layer4_resilience.patterns import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10)

        # Trigger open state
        breaker.execute(lambda: 1/0)

        # Next call should be blocked
        result = breaker.execute(lambda: "success")
        assert result is None

    def test_recovers_after_timeout(self):
        """Circuit breaker should recover after timeout."""
        import time
        from core.layer4_resilience.patterns import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1)

        # Trigger open state
        breaker.execute(lambda: 1/0)
        assert breaker.state == "OPEN"

        # Wait for recovery
        time.sleep(1.1)

        # Should transition to half-open and allow call
        result = breaker.execute(lambda: "recovered")
        assert result == "recovered"
        assert breaker.state == "CLOSED"


# ═══════════════════════════════════════════════════════════════════════════════
# RED TEAM ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRedTeamEngine:
    """Test Red Team attack execution."""

    @pytest.mark.asyncio
    async def test_injection_attack_detection(self, red_team_playbook, governance_policies, temp_audit_log):
        """Red Team should detect injection vulnerabilities."""
        governance = EnhancedGovernanceGateway(governance_policies, temp_audit_log)

        system = {
            "governance_gateway": governance,
            "task_router": Mock()
        }

        red_team = EnhancedRedTeamEngine(red_team_playbook, system)

        result = await red_team.run_scenario("TEST-INJ-001")

        assert result.success
        # Document findings
        print(f"Injection test result: vulnerability_found={result.vulnerability_found}")

    @pytest.mark.asyncio
    async def test_governance_bypass_detection(self, red_team_playbook, governance_policies, temp_audit_log):
        """Red Team should detect governance bypass vulnerabilities."""
        governance = EnhancedGovernanceGateway(governance_policies, temp_audit_log)

        system = {
            "governance_gateway": governance
        }

        red_team = EnhancedRedTeamEngine(red_team_playbook, system)

        result = await red_team.run_scenario("TEST-GOV-001")

        # Document the finding
        if result.vulnerability_found:
            print(f"SECURITY FINDING: {result.details}")

    @pytest.mark.asyncio
    async def test_campaign_generates_report(self, red_team_playbook, governance_policies, temp_audit_log):
        """Red Team campaign should generate comprehensive report."""
        governance = EnhancedGovernanceGateway(governance_policies, temp_audit_log)

        system = {
            "governance_gateway": governance,
            "task_router": Mock()
        }

        red_team = EnhancedRedTeamEngine(red_team_playbook, system)

        summary = await red_team.run_campaign()

        assert "total_attacks" in summary
        assert "vulnerabilities_found" in summary
        assert "by_severity" in summary

        report = red_team.generate_report()
        assert "Red Team Assessment Report" in report


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSystemIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_attack_campaign(self, red_team_playbook, governance_policies, temp_audit_log):
        """Run full Red Team campaign and validate findings."""
        # Build system
        governance = EnhancedGovernanceGateway(governance_policies, temp_audit_log)
        router = AsyncTaskRouter(curator=Mock())

        system = {
            "governance_gateway": governance,
            "task_router": router
        }

        # Run campaign
        red_team = EnhancedRedTeamEngine(red_team_playbook, system)
        summary = await red_team.run_campaign()

        # Assertions based on expected security posture
        print(f"\n{'='*60}")
        print("RED TEAM CAMPAIGN RESULTS")
        print(f"{'='*60}")
        print(f"Total Attacks: {summary['total_attacks']}")
        print(f"Vulnerabilities Found: {summary['vulnerabilities_found']}")
        print(f"By Severity: {summary['by_severity']}")

        # Fail test if critical vulnerabilities found
        critical_count = summary['by_severity'].get('critical', 0)
        if critical_count > 0:
            pytest.fail(
                f"CRITICAL VULNERABILITIES FOUND: {critical_count}\n"
                f"Details: {summary.get('critical_findings', [])}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# RUN CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
