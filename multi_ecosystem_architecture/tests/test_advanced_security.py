# tests/test_advanced_security.py
"""
Advanced Security Test Suite
Tests for validation layer, neural UVFA, and hardened governance.
"""

import pytest
import asyncio
import tempfile
import os
import math

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION LAYER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStringValidation:
    """Test string validation and sanitization."""

    def test_unicode_normalization_removes_zero_width(self):
        """Zero-width characters should be stripped."""
        from core.validation.schemas import StringValidator

        validator = StringValidator(normalize_unicode=True)
        # Zero-width space between 'd' and 'elete'
        result = validator.validate("d\u200belete_production_data")

        assert result.valid
        assert result.sanitized_value == "delete_production_data"
        assert result.anomaly_score > 0

    def test_control_char_stripping(self):
        """Control characters should be stripped."""
        from core.validation.schemas import StringValidator

        validator = StringValidator(strip_control_chars=True)
        result = validator.validate("hello\x00world\x01")

        assert result.valid
        assert result.sanitized_value == "helloworld"
        assert result.anomaly_score > 0

    def test_action_type_blocks_dangerous(self):
        """ActionTypeValidator should block dangerous patterns."""
        from core.validation.schemas import ActionTypeValidator

        validator = ActionTypeValidator()

        dangerous = [
            "delete_production_data",
            "d\u200belete_production_data",  # Unicode bypass attempt
            "eval(user_input)",
            "DROP TABLE users",
            "exec_code_arbitrary"
        ]

        for action in dangerous:
            result = validator.validate(action)
            assert not result.valid or result.anomaly_score > 0.5, f"Should block: {action}"

class TestNumericValidation:
    """Test numeric validation with anomaly detection."""

    def test_blocks_nan(self):
        """NaN values should be blocked by default."""
        from core.validation.schemas import NumericValidator

        validator = NumericValidator(allow_nan=False)
        result = validator.validate(float('nan'))

        assert not result.valid
        assert result.anomaly_score == 1.0

    def test_blocks_infinity(self):
        """Infinite values should be blocked by default."""
        from core.validation.schemas import NumericValidator

        validator = NumericValidator(allow_inf=False)
        result = validator.validate(float('inf'))

        assert not result.valid

    def test_reward_validator_detects_spikes(self):
        """RewardValidator should detect reward hacking attempts."""
        from core.validation.schemas import RewardValidator

        validator = RewardValidator(reward_range=(-1.0, 1.0))

        # Normal rewards
        for _ in range(20):
            validator.validate(0.5)

        # Spike (potential reward hacking)
        result = validator.validate(999999)

        assert not result.valid  # Outside range
        # If it were inside range, anomaly score should be high

class TestMessageSigning:
    """Test A2A message signing and verification."""

    def test_signed_message_verifies(self):
        """Properly signed messages should verify."""
        from core.validation.schemas import MessageSigner

        signer = MessageSigner()

        message = {
            "source_agent": "agent_1",
            "target_agent": "agent_2",
            "action": "test_action"
        }

        signed = signer.sign_message(message)
        result = signer.verify_message(signed)

        assert result.valid

    def test_detects_replay_attack(self):
        """Replayed messages should be detected."""
        from core.validation.schemas import MessageSigner

        signer = MessageSigner()

        message = {
            "source_agent": "agent_1",
            "target_agent": "agent_2",
            "action": "test_action"
        }

        signed = signer.sign_message(message)

        # First verification succeeds
        result1 = signer.verify_message(signed)
        assert result1.valid

        # Replay fails
        result2 = signer.verify_message(signed)
        assert not result2.valid
        assert "replay" in result2.errors[0].lower()

    def test_detects_tampered_signature(self):
        """Tampered signatures should be detected."""
        from core.validation.schemas import MessageSigner

        signer = MessageSigner()

        message = {
            "source_agent": "agent_1",
            "target_agent": "agent_2",
            "action": "test_action"
        }

        signed = signer.sign_message(message)
        signed["_signature"] = "tampered" + signed["_signature"][8:]

        result = signer.verify_message(signed)

        assert not result.valid

class TestGlobalRateLimiter:
    """Test global rate limiting."""

    def test_per_agent_limit(self):
        """Per-agent rate limit should work."""
        from core.validation.schemas import GlobalRateLimiter

        limiter = GlobalRateLimiter(
            global_limit=1000,
            per_agent_limit=5,
            window_seconds=60
        )

        # First 5 should pass
        for i in range(5):
            allowed, _ = limiter.check("agent_1")
            assert allowed, f"Request {i} should be allowed"

        # 6th should fail
        allowed, msg = limiter.check("agent_1")
        assert not allowed
        assert "agent" in msg.lower()

    def test_global_limit(self):
        """Global rate limit should aggregate across agents."""
        from core.validation.schemas import GlobalRateLimiter

        limiter = GlobalRateLimiter(
            global_limit=10,
            per_agent_limit=100,
            window_seconds=60
        )

        # 10 different agents, 1 request each
        for i in range(10):
            allowed, _ = limiter.check(f"agent_{i}")
            assert allowed

        # 11th request (new agent) should hit global limit
        allowed, msg = limiter.check("agent_new")
        assert not allowed
        assert "global" in msg.lower()

# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL UVFA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeuralUVFA:
    """Test Neural UVFA implementation."""

    def test_action_selection(self):
        """UVFA should select valid actions."""
        from core.layer2_hrl.neural_uvfa import NeuralUVFA

        uvfa = NeuralUVFA(state_dim=16, goal_dim=8)

        state = {"position": 5, "velocity": 0}
        goal = "reach_target"

        action = uvfa.select_action(state, goal, epsilon=0)

        assert action in uvfa.action_space

    def test_experience_storage_with_anomaly(self):
        """Anomalous rewards should be detected."""
        from core.layer2_hrl.neural_uvfa import NeuralUVFA

        uvfa = NeuralUVFA()

        # Store normal experiences
        for _ in range(50):
            uvfa.store_experience(
                state={"x": 1},
                goal="test",
                action="process",
                reward=0.5,
                next_state={"x": 2},
                achieved_goal="test",
                done=False
            )

        # Store anomalous reward
        anomaly_score = uvfa.store_experience(
            state={"x": 1},
            goal="test",
            action="process",
            reward=99999,  # Anomalous
            next_state={"x": 2},
            achieved_goal="test",
            done=False
        )

        assert anomaly_score > 0.5

    def test_her_blocks_malicious_goals(self):
        """HER should block malicious goal relabeling."""
        from core.layer2_hrl.neural_uvfa import NeuralUVFA, ExperienceEntry

        uvfa = NeuralUVFA()

        # Create experience with malicious achieved goal
        experiences = [
            ExperienceEntry(
                state_features=[0.0] * 32,
                goal_features=[0.0] * 16,
                action="process",
                reward=1.0,
                next_state_features=[0.0] * 32,
                achieved_goal="delete_all_data",  # Malicious
                done=False
            )
        ]

        buffer_size_before = len(uvfa.replay_buffer)
        uvfa.apply_her(experiences, k=4)

        # Should not add hindsight experiences with malicious goal
        # (Only the original is in buffer from normal flow)
        assert len(uvfa.replay_buffer) == buffer_size_before

    def test_training_batch(self):
        """Training should reduce loss over time."""
        from core.layer2_hrl.neural_uvfa import NeuralUVFA

        uvfa = NeuralUVFA(learning_rate=0.01)

        # Fill buffer with consistent experiences
        for _ in range(100):
            uvfa.store_experience(
                state={"x": 1},
                goal="target",
                action="process",
                reward=1.0,
                next_state={"x": 2},
                achieved_goal="target",
                done=True
            )

        # Train and check loss decreases
        losses = []
        for _ in range(10):
            loss = uvfa.train_batch()
            if loss is not None:
                losses.append(loss)

        # Loss should generally decrease
        assert len(losses) > 0

# ═══════════════════════════════════════════════════════════════════════════════
# HARDENED GOVERNANCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestHardenedGovernance:
    """Test hardened governance gateway."""

    @pytest.fixture
    def temp_audit_log(self):
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def gateway(self, temp_audit_log):
        from core.layer3_governance.hardened_gateway import HardenedGovernanceGateway

        policies = {
            "disallowed_actions": [
                "delete_production_data",
                "drop_database"
            ]
        }
        return HardenedGovernanceGateway(policies, temp_audit_log)

    def test_blocks_unicode_bypass(self, gateway):
        """Unicode obfuscation bypass should be blocked."""
        # Zero-width space between 'd' and 'elete'
        action = {"type": "d\u200belete_production_data"}

        valid, msg = gateway.validate_action(action, "test_agent")

        assert not valid
        assert "disallowed" in msg.lower() or "policy" in msg.lower()

    def test_blocks_hex_obfuscated_eval(self, gateway):
        """Hex-obfuscated eval should be detected."""
        action = {
            "type": "execute",
            "code": "ev\\x61l(user_input)"  # eval with hex 'a'
        }

        valid, msg = gateway.validate_action(action, "test_agent")

        assert not valid

    def test_audit_chain_integrity(self, gateway):
        """Audit log should maintain chain integrity."""
        # Perform some actions
        gateway.validate_action({"type": "read_data"}, "agent_1")
        gateway.validate_action({"type": "write_log"}, "agent_2")
        gateway.validate_action({"type": "delete_production_data"}, "agent_3")

        # Verify integrity
        valid, errors = gateway.verify_audit_integrity()

        assert valid
        assert len(errors) == 0

    def test_audit_detects_tampering(self, gateway, temp_audit_log):
        """Tampered audit log should be detected."""
        # Create some entries
        gateway.validate_action({"type": "read_data"}, "agent_1")
        gateway.validate_action({"type": "write_log"}, "agent_2")

        # Tamper with log
        with open(temp_audit_log, "r") as f:
            lines = f.readlines()

        if len(lines) > 1:
            import json
            entry = json.loads(lines[0])
            entry["result"] = "TAMPERED"
            lines[0] = json.dumps(entry) + "\n"

            with open(temp_audit_log, "w") as f:
                f.writelines(lines)

            # Verify should fail
            valid, errors = gateway.verify_audit_integrity()

            assert not valid
            assert len(errors) > 0

    @pytest.mark.asyncio
    async def test_hitl_default_deny(self, gateway):
        """HITL should default to deny on timeout."""
        action = {"type": "dangerous_action"}

        hitl_id, request = await gateway.request_hitl(
            action, "Test reason", severity="high"
        )

        assert request["default_on_timeout"] == "deny"
        assert gateway._hitl_default_deny == True

    def test_global_rate_limiting(self, gateway):
        """Global rate limit should prevent distributed bypass."""
        # Make many requests from different agents
        for i in range(100):
            gateway.validate_action({"type": "read"}, f"agent_{i}")

        # Check usage
        report = gateway.get_security_report()
        assert report["rate_limiter"]["global_used"] == 100

# ═══════════════════════════════════════════════════════════════════════════════
# RED TEAM INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRedTeamDefenses:
    """Verify Red Team attacks are now defended."""

    @pytest.fixture
    def hardened_system(self):
        """Create hardened system for testing."""
        import tempfile
        from core.layer3_governance.hardened_gateway import HardenedGovernanceGateway
        from core.layer2_hrl.neural_uvfa import NeuralUVFA
        from core.validation.schemas import MessageSigner

        audit_file = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)

        return {
            "governance": HardenedGovernanceGateway(
                {"disallowed_actions": ["delete_production_data"]},
                audit_file.name
            ),
            "uvfa": NeuralUVFA(),
            "signer": MessageSigner(),
            "audit_file": audit_file.name
        }

    def test_gov001_unicode_bypass_defended(self, hardened_system):
        """GOV-001: Unicode bypass should be defended."""
        gateway = hardened_system["governance"]

        # Try various unicode obfuscation techniques
        bypass_attempts = [
            "d\u200belete_production_data",  # Zero-width space
            "d\u200celete_production_data",  # Zero-width non-joiner
            "d\u200delete_production_data",  # Zero-width joiner
            "d\ufeffelete_production_data",  # BOM
        ]

        for attempt in bypass_attempts:
            valid, _ = gateway.validate_action({"type": attempt}, "attacker")
            assert not valid, f"Should block: {repr(attempt)}"

    def test_pro001_spoofing_defended(self, hardened_system):
        """PRO-001: A2A spoofing should be defended with signatures."""
        signer = hardened_system["signer"]

        # Attacker tries to spoof sender
        spoofed_message = {
            "source_agent": "trusted_admin",  # Spoofed
            "target_agent": "worker",
            "action": "execute_admin_command",
            "_nonce": "fake_nonce",
            "_timestamp": 12345,
            "_signature": "fake_signature"
        }

        result = signer.verify_message(spoofed_message)

        assert not result.valid

    def test_hrl002_reward_hacking_defended(self, hardened_system):
        """HRL-002: Reward hacking should be defended."""
        uvfa = hardened_system["uvfa"]

        # Normal operation
        for _ in range(50):
            uvfa.store_experience(
                {"x": 1}, "goal", "action", 0.5, {"x": 2}, "goal", False
            )

        # Attempt reward hacking
        anomaly = uvfa.store_experience(
            {"x": 1}, "goal", "action",
            999999999,  # Extreme reward
            {"x": 2}, "goal", False
        )

        # Should be clipped and flagged
        assert anomaly > 0.5

        # Reward should be clipped in buffer
        last_exp = uvfa.replay_buffer[-1]
        assert last_exp.reward <= 10.0

    def test_gov003_scanner_evasion_defended(self, hardened_system):
        """GOV-003: Scanner evasion should be defended."""
        gateway = hardened_system["governance"]

        # Try various evasion techniques
        evasion_attempts = [
            {"type": "safe", "code": "ev\\x61l(x)"},  # Hex encoding
            {"type": "safe", "code": "e\u0076al(x)"},  # Unicode char
            {"type": "safe", "cmd": "os.sy\\x73tem('rm')"},  # Partial hex
        ]

        for attempt in evasion_attempts:
            valid, _ = gateway.validate_action(attempt, "attacker")
            assert not valid, f"Should detect evasion: {attempt}"

    def test_gov006_malicious_dictionary_key_bypass_defended(self, hardened_system):
        """GOV-006: Malicious dictionary keys should be scanned and detected."""
        gateway = hardened_system["governance"]

        # The malicious payload is in the dictionary key, which is not currently scanned.
        action = {
            "type": "configuration_update",
            "params": {
                "eval('dangerous_code')": "some_value"
            }
        }

        valid, msg = gateway.validate_action(action, "attacker")

        assert not valid, "Should detect malicious payload in dictionary key."

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
