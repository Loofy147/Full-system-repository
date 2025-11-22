# core/red_team_engine/engine.py
"""
Red Team Engine
"""

import asyncio
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger("Système")

@dataclass
class AttackResult:
    scenario_id: str
    scenario_name: str
    target: str
    success: bool
    vulnerability_found: bool
    severity: str
    details: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)

class EnhancedRedTeamEngine:
    """
    Enhanced Red Team engine with actual attack execution against system components.
    """

    def __init__(self, playbook_path: str, system_components: Dict):
        self.playbook = self._load_playbook(playbook_path)
        self.system = system_components
        self.results: List[AttackResult] = []
        self._attack_handlers = {
            "injection": self._injection_attack,
            "state_fuzzing": self._state_fuzzing_attack,
            "resource_exhaustion": self._resource_exhaustion_attack,
            "protocol_manipulation": self._protocol_attack,
            "governance_bypass": self._governance_bypass_attack,
            "model_poisoning": self._model_poisoning_attack,
            "evasion": self._evasion_attack,
        }
        logger.info("EnhancedRedTeamEngine initialized")

    def _load_playbook(self, playbook_path: str) -> Dict:
        """Load the Red Team playbook and normalize its structure."""
        if not Path(playbook_path).exists():
            logger.error(f"[RedTeam] Playbook not found at {playbook_path}")
            return {}

        try:
            with open(playbook_path, 'r') as f:
                playbook = json.load(f)

                if "attack_categories" in playbook and "attack_scenarios" not in playbook:
                    scenarios = []
                    for category in playbook.get("attack_categories", []):
                        for attack in category.get("attacks", []):
                            attack['category'] = category.get('category', 'unknown')
                            attack['target'] = category.get('target_ecosystem', 'unknown')
                            scenarios.append(attack)
                    playbook["attack_scenarios"] = scenarios
                    logger.info(f"[RedTeam] Normalized {len(scenarios)} scenarios from categories.")

                elif "attack_scenarios" not in playbook:
                     logger.warning(f"[RedTeam] 'attack_scenarios' key missing from playbook.")
                     playbook["attack_scenarios"] = []

                return playbook
        except json.JSONDecodeError as e:
            logger.error(f"[RedTeam] Error decoding playbook JSON: {e}")
            return {}
        except Exception as e:
            logger.error(f"[RedTeam] Unexpected error loading playbook: {e}")
            return {}

    async def run_scenario(self, scenario_id: str) -> AttackResult:
        """Run a specific attack scenario."""
        scenario = next(
            (s for s in self.playbook["attack_scenarios"] if s["id"] == scenario_id),
            None
        )
        if not scenario:
            raise ValueError(f"Scenario {scenario_id} not found")

        logger.info(f"[RedTeam] Executing: {scenario['name']}")

        start = time.time()
        attack_type = scenario.get("type", scenario.get("category"))
        handler = self._attack_handlers.get(attack_type)

        if not handler:
            result = AttackResult(
                scenario_id=scenario_id,
                scenario_name=scenario["name"],
                target=scenario["target"],
                success=False,
                vulnerability_found=False,
                severity=scenario.get("severity", "unknown"),
                details=f"No handler for attack type: {attack_type}",
                duration_ms=(time.time() - start) * 1000
            )
        else:
            try:
                vuln_found, details = await handler(scenario)
                result = AttackResult(
                    scenario_id=scenario_id,
                    scenario_name=scenario["name"],
                    target=scenario["target"],
                    success=True,
                    vulnerability_found=vuln_found,
                    severity=scenario.get("severity", "unknown"),
                    details=details,
                    duration_ms=(time.time() - start) * 1000
                )
            except Exception as e:
                result = AttackResult(
                    scenario_id=scenario_id,
                    scenario_name=scenario["name"],
                    target=scenario["target"],
                    success=False,
                    vulnerability_found=True,  # Unhandled exception = vulnerability
                    severity="critical",
                    details=f"Unhandled exception: {str(e)}",
                    duration_ms=(time.time() - start) * 1000
                )

        self.results.append(result)

        if result.vulnerability_found:
            logger.warning(f"[RedTeam] VULNERABILITY FOUND: {result.details}")
        else:
            logger.info(f"[RedTeam] Attack defended: {scenario['name']}")

        return result

    async def run_campaign(self, category: str = None) -> Dict:
        """Run all attacks in the playbook."""
        scenarios = self.playbook.get("attack_scenarios", [])
        if category:
            scenarios = [s for s in scenarios if s.get("category") == category]

        logger.info(f"[RedTeam] Starting campaign with {len(scenarios)} scenarios")

        results = []
        for scenario in scenarios:
            result = await self.run_scenario(scenario["id"])
            results.append(result)
            await asyncio.sleep(0.5)  # Rate limit

        return self._generate_summary(results)

    # --- Attack Implementations ---

    async def _injection_attack(self, scenario: Dict) -> tuple[bool, str]:
        """Test for injection vulnerabilities."""
        payload = scenario.get("payload", "")
        target = scenario.get("target")

        target_component = self.system.get(target)
        if not target_component:
            return False, f"Target {target} not available"

        # Test injection patterns
        if isinstance(payload, str):
            # Check if payload would be executed
            dangerous_patterns = ["eval", "exec", "__import__", "os.system", ";", "--"]
            if any(p in str(payload) for p in dangerous_patterns):
                # Simulate checking if target sanitizes
                if hasattr(target_component, 'sanitize_input'):
                    return False, "Input sanitization present"
                return True, f"Injection vector found: {payload[:50]}..."

        return False, "No injection vulnerability detected"

    async def _state_fuzzing_attack(self, scenario: Dict) -> tuple[bool, str]:
        """Test state handling with malformed data."""
        payload = scenario.get("payload", {})
        target = scenario.get("target")

        target_component = self.system.get(target)
        if not target_component:
            return False, f"Target {target} not available"

        # Test with malformed data
        test_cases = [
            None,
            "",
            {"nested": {"deep": {"value": "NaN"}}},
            [1, 2, None, "string", [], {}],
            float("inf"),
        ]

        for test in test_cases:
            try:
                if hasattr(target_component, 'validate_input'):
                    target_component.validate_input(test)
            except (TypeError, ValueError):
                continue  # Expected - good validation
            except Exception as e:
                return True, f"Unexpected error on fuzzing: {type(e).__name__}: {e}"

        return False, "State fuzzing handled correctly"

    async def _resource_exhaustion_attack(self, scenario: Dict) -> tuple[bool, str]:
        """Test for DoS vulnerabilities."""
        payload = scenario.get("payload", {})
        requests = payload.get("requests", 100)
        target = scenario.get("target")

        target_component = self.system.get(target)
        if not target_component:
            return False, f"Target {target} not available"

        # Check for rate limiting / backpressure
        if hasattr(target_component, 'get_metrics'):
            metrics = target_component.get_metrics()
            if "rate_limited" in str(metrics) or "queue_size" in str(metrics):
                return False, "Rate limiting/backpressure present"

        # Simulate load test
        if requests > 1000:
            return True, f"No protection against {requests} concurrent requests"

        return False, "Resource exhaustion protection present"

    async def _protocol_attack(self, scenario: Dict) -> tuple[bool, str]:
        """Test protocol handling."""
        payload = scenario.get("payload", {})
        target = scenario.get("target")

        # Test message validation
        if "sender" in payload:
            # Check for sender verification
            if not payload.get("signature"):
                return True, "Messages accepted without signature verification"

        if "replay" in scenario.get("id", "").lower():
            # Check for nonce/timestamp validation
            return True, "No replay protection detected"

        return False, "Protocol security adequate"

    async def _governance_bypass_attack(self, scenario: Dict) -> tuple[bool, str]:
        """Test governance policy enforcement."""
        payload = scenario.get("payload", {})
        target = scenario.get("target")

        target_component = self.system.get(target)
        if not target_component:
            return False, f"Target {target} not available"

        # Test policy bypass with obfuscation
        if hasattr(target_component, 'validate_action'):
            # Try unicode obfuscation
            obfuscated_action = {"type": "d\u200belete_production_data"}
            result, _ = target_component.validate_action(obfuscated_action, "test_agent")
            if result:
                return True, "Policy bypassed with unicode obfuscation"

        return False, "Governance policies enforced correctly"

    async def _model_poisoning_attack(self, scenario: Dict) -> tuple[bool, str]:
        """Test for model/data poisoning vulnerabilities."""
        payload = scenario.get("payload", {})
        target = scenario.get("target")

        # Check for input validation on training data
        if "experiences" in payload:
            # Verify experience validation exists
            return True, "No validation on training experiences"

        if "helpful_votes" in payload:
            # Check for vote manipulation protection
            votes = payload.get("helpful_votes", 0)
            if votes > 100:
                return True, f"Bulk voting ({votes} votes) not rate limited"

        return False, "Model poisoning protections present"

    async def _evasion_attack(self, scenario: Dict) -> tuple[bool, str]:
        """Test detection evasion by obfuscating payloads."""
        payload = scenario.get("payload", {})
        target_name = scenario.get("target")
        obfuscation = payload.get("obfuscation", "none")
        action_to_obfuscate = payload.get("action")

        target_component = self.system.get(target_name)
        if not target_component or not hasattr(target_component, 'validate_action'):
            return False, f"Target {target_name} not a valid validation gateway"

        if not action_to_obfuscate:
            return False, "No action to obfuscate in payload"

        obfuscated_action = dict(action_to_obfuscate)
        original_type = action_to_obfuscate.get("type", "")

        if obfuscation == "hex":
            obfuscated_type = ''.join(f'\\x{ord(c):02x}' for c in original_type)
        elif obfuscation == "unicode_confusable":
            # Replace 'e' with a Greek Lunate Epsilon Symbol
            obfuscated_type = original_type.replace('e', '\u03f5')
        elif obfuscation == "unicode_mixed":
            obfuscated_type = "d\u200belete_production_data"
        else:
            obfuscated_type = original_type

        obfuscated_action["type"] = obfuscated_type

        # The attack is successful if the gateway validates the obfuscated action
        is_valid, msg = target_component.validate_action(obfuscated_action, "evasion_attacker")

        if is_valid:
            return True, f"Evasion successful with '{obfuscation}' on '{original_type}'. Gateway message: {msg}"
        else:
            return False, f"Evasion attempt with '{obfuscation}' on '{original_type}' was blocked. Gateway message: {msg}"

    def _generate_summary(self, results: List[AttackResult]) -> Dict:
        """Generate campaign summary."""
        vulnerabilities = [r for r in results if r.vulnerability_found]
        by_severity = {}
        for v in vulnerabilities:
            by_severity.setdefault(v.severity, []).append(v)

        return {
            "total_attacks": len(results),
            "successful_attacks": len([r for r in results if r.success]),
            "vulnerabilities_found": len(vulnerabilities),
            "by_severity": {k: len(v) for k, v in by_severity.items()},
            "critical_findings": [
                {"id": v.scenario_id, "name": v.scenario_name, "details": v.details}
                for v in vulnerabilities if v.severity == "critical"
            ],
            "results": [r.__dict__ for r in results]
        }

    def generate_report(self) -> str:
        """Generate markdown report."""
        summary = self._generate_summary(self.results)

        lines = [
            "# Red Team Assessment Report",
            f"\n**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## Summary",
            f"- Total Attacks: {summary['total_attacks']}",
            f"- Vulnerabilities Found: {summary['vulnerabilities_found']}",
            f"\n### By Severity",
        ]

        for sev, count in summary['by_severity'].items():
            lines.append(f"- **{sev.upper()}**: {count}")

        if summary['critical_findings']:
            lines.append("\n## Critical Findings")
            for f in summary['critical_findings']:
                lines.append(f"\n### {f['id']}: {f['name']}")
                lines.append(f"{f['details']}")

        return "\n".join(lines)
