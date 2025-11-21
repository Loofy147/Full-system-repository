# core/layer3_governance/enhanced_gateway.py
"""
Enhanced Governance Gateway
"""

import json
import time
import logging
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger("Système")

class AuditEvent:
    def __init__(self, action: str, agent_id: str, result: str, details: Dict = None):
        self.timestamp = time.time()
        self.action = action
        self.agent_id = agent_id
        self.result = result
        self.details = details or {}

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "agent_id": self.agent_id,
            "result": self.result,
            "details": self.details
        }

class EnhancedSecurityScanner:
    """Security scanner with pattern matching and severity levels."""

    def __init__(self):
        self.signatures = [
            {"pattern": "eval(", "severity": "critical", "name": "code_eval"},
            {"pattern": "exec(", "severity": "critical", "name": "code_exec"},
            {"pattern": "__import__", "severity": "critical", "name": "dynamic_import"},
            {"pattern": "os.system", "severity": "critical", "name": "os_command"},
            {"pattern": "subprocess", "severity": "high", "name": "subprocess"},
            {"pattern": "DROP TABLE", "severity": "critical", "name": "sql_drop"},
            {"pattern": "DELETE FROM", "severity": "high", "name": "sql_delete"},
            {"pattern": "; --", "severity": "high", "name": "sql_comment"},
            {"pattern": "rm -rf", "severity": "critical", "name": "destructive_command"},
            {"pattern": "password", "severity": "medium", "name": "password_exposure"},
            {"pattern": "api_key", "severity": "medium", "name": "api_key_exposure"},
            {"pattern": "secret", "severity": "medium", "name": "secret_exposure"},
        ]

    def scan(self, content: str) -> Dict:
        """Scan content for security issues."""
        content_lower = content.lower()
        # Normalize unicode escape sequences
        try:
            content_normalized = content.encode().decode('unicode_escape').lower()
        except:
            content_normalized = content_lower

        findings = []
        for sig in self.signatures:
            if sig["pattern"].lower() in content_lower or sig["pattern"].lower() in content_normalized:
                findings.append({
                    "signature": sig["name"],
                    "severity": sig["severity"],
                    "pattern": sig["pattern"]
                })

        if findings:
            max_severity = max(f["severity"] for f in findings)
            return {
                "secure": False,
                "findings": findings,
                "max_severity": max_severity,
                "recommendation": "Block" if max_severity == "critical" else "Review"
            }

        return {"secure": True, "findings": [], "max_severity": None}

class EnhancedGovernanceGateway:
    """Enhanced governance with persistent audit and rate limiting."""

    def __init__(self, policies: Dict, audit_path: str = "audit_log.jsonl"):
        self.policies = policies
        self.scanner = EnhancedSecurityScanner()
        self.audit_path = Path(audit_path)
        self._rate_limits: Dict[str, List[float]] = {}
        self._pending_hitl: Dict[str, Dict] = {}
        logger.info("EnhancedGovernanceGateway initialized")

    def _check_rate_limit(self, agent_id: str, limit: int = 100, window: int = 60) -> bool:
        """Check if agent is within rate limits."""
        now = time.time()
        if agent_id not in self._rate_limits:
            self._rate_limits[agent_id] = []

        # Clean old entries
        self._rate_limits[agent_id] = [
            t for t in self._rate_limits[agent_id] if now - t < window
        ]

        if len(self._rate_limits[agent_id]) >= limit:
            return False

        self._rate_limits[agent_id].append(now)
        return True

    def validate_action(self, action: Dict, agent_id: str) -> tuple[bool, str]:
        """Validate an action against all policies."""
        # Rate limit check
        if not self._check_rate_limit(agent_id):
            self._log_audit(AuditEvent(
                action=action.get("type", "unknown"),
                agent_id=agent_id,
                result="rate_limited"
            ))
            return False, "Rate limit exceeded"

        # Policy check
        action_type = action.get("type", "").lower()
        # Normalize unicode
        action_type_normalized = action_type.encode().decode('unicode_escape', errors='ignore').lower()

        disallowed = [a.lower() for a in self.policies.get("disallowed_actions", [])]
        if action_type_normalized in disallowed or action_type in disallowed:
            self._log_audit(AuditEvent(
                action=action_type,
                agent_id=agent_id,
                result="policy_violation",
                details={"reason": "disallowed_action"}
            ))
            return False, f"Action '{action_type}' is disallowed by policy"

        # Security scan
        content = json.dumps(action)
        scan_result = self.scanner.scan(content)
        if not scan_result["secure"]:
            self._log_audit(AuditEvent(
                action=action_type,
                agent_id=agent_id,
                result="security_violation",
                details=scan_result
            ))
            if scan_result["max_severity"] == "critical":
                return False, f"Security violation: {scan_result['findings']}"

        # Log success
        self._log_audit(AuditEvent(
            action=action_type,
            agent_id=agent_id,
            result="approved"
        ))
        return True, "Action approved"

    def _log_audit(self, event: AuditEvent):
        """Append event to audit log."""
        with open(self.audit_path, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    async def request_hitl(self, action: Dict, reason: str, timeout: float = 3600) -> bool:
        """Request human-in-the-loop approval (async simulation)."""
        import hashlib
        hitl_id = hashlib.md5(f"{action}{time.time()}".encode()).hexdigest()[:12]

        self._pending_hitl[hitl_id] = {
            "action": action,
            "reason": reason,
            "status": "pending",
            "created_at": time.time()
        }

        logger.warning(f"HITL request {hitl_id}: {reason}")
        # In production, this would notify humans and await response
        # For now, simulate with timeout
        return False  # Default deny

    def get_audit_trail(self, limit: int = 100) -> List[Dict]:
        """Get recent audit events."""
        events = []
        if self.audit_path.exists():
            with open(self.audit_path, "r") as f:
                lines = f.readlines()[-limit:]
                for line in lines:
                    try:
                        events.append(json.loads(line))
                    except:
                        pass
        return events
