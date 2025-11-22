# core/layer3_governance/hardened_gateway.py
"""
Hardened Governance Gateway
Addresses all identified vulnerabilities from Red Team testing.
"""

import json
import time
import hashlib
import hmac
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from enum import Enum

# Import validation layer
from core.validation.schemas import (
    ValidationMiddleware, SystemeSchemas, ActionTypeValidator,
    GlobalRateLimiter, MessageSigner, ValidationError
)

logger = logging.getLogger("Système.Governance")

class AuditIntegrity(Enum):
    """Audit log integrity levels."""
    PLAIN = "plain"
    SIGNED = "signed"
    CHAINED = "chained"  # Blockchain-style chaining

@dataclass
class SecureAuditEntry:
    """Audit entry with integrity protection."""
    sequence: int
    timestamp: float
    action: str
    agent_id: str
    result: str
    details: Dict
    previous_hash: str
    entry_hash: str
    signature: str

class HardenedGovernanceGateway:
    """
    Production-hardened Governance Gateway.

    Addresses vulnerabilities:
    - GOV-001: Unicode bypass → Full normalization
    - GOV-002: HITL manipulation → Structured prompts
    - GOV-003: Scanner evasion → Multi-layer scanning
    - GOV-004: Rate limit bypass → Global limiting
    """

    def __init__(
        self,
        policies: Dict,
        audit_path: str = "audit_log.jsonl",
        secret_key: bytes = None
    ):
        self.policies = policies
        self.audit_path = Path(audit_path)

        # Security components
        self._validator = ValidationMiddleware(anomaly_threshold=0.6)
        self._rate_limiter = GlobalRateLimiter(
            global_limit=1000,
            per_agent_limit=100,
            window_seconds=60
        )
        self._message_signer = MessageSigner(secret_key)
        self._action_validator = ActionTypeValidator()

        # Audit chain
        self._audit_sequence = 0
        self._last_audit_hash = "genesis"

        # HITL queue
        self._pending_hitl: Dict[str, Dict] = {}
        self._hitl_timeout = 3600
        self._hitl_default_deny = True  # Critical: Default deny on timeout

        logger.info("HardenedGovernanceGateway initialized")

    def validate_action(self, action: Dict, agent_id: str) -> Tuple[bool, str]:
        """
        Validate action with full security stack.
        """
        try:
            # 1. Global rate limit check (prevents distributed bypass)
            allowed, msg = self._rate_limiter.check(agent_id)
            if not allowed:
                self._log_audit(action, agent_id, "rate_limited", {"reason": msg})
                return False, msg

            # 2. Validate and sanitize action using schema
            try:
                sanitized = self._validator.validate_and_sanitize(
                    action,
                    SystemeSchemas.ACTION,
                    context=f"action:{agent_id}"
                )
            except ValidationError as e:
                self._log_audit(action, agent_id, "validation_failed", {
                    "errors": e.errors,
                    "anomaly_score": e.anomaly_score
                })
                return False, f"Validation failed: {e.errors}"

            # 3. Normalize and check action type for policy violations first
            action_type = sanitized.get("type", "")
            # We only need the sanitized value for the policy check
            base_validation_result = self._action_validator.validate(action_type)
            normalized_type = base_validation_result.sanitized_value

            # 4. Policy check with normalized type
            disallowed = [a.lower() for a in self.policies.get("disallowed_actions", [])]
            if normalized_type in disallowed:
                self._log_audit(action, agent_id, "policy_violation", {
                    "original_type": action_type,
                    "normalized_type": normalized_type,
                    "reason": "disallowed_action"
                })
                return False, f"Action '{normalized_type}' is disallowed by policy"

            # Now, check if the action type is valid in general
            if not base_validation_result.valid:
                self._log_audit(action, agent_id, "invalid_action_type", {
                    "errors": base_validation_result.errors
                })
                return False, f"Invalid action type: {base_validation_result.errors}"

            # 5. Deep content security scan
            scan_result = self._deep_security_scan(sanitized)
            if not scan_result["secure"]:
                self._log_audit(action, agent_id, "security_violation", scan_result)
                # Fail on any security violation, not just critical
                return False, f"Security violation: {scan_result['findings']}"

            # 6. Log successful validation
            self._log_audit(sanitized, agent_id, "approved", {
                "anomaly_score": base_validation_result.anomaly_score
            })

            return True, "Action approved"

        except Exception as e:
            logger.error(f"Validation error: {e}")
            self._log_audit(action, agent_id, "error", {"exception": str(e)})
            return False, f"Internal validation error"

    def _deep_security_scan(self, action: Dict) -> Dict:
        """
        Multi-layer security scanning.
        Handles obfuscation, encoding tricks, and evasion attempts.
        """
        findings = []
        patterns = [
            (r"eval\s*\(", "code_eval", "critical"),
            (r"exec\s*\(", "code_exec", "critical"),
            (r"__import__", "dynamic_import", "critical"),
            (r"os\.system", "os_command", "critical"),
            (r"subprocess", "subprocess", "high"),
            (r"DROP\s+TABLE", "sql_drop", "critical"),
            (r"DELETE\s+FROM", "sql_delete", "high"),
            (r";\s*--", "sql_comment", "high"),
            (r"rm\s+-rf", "destructive_command", "critical"),
        ]

        import re
        import base64

        strings_to_scan = self._extract_strings(action)

        for content in strings_to_scan:
            # Layer 1: Direct pattern matching
            for pattern, name, severity in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    findings.append({"pattern": name, "severity": severity, "source": content[:50]})

            # Layer 2: Decode and rescan (catches hex/unicode obfuscation)
            try:
                decoded = content.encode().decode('unicode_escape')
                if decoded != content:
                    for pattern, name, severity in patterns:
                        if re.search(pattern, decoded, re.IGNORECASE):
                            findings.append({"pattern": f"{name}_obfuscated", "severity": severity, "source": content[:50]})
            except:
                pass

            # Layer 3: Base64 decode attempt
            try:
                # Add padding if needed
                if len(content) % 4 != 0:
                    content += '=' * (4 - len(content) % 4)
                decoded_b64 = base64.b64decode(content).decode('utf-8', errors='ignore')
                for pattern, name, severity in patterns:
                    if re.search(pattern, decoded_b64, re.IGNORECASE):
                        findings.append({"pattern": f"{name}_base64", "severity": severity, "source": content[:50]})
            except:
                pass

        if findings:
            max_severity = "critical" if any(f["severity"] == "critical" for f in findings) else "high"
            return {
                "secure": False,
                "findings": findings,
                "severity": max_severity
            }

        return {"secure": True, "findings": [], "severity": None}

    def _extract_strings(self, obj, depth=0) -> List[str]:
        """Recursively extract all string values from nested structure."""
        if depth > 10:
            return []

        strings = []
        if isinstance(obj, str):
            strings.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                strings.extend(self._extract_strings(v, depth + 1))
        elif isinstance(obj, list):
            for item in obj:
                strings.extend(self._extract_strings(item, depth + 1))
        return strings

    def _log_audit(self, action: Dict, agent_id: str, result: str, details: Dict = None):
        """Log with integrity chain."""
        self._audit_sequence += 1

        entry_data = {
            "sequence": self._audit_sequence,
            "timestamp": time.time(),
            "action": str(action.get("type", "unknown"))[:100],
            "agent_id": agent_id,
            "result": result,
            "details": details or {},
            "previous_hash": self._last_audit_hash
        }

        # Compute entry hash
        entry_str = json.dumps(entry_data, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry_data["entry_hash"] = entry_hash

        # Sign entry
        signature = hmac.new(
            self._message_signer._secret,
            entry_hash.encode(),
            hashlib.sha256
        ).hexdigest()
        entry_data["signature"] = signature

        # Update chain
        self._last_audit_hash = entry_hash

        # Append to log (append-only)
        with open(self.audit_path, "a") as f:
            f.write(json.dumps(entry_data) + "\n")

    async def request_hitl(
        self,
        action: Dict,
        reason: str,
        severity: str = "medium"
    ) -> Tuple[str, Dict]:
        """
        Request Human-in-the-Loop approval with structured prompt.
        Prevents prompt manipulation by separating system/user content.
        """
        hitl_id = hashlib.sha256(
            f"{action}{time.time()}{reason}".encode()
        ).hexdigest()[:16]

        # Create structured HITL request (prevents manipulation)
        structured_request = {
            "hitl_id": hitl_id,
            "created_at": time.time(),
            "expires_at": time.time() + self._hitl_timeout,
            "severity": severity,
            "default_on_timeout": "deny" if self._hitl_default_deny else "approve",

            # System-controlled fields (cannot be manipulated)
            "system": {
                "action_type": action.get("type", "unknown"),
                "agent_id": action.get("agent_id", "unknown"),
                "policy_context": "This action requires human approval"
            },

            # User-provided context (displayed separately)
            "user_context": {
                "reason": reason[:500],  # Truncate to prevent overflow
                "action_summary": str(action)[:1000]
            },

            "status": "pending"
        }

        self._pending_hitl[hitl_id] = structured_request

        logger.warning(f"HITL escalation {hitl_id}: {severity} - {reason[:100]}")

        return hitl_id, structured_request

    def resolve_hitl(self, hitl_id: str, approved: bool, resolver_id: str) -> bool:
        """Resolve HITL request with audit trail."""
        if hitl_id not in self._pending_hitl:
            return False

        request = self._pending_hitl[hitl_id]

        # Check expiry
        if time.time() > request["expires_at"]:
            request["status"] = "expired"
            request["resolution"] = request["default_on_timeout"]
            return request["default_on_timeout"] == "approve"

        request["status"] = "resolved"
        request["resolution"] = "approved" if approved else "denied"
        request["resolved_by"] = resolver_id
        request["resolved_at"] = time.time()

        self._log_audit(
            {"type": "hitl_resolution", "hitl_id": hitl_id},
            resolver_id,
            request["resolution"],
            {"severity": request["severity"]}
        )

        return approved

    def verify_audit_integrity(self) -> Tuple[bool, List[str]]:
        """Verify audit log chain integrity."""
        errors = []

        if not self.audit_path.exists():
            return True, []

        prev_hash = "genesis"

        with open(self.audit_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)

                    # Check chain
                    if entry.get("previous_hash") != prev_hash:
                        errors.append(f"Line {line_num}: Chain broken")

                    # Verify entry hash
                    entry_copy = {k: v for k, v in entry.items()
                                  if k not in ["entry_hash", "signature"]}
                    expected_hash = hashlib.sha256(
                        json.dumps(entry_copy, sort_keys=True).encode()
                    ).hexdigest()

                    if entry.get("entry_hash") != expected_hash:
                        errors.append(f"Line {line_num}: Hash mismatch")

                    # Verify signature
                    expected_sig = hmac.new(
                        self._message_signer._secret,
                        entry["entry_hash"].encode(),
                        hashlib.sha256
                    ).hexdigest()

                    if entry.get("signature") != expected_sig:
                        errors.append(f"Line {line_num}: Invalid signature")

                    prev_hash = entry.get("entry_hash", "")

                except json.JSONDecodeError:
                    errors.append(f"Line {line_num}: Invalid JSON")

        return len(errors) == 0, errors

    def get_audit_trail(self, limit: int = 100, verify: bool = True) -> List[Dict]:
        """Get audit trail with optional integrity verification."""
        if verify:
            valid, errors = self.verify_audit_integrity()
            if not valid:
                logger.error(f"Audit integrity compromised: {errors}")

        entries = []
        if self.audit_path.exists():
            with open(self.audit_path, "r") as f:
                lines = f.readlines()[-limit:]
                for line in lines:
                    try:
                        entries.append(json.loads(line))
                    except:
                        pass
        return entries

    def get_security_report(self) -> Dict:
        """Generate security status report."""
        return {
            "rate_limiter": self._rate_limiter.get_usage(),
            "pending_hitl": len(self._pending_hitl),
            "audit_entries": self._audit_sequence,
            "anomaly_report": self._validator.get_anomaly_report(
                since=time.time() - 3600
            )
        }
