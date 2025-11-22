# core/validation/schemas.py
"""
Advanced Input Validation Layer
Provides schema validation, sanitization, and anomaly detection for all system boundaries.
"""

import re
import json
import hashlib
import hmac
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("Système.Validation")

T = TypeVar('T')

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION RESULT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Result of a validation operation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_value: Any = None
    anomaly_score: float = 0.0  # 0.0 = normal, 1.0 = highly anomalous

    def __bool__(self) -> bool:
        return self.valid

class ValidationError(Exception):
    """Raised when validation fails critically."""
    def __init__(self, errors: List[str], anomaly_score: float = 0.0):
        self.errors = errors
        self.anomaly_score = anomaly_score
        super().__init__(f"Validation failed: {'; '.join(errors)}")

# ═══════════════════════════════════════════════════════════════════════════════
# BASE VALIDATORS
# ═══════════════════════════════════════════════════════════════════════════════

class Validator(ABC, Generic[T]):
    """Abstract base validator."""

    @abstractmethod
    def validate(self, value: Any) -> ValidationResult:
        pass

    def __and__(self, other: 'Validator') -> 'CompositeValidator':
        return CompositeValidator([self, other], mode='all')

    def __or__(self, other: 'Validator') -> 'CompositeValidator':
        return CompositeValidator([self, other], mode='any')

class CompositeValidator(Validator):
    """Combines multiple validators."""

    def __init__(self, validators: List[Validator], mode: str = 'all'):
        self.validators = validators
        self.mode = mode  # 'all' or 'any'

    def validate(self, value: Any) -> ValidationResult:
        errors, warnings = [], []
        max_anomaly = 0.0

        for v in self.validators:
            result = v.validate(value)
            if self.mode == 'all' and not result.valid:
                errors.extend(result.errors)
            elif self.mode == 'any' and result.valid:
                return result
            warnings.extend(result.warnings)
            max_anomaly = max(max_anomaly, result.anomaly_score)

        if self.mode == 'any':
            return ValidationResult(False, errors, warnings, anomaly_score=max_anomaly)
        return ValidationResult(len(errors) == 0, errors, warnings, value, max_anomaly)

# ═══════════════════════════════════════════════════════════════════════════════
# STRING VALIDATORS & SANITIZERS
# ═══════════════════════════════════════════════════════════════════════════════

class StringValidator(Validator[str]):
    """Validates and sanitizes strings with security focus."""

    def __init__(
        self,
        min_length: int = 0,
        max_length: int = 10000,
        pattern: str = None,
        normalize_unicode: bool = True,
        strip_control_chars: bool = True,
        allowed_chars: str = None
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = re.compile(pattern) if pattern else None
        self.normalize_unicode = normalize_unicode
        self.strip_control_chars = strip_control_chars
        self.allowed_chars = set(allowed_chars) if allowed_chars else None

    def validate(self, value: Any) -> ValidationResult:
        errors, warnings = [], []
        anomaly = 0.0

        if not isinstance(value, str):
            return ValidationResult(False, [f"Expected string, got {type(value).__name__}"])

        sanitized = value

        # Unicode normalization (prevents obfuscation attacks)
        if self.normalize_unicode:
            original = sanitized
            sanitized = unicodedata.normalize('NFKC', sanitized)
            # Remove zero-width characters
            sanitized = re.sub(r'[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff]', '', sanitized)
            if sanitized != original:
                warnings.append("Unicode normalization applied")
                anomaly += 0.3

        # Strip control characters
        if self.strip_control_chars:
            original = sanitized
            sanitized = ''.join(c for c in sanitized if unicodedata.category(c) != 'Cc' or c in '\n\r\t')
            if sanitized != original:
                warnings.append("Control characters stripped")
                anomaly += 0.4

        # Length checks
        if len(sanitized) < self.min_length:
            errors.append(f"String too short: {len(sanitized)} < {self.min_length}")
        if len(sanitized) > self.max_length:
            errors.append(f"String too long: {len(sanitized)} > {self.max_length}")
            anomaly += 0.5

        # Pattern match
        if self.pattern and not self.pattern.match(sanitized):
            errors.append(f"String does not match required pattern")

        # Allowed characters
        if self.allowed_chars:
            invalid = set(sanitized) - self.allowed_chars
            if invalid:
                errors.append(f"Invalid characters: {invalid}")
                anomaly += 0.3

        return ValidationResult(len(errors) == 0, errors, warnings, sanitized, min(1.0, anomaly))

class ActionTypeValidator(StringValidator):
    """Validates action types with normalization to prevent bypass attacks."""

    DANGEROUS_PATTERNS = [
        r'delete.*prod', r'drop.*table', r'exec.*code', r'bypass.*',
        r'inject.*', r'eval\s*\(', r'__import__', r'os\.system'
    ]

    def __init__(self):
        super().__init__(
            min_length=1,
            max_length=100,
            normalize_unicode=True,
            strip_control_chars=True
        )
        self._dangerous_re = [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS]

    def validate(self, value: Any) -> ValidationResult:
        # First, run base string validation with normalization
        result = super().validate(value)
        if not result.valid:
            return result

        sanitized = result.sanitized_value.lower().strip()

        # Check against dangerous patterns AFTER normalization
        for pattern in self._dangerous_re:
            if pattern.search(sanitized):
                result.errors.append(f"Action matches dangerous pattern")
                result.anomaly_score = max(result.anomaly_score, 0.9)
                result.valid = False
                break

        result.sanitized_value = sanitized
        return result

# ═══════════════════════════════════════════════════════════════════════════════
# NUMERIC VALIDATORS WITH ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class NumericValidator(Validator[Union[int, float]]):
    """Validates numeric values with anomaly detection."""

    def __init__(
        self,
        min_value: float = float('-inf'),
        max_value: float = float('inf'),
        allow_nan: bool = False,
        allow_inf: bool = False,
        historical_stats: Dict = None
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.allow_nan = allow_nan
        self.allow_inf = allow_inf
        self.stats = historical_stats or {}

    def validate(self, value: Any) -> ValidationResult:
        errors, warnings = [], []
        anomaly = 0.0

        # Type conversion
        try:
            if isinstance(value, str):
                value = float(value)
            elif not isinstance(value, (int, float)):
                return ValidationResult(False, [f"Expected numeric, got {type(value).__name__}"])
        except (ValueError, TypeError):
            return ValidationResult(False, ["Cannot convert to numeric"])

        # NaN/Inf checks (critical for HRL/reward hacking prevention)
        import math
        if math.isnan(value):
            if not self.allow_nan:
                errors.append("NaN values not allowed")
                anomaly = 1.0
            return ValidationResult(len(errors) == 0, errors, warnings, 0.0, anomaly)

        if math.isinf(value):
            if not self.allow_inf:
                errors.append("Infinite values not allowed")
                anomaly = 1.0
            return ValidationResult(len(errors) == 0, errors, warnings, self.max_value if value > 0 else self.min_value, anomaly)

        # Range checks
        if value < self.min_value:
            errors.append(f"Value {value} below minimum {self.min_value}")
            anomaly += 0.5
        if value > self.max_value:
            errors.append(f"Value {value} above maximum {self.max_value}")
            anomaly += 0.5

        # Statistical anomaly detection
        if self.stats:
            mean = self.stats.get('mean', 0)
            std = self.stats.get('std', 1)
            if std > 0:
                z_score = abs(value - mean) / std
                if z_score > 3:
                    warnings.append(f"Value is {z_score:.1f} std devs from mean")
                    anomaly += min(0.5, z_score / 10)

        return ValidationResult(len(errors) == 0, errors, warnings, value, min(1.0, anomaly))

class RewardValidator(NumericValidator):
    """Specialized validator for RL rewards - prevents reward hacking."""

    def __init__(self, reward_range: tuple = (-10.0, 10.0)):
        super().__init__(
            min_value=reward_range[0],
            max_value=reward_range[1],
            allow_nan=False,
            allow_inf=False
        )
        self._recent_rewards: List[float] = []
        self._max_history = 1000

    def validate(self, value: Any) -> ValidationResult:
        result = super().validate(value)
        if not result.valid:
            return result

        reward = result.sanitized_value

        # Track reward history for anomaly detection
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > self._max_history:
            self._recent_rewards.pop(0)

        # Detect sudden reward spikes (reward hacking indicator)
        if len(self._recent_rewards) > 10:
            recent_mean = sum(self._recent_rewards[-10:]) / 10
            overall_mean = sum(self._recent_rewards) / len(self._recent_rewards)

            if abs(reward) > abs(overall_mean) * 10 and abs(reward) > 1:
                result.warnings.append(f"Reward spike detected: {reward} vs mean {overall_mean:.2f}")
                result.anomaly_score = max(result.anomaly_score, 0.8)

        return result

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA VALIDATOR FOR COMPLEX OBJECTS
# ═══════════════════════════════════════════════════════════════════════════════

class SchemaValidator(Validator[Dict]):
    """Validates complex objects against a schema."""

    def __init__(self, schema: Dict[str, Validator], strict: bool = False):
        self.schema = schema
        self.strict = strict  # If True, reject unknown fields

    def validate(self, value: Any) -> ValidationResult:
        if not isinstance(value, dict):
            return ValidationResult(False, [f"Expected dict, got {type(value).__name__}"])

        errors, warnings = [], []
        sanitized = {}
        max_anomaly = 0.0

        # Check required fields
        for field_name, validator in self.schema.items():
            if field_name not in value:
                if not field_name.startswith('?'):  # ? prefix = optional
                    errors.append(f"Missing required field: {field_name}")
                continue

            field_result = validator.validate(value[field_name])
            if not field_result.valid:
                errors.extend([f"{field_name}: {e}" for e in field_result.errors])
            warnings.extend([f"{field_name}: {w}" for w in field_result.warnings])
            sanitized[field_name] = field_result.sanitized_value
            max_anomaly = max(max_anomaly, field_result.anomaly_score)

        # Check for unknown fields
        if self.strict:
            schema_keys = {k.lstrip('?') for k in self.schema.keys()}
            unknown = set(value.keys()) - schema_keys
            if unknown:
                errors.append(f"Unknown fields: {unknown}")
                max_anomaly = max(max_anomaly, 0.5)
        else:
            # Copy unknown fields as-is
            for k, v in value.items():
                if k not in sanitized:
                    sanitized[k] = v

        return ValidationResult(len(errors) == 0, errors, warnings, sanitized, max_anomaly)

# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE SIGNING & VERIFICATION (A2A Security)
# ═══════════════════════════════════════════════════════════════════════════════

class MessageSigner:
    """Signs and verifies A2A messages to prevent spoofing."""

    def __init__(self, secret_key: bytes = None):
        self._secret = secret_key or self._generate_key()
        self._nonce_cache: Dict[str, float] = {}
        self._nonce_ttl = 300  # 5 minutes

    def _generate_key(self) -> bytes:
        import secrets
        return secrets.token_bytes(32)

    def sign_message(self, message: Dict) -> Dict:
        """Add signature and nonce to message."""
        import secrets

        nonce = secrets.token_hex(16)
        timestamp = time.time()

        # Create canonical representation
        canonical = json.dumps({
            'sender': message.get('source_agent'),
            'target': message.get('target_agent'),
            'action': message.get('action'),
            'nonce': nonce,
            'timestamp': timestamp
        }, sort_keys=True)

        signature = hmac.new(
            self._secret,
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()

        return {
            **message,
            '_nonce': nonce,
            '_timestamp': timestamp,
            '_signature': signature
        }

    def verify_message(self, message: Dict) -> ValidationResult:
        """Verify message signature and check for replay."""
        errors, warnings = [], []

        nonce = message.get('_nonce')
        timestamp = message.get('_timestamp')
        signature = message.get('_signature')

        if not all([nonce, timestamp, signature]):
            return ValidationResult(False, ["Message not signed"])

        # Check timestamp freshness
        age = time.time() - timestamp
        if age > self._nonce_ttl:
            return ValidationResult(False, [f"Message expired ({age:.0f}s old)"])
        if age < -60:  # Allow 1 min clock skew
            return ValidationResult(False, ["Message from future"])

        # Check for replay
        if nonce in self._nonce_cache:
            return ValidationResult(False, ["Replay attack detected"], anomaly_score=1.0)

        # Verify signature
        canonical = json.dumps({
            'sender': message.get('source_agent'),
            'target': message.get('target_agent'),
            'action': message.get('action'),
            'nonce': nonce,
            'timestamp': timestamp
        }, sort_keys=True)

        expected = hmac.new(
            self._secret,
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected):
            return ValidationResult(False, ["Invalid signature"], anomaly_score=1.0)

        # Cache nonce
        self._nonce_cache[nonce] = timestamp
        self._cleanup_nonce_cache()

        return ValidationResult(True, [], warnings)

    def _cleanup_nonce_cache(self):
        """Remove expired nonces."""
        now = time.time()
        expired = [n for n, t in self._nonce_cache.items() if now - t > self._nonce_ttl]
        for n in expired:
            del self._nonce_cache[n]

# ═══════════════════════════════════════════════════════════════════════════════
# PREDEFINED SCHEMAS FOR SYSTÈME COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

class SystemeSchemas:
    """Predefined validation schemas for all system components."""

    # Task schema
    TASK = SchemaValidator({
        'id': StringValidator(min_length=1, max_length=64, pattern=r'^[\w\-]+$'),
        'capability_required': StringValidator(min_length=1, max_length=50),
        'payload': SchemaValidator({}, strict=False),
        '?priority': StringValidator(pattern=r'^(LOW|MEDIUM|HIGH|CRITICAL)$'),
        '?source_agent': StringValidator(max_length=64),
    }, strict=False)

    # Action schema (for governance)
    ACTION = SchemaValidator({
        'type': StringValidator(min_length=1, max_length=100, normalize_unicode=True, strip_control_chars=True),
        '?payload': SchemaValidator({}, strict=False),
        '?permissions': SchemaValidator({}, strict=False),
    }, strict=False)

    # HRL Experience schema
    EXPERIENCE = SchemaValidator({
        'state': SchemaValidator({}, strict=False),
        'action': StringValidator(max_length=50),
        'reward': RewardValidator((-10.0, 10.0)),
        'next_state': SchemaValidator({}, strict=False),
        '?achieved_goal': StringValidator(max_length=100),
        '?desired_goal': StringValidator(max_length=100),
    }, strict=False)

    # Playbook bullet schema
    BULLET = SchemaValidator({
        'content': StringValidator(min_length=5, max_length=1000, normalize_unicode=True),
        '?section': StringValidator(pattern=r'^(defaults|strategies|pitfalls|tools|examples)$'),
        '?tags': SchemaValidator({}, strict=False),
        '?helpful_count': NumericValidator(min_value=0, max_value=100000),
    }, strict=False)

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationMiddleware:
    """Middleware for automatic validation at system boundaries."""

    def __init__(self, anomaly_threshold: float = 0.7):
        self.anomaly_threshold = anomaly_threshold
        self._anomaly_log: List[Dict] = []

    def validate_and_sanitize(
        self,
        data: Any,
        schema: Validator,
        context: str = "unknown"
    ) -> Any:
        """Validate data and return sanitized version or raise."""
        result = schema.validate(data)

        if result.anomaly_score >= self.anomaly_threshold:
            self._log_anomaly(data, result, context)

        if not result.valid:
            logger.warning(f"[Validation:{context}] Failed: {result.errors}")
            raise ValidationError(result.errors, result.anomaly_score)

        if result.warnings:
            logger.info(f"[Validation:{context}] Warnings: {result.warnings}")

        return result.sanitized_value

    def _log_anomaly(self, data: Any, result: ValidationResult, context: str):
        """Log anomalous input for review."""
        self._anomaly_log.append({
            'timestamp': time.time(),
            'context': context,
            'anomaly_score': result.anomaly_score,
            'errors': result.errors,
            'warnings': result.warnings,
            'data_hash': hashlib.md5(str(data).encode()).hexdigest()
        })
        logger.warning(f"[Anomaly:{context}] Score {result.anomaly_score:.2f}: {result.warnings}")

    def get_anomaly_report(self, since: float = 0) -> List[Dict]:
        """Get anomaly log entries since timestamp."""
        return [a for a in self._anomaly_log if a['timestamp'] >= since]

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════════

class GlobalRateLimiter:
    """Global rate limiting that aggregates across all agents/sessions."""

    def __init__(
        self,
        global_limit: int = 1000,
        per_agent_limit: int = 100,
        window_seconds: int = 60
    ):
        self.global_limit = global_limit
        self.per_agent_limit = per_agent_limit
        self.window = window_seconds
        self._global_requests: List[float] = []
        self._agent_requests: Dict[str, List[float]] = {}

    def check(self, agent_id: str = None) -> tuple[bool, str]:
        """Check if request is allowed."""
        now = time.time()
        cutoff = now - self.window

        # Clean and check global
        self._global_requests = [t for t in self._global_requests if t > cutoff]
        if len(self._global_requests) >= self.global_limit:
            return False, f"Global rate limit exceeded ({self.global_limit}/{self.window}s)"

        # Clean and check per-agent
        if agent_id:
            if agent_id not in self._agent_requests:
                self._agent_requests[agent_id] = []
            self._agent_requests[agent_id] = [
                t for t in self._agent_requests[agent_id] if t > cutoff
            ]
            if len(self._agent_requests[agent_id]) >= self.per_agent_limit:
                return False, f"Agent rate limit exceeded ({self.per_agent_limit}/{self.window}s)"

        # Record request
        self._global_requests.append(now)
        if agent_id:
            self._agent_requests[agent_id].append(now)

        return True, "OK"

    def get_usage(self) -> Dict:
        """Get current rate limit usage."""
        now = time.time()
        cutoff = now - self.window

        self._global_requests = [t for t in self._global_requests if t > cutoff]

        return {
            'global_used': len(self._global_requests),
            'global_limit': self.global_limit,
            'global_pct': len(self._global_requests) / self.global_limit * 100,
            'agents': {
                aid: len([t for t in times if t > cutoff])
                for aid, times in self._agent_requests.items()
            }
        }
