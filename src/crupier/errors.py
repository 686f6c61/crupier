"""Typed exceptions for Crupier."""


class CrupierError(Exception):
    """Base exception for all Crupier errors."""


class CrupierConfigError(CrupierError):
    """Raised when project or runtime configuration is invalid."""

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


class CrupierPolicyError(CrupierError):
    """Raised when a request or route violates configured policy."""


class CrupierProviderAuthError(CrupierError):
    """Raised when a provider credential is missing or rejected."""

    def __init__(self, message: str, *, provider: str | None = None, env_key: str | None = None, hint: str | None = None):
        super().__init__(message)
        self.provider = provider
        self.env_key = env_key
        self.hint = hint


class CrupierProviderRateLimitError(CrupierError):
    """Raised when a provider rate limit blocks execution."""


class CrupierProviderUnavailableError(CrupierError):
    """Raised when a provider or adapter cannot execute a route."""

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class CrupierModelUnsupportedError(CrupierError):
    """Raised when a model cannot satisfy required request capabilities."""


class CrupierRouteValidationError(CrupierError):
    """Raised when a proposed route fails constraint validation."""


class CrupierBudgetExceededError(CrupierError):
    """Raised when estimated or actual cost exceeds a hard budget."""


class CrupierExecutionLimitError(CrupierError):
    """Raised when a live route exhausts its call or latency budget."""


class CrupierToolApprovalRequired(CrupierError):
    """Raised when a route wants to execute a tool that requires human approval."""


class CrupierStructuredOutputError(CrupierError):
    """Raised when structured output cannot be validated or repaired."""


class CrupierUpdateRequiresConfirmation(CrupierError):
    """Raised when an update changes active routing recommendations and needs approval."""
