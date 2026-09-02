"""
app/core/exceptions.py

Centralized exception definitions for the TTS API.
All custom exceptions inherit from TTSAPIError.
FastAPI exception handlers (registered in main.py) convert these to
consistent JSON error responses.

Error response format:
    {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Human-readable description",
            "request_id": "01J..."
        }
    }
"""
from __future__ import annotations

from http import HTTPStatus


class TTSAPIError(Exception):
    """
    Base exception for all TTS API errors.
    Subclass this to create domain-specific exceptions.
    """

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code: str = "INTERNAL_ERROR"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        error_code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.message = message or self.__class__.message
        self.error_code = error_code or self.__class__.error_code
        self.status_code = status_code or self.__class__.status_code
        super().__init__(self.message)

    def to_dict(self, request_id: str | None = None) -> dict:
        payload: dict = {
            "code": self.error_code,
            "message": self.message,
        }
        if request_id:
            payload["request_id"] = request_id
        return {"error": payload}


# ─── Request / Validation Errors (4xx) ───────────────────────────────────────

class ValidationError(TTSAPIError):
    """Input failed Pydantic or custom validation."""
    status_code = 422
    error_code = "VALIDATION_ERROR"
    message = "The request contains invalid or missing fields."


class EmptyTextError(TTSAPIError):
    """Text input was empty or only whitespace."""
    status_code = 422
    error_code = "EMPTY_TEXT"
    message = "Text cannot be empty."


class TextTooLongError(TTSAPIError):
    """Text exceeds the configured maximum length."""
    status_code = 422
    error_code = "TEXT_TOO_LONG"
    message = "Text exceeds the maximum allowed length."

    def __init__(self, length: int, max_length: int) -> None:
        super().__init__(
            message=(
                f"Text length {length:,} characters exceeds the maximum "
                f"of {max_length:,} characters. Use the /tts/async endpoint "
                "for long content."
            )
        )


class UnsupportedLanguageError(TTSAPIError):
    """Language code is not supported by the loaded model."""
    status_code = 422
    error_code = "UNSUPPORTED_LANGUAGE"
    message = "The specified language is not supported by the current model."

    def __init__(self, language: str, supported: list[str]) -> None:
        super().__init__(
            message=(
                f"Language '{language}' is not supported. "
                f"Supported languages: {', '.join(sorted(supported))}"
            )
        )


class UnsupportedVoiceError(TTSAPIError):
    """Voice ID is not available."""
    status_code = 422
    error_code = "UNSUPPORTED_VOICE"
    message = "The specified voice is not available."

    def __init__(self, voice: str) -> None:
        super().__init__(
            message=f"Voice '{voice}' is not available. "
                    "Use GET /api/v1/voices to list available voices."
        )


class UnsupportedFormatError(TTSAPIError):
    """Audio output format is not supported."""
    status_code = 422
    error_code = "UNSUPPORTED_FORMAT"
    message = "The specified audio format is not supported."

    def __init__(self, fmt: str, supported: list[str]) -> None:
        super().__init__(
            message=f"Format '{fmt}' is not supported. Supported: {', '.join(supported)}"
        )


class InvalidSpeedError(TTSAPIError):
    """Speech speed is out of the allowed range."""
    status_code = 422
    error_code = "INVALID_SPEED"
    message = "Speed must be between 0.5 and 2.0."


class RequestTooLargeError(TTSAPIError):
    """HTTP request body exceeds the configured limit."""
    status_code = 413
    error_code = "REQUEST_TOO_LARGE"
    message = "Request payload is too large."


# ─── Authentication / Authorization Errors (4xx) ─────────────────────────────

class AuthenticationError(TTSAPIError):
    """API key is missing or malformed."""
    status_code = 401
    error_code = "AUTHENTICATION_REQUIRED"
    message = "API key is required. Provide it in the X-API-Key header."


class InvalidAPIKeyError(TTSAPIError):
    """API key provided but not valid."""
    status_code = 403
    error_code = "INVALID_API_KEY"
    message = "The provided API key is invalid or has been revoked."


class RateLimitExceededError(TTSAPIError):
    """Client has exceeded the configured rate limit."""
    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"
    message = "Rate limit exceeded. Please slow down your requests."

    def __init__(self, retry_after: int = 60) -> None:
        super().__init__()
        self.retry_after = retry_after


# ─── Resource Errors (4xx) ────────────────────────────────────────────────────

class NotFoundError(TTSAPIError):
    """Requested resource does not exist."""
    status_code = 404
    error_code = "NOT_FOUND"
    message = "The requested resource was not found."

    def __init__(self, resource: str, identifier: str) -> None:
        super().__init__(
            message=f"{resource} '{identifier}' not found."
        )


class ConflictError(TTSAPIError):
    """Resource conflict (e.g., duplicate key)."""
    status_code = 409
    error_code = "CONFLICT"
    message = "Resource conflict."


# ─── Model / Engine Errors (5xx) ──────────────────────────────────────────────

class ModelNotLoadedError(TTSAPIError):
    """TTS model has not been loaded yet."""
    status_code = 503
    error_code = "MODEL_NOT_LOADED"
    message = (
        "The TTS model is not loaded. "
        "The service is starting up or the model failed to initialize."
    )


class ModelLoadError(TTSAPIError):
    """TTS model failed to load."""
    status_code = 503
    error_code = "MODEL_LOAD_ERROR"
    message = "The TTS model failed to load."


class SynthesisError(TTSAPIError):
    """Audio synthesis failed during model inference."""
    status_code = 500
    error_code = "SYNTHESIS_ERROR"
    message = "Audio synthesis failed."


class AudioProcessingError(TTSAPIError):
    """Post-processing of generated audio failed."""
    status_code = 500
    error_code = "AUDIO_PROCESSING_ERROR"
    message = "Audio processing failed."


# ─── Storage / Infrastructure Errors (5xx) ───────────────────────────────────

class StorageError(TTSAPIError):
    """File or object storage operation failed."""
    status_code = 500
    error_code = "STORAGE_ERROR"
    message = "Storage operation failed."


class CacheError(TTSAPIError):
    """Cache operation failed (non-fatal — treated as cache miss)."""
    status_code = 500
    error_code = "CACHE_ERROR"
    message = "Cache operation failed."


class ServiceUnavailableError(TTSAPIError):
    """A required backend service (Redis, DB) is unavailable."""
    status_code = 503
    error_code = "SERVICE_UNAVAILABLE"
    message = "A required service is temporarily unavailable."
