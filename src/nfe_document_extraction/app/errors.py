"""Rejections the public API returns before or instead of a model call."""


class RequestRejected(Exception):
    """A request the demo will not run. ``status_code`` is the HTTP status."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
