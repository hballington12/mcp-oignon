"""Client-side rate limiting and 429 detection for OpenAlex calls.

OpenAlex allows 10 requests/second. Parallel batch fetches and
multi-strategy searches can burst past that, after which OpenAlex 429s
every request for a while. Previously those errors were swallowed and
surfaced as "paper not found", so agents gave up instead of retrying.
All API calls go through throttle(), and 429s raise
OpenAlexRateLimitError so tools can report the real cause.
"""

import threading
import time
from collections import deque

# Conservative margin under OpenAlex's documented 10 req/s
MAX_REQUESTS_PER_SECOND = 6


class OpenAlexRateLimitError(Exception):
    """OpenAlex is rate limiting us (HTTP 429)."""

    def __init__(self):
        super().__init__(
            "OpenAlex rate limit exceeded (HTTP 429). Wait a minute and retry. "
            "Setting OPENALEX_EMAIL enables the more reliable polite pool."
        )


def is_rate_limit(exc: BaseException | None) -> bool:
    """Detect a 429 anywhere in the exception chain."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if "429" in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


_lock = threading.Lock()
_request_times: deque[float] = deque()


def throttle() -> None:
    """Block until a request slot is free (global, thread-safe).

    Sliding one-second window shared by all threads in the process.
    """
    while True:
        with _lock:
            now = time.monotonic()
            while _request_times and now - _request_times[0] > 1.0:
                _request_times.popleft()
            if len(_request_times) < MAX_REQUESTS_PER_SECOND:
                _request_times.append(now)
                return
            wait = 1.0 - (now - _request_times[0])
        time.sleep(max(wait, 0.005))
