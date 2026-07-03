"""Offline unit tests for OpenAlex rate limiting."""

import time

from oignon.core import ratelimit
from oignon.core.ratelimit import OpenAlexRateLimitError, is_rate_limit, throttle


class TestIsRateLimit:
    def test_detects_429_in_message(self):
        exc = Exception("too many 429 error responses")
        assert is_rate_limit(exc) is True

    def test_detects_429_in_cause_chain(self):
        inner = Exception("too many 429 error responses")
        outer = Exception("Max retries exceeded")
        outer.__cause__ = inner
        assert is_rate_limit(outer) is True

    def test_ignores_other_errors(self):
        assert is_rate_limit(ValueError("connection reset")) is False

    def test_handles_none(self):
        assert is_rate_limit(None) is False


class TestThrottle:
    def test_enforces_requests_per_second(self, monkeypatch):
        monkeypatch.setattr(ratelimit, "MAX_REQUESTS_PER_SECOND", 2)
        ratelimit._request_times.clear()

        start = time.monotonic()
        for _ in range(3):
            throttle()
        elapsed = time.monotonic() - start

        # First two are instant; the third must wait for the window
        assert elapsed >= 0.9

        ratelimit._request_times.clear()

    def test_burst_within_limit_is_instant(self, monkeypatch):
        monkeypatch.setattr(ratelimit, "MAX_REQUESTS_PER_SECOND", 5)
        ratelimit._request_times.clear()

        start = time.monotonic()
        for _ in range(5):
            throttle()
        elapsed = time.monotonic() - start

        assert elapsed < 0.1

        ratelimit._request_times.clear()


class TestErrorMessage:
    def test_mentions_polite_pool(self):
        err = OpenAlexRateLimitError()
        assert "429" in str(err)
        assert "OPENALEX_EMAIL" in str(err)
