"""
Small throttled JSON-over-HTTP client with retries, shared by the photo API fetchers.
"""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

USER_AGENT = "spatial-pipeline (+https://github.com/GIlbertoRCP/spatial_pipeline)"
RETRYABLE_STATUS = (429, 500, 502, 503, 504)


class ThrottledJSONClient:
    """
    GET-only JSON client that spaces requests out and retries transient failures.

    Args:
        min_interval: Minimum seconds between requests.
        max_retries: Retries for network errors and HTTP 429 / 5xx responses.
        timeout: Per-request timeout in seconds.
        sleep: Sleep function (injectable for tests).
        opener: Callable taking a urllib Request and returning a response (injectable for tests).
        user_agent: User-Agent header sent with every request.
    """

    def __init__(
        self,
        min_interval: float = 1.0,
        max_retries: int = 4,
        timeout: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        opener: Callable[..., Any] | None = None,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.user_agent = user_agent
        self._sleep = sleep
        self._opener = opener or urllib.request.urlopen
        self._last_request = -math.inf
        self.requests_made = 0

    def _throttle(self) -> None:
        wait = self._last_request + self.min_interval - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()

    def get_json(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        """
        Sends a GET request and returns the decoded JSON body.

        Args:
            url: Endpoint URL.
            params: Query parameters.

        Returns:
            dict[str, Any]: Decoded JSON response.

        Raises:
            urllib.error.HTTPError: For non-retryable HTTP errors, or after the last retry.
            urllib.error.URLError: For network errors after the last retry.
        """
        request = urllib.request.Request(
            url + "?" + urllib.parse.urlencode(params), headers={"User-Agent": self.user_agent}
        )
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.requests_made += 1
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                return payload
            except urllib.error.HTTPError as e:
                if e.code not in RETRYABLE_STATUS or attempt == self.max_retries:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.max_retries:
                    raise
            backoff = min(60.0, 2.0 ** (attempt + 1))
            logger.warning("Request to %s failed (attempt %d); retrying in %.0fs", url, attempt + 1, backoff)
            self._sleep(backoff)
        raise AssertionError("unreachable")  # pragma: no cover
