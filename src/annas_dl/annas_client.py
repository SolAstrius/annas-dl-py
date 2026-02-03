"""Anna's Archive API client with mirror failover."""

import logging
from dataclasses import dataclass
from typing import Self

import httpx

logger = logging.getLogger(__name__)

# Available Anna's Archive mirror domains (in order of preference)
MIRROR_DOMAINS = [
    "https://annas-archive.li",
    "https://annas-archive.pm",
    "https://annas-archive.in",
]


@dataclass
class BookMetadata:
    """Metadata fetched from Anna's Archive API."""

    title: str
    author: str
    publisher: str
    format: str
    year: str
    language: str | None = None
    size: str | None = None
    ipfs_cid: str | None = None


class AnnasClientError(Exception):
    """Error from Anna's Archive API."""


class AnnasClient:
    """Async client for Anna's Archive with automatic mirror failover."""

    def __init__(self, http: httpx.AsyncClient, timeout: float = 15.0):
        self._http = http
        self._timeout = timeout
        self._current_domain_idx = 0

    @classmethod
    def create(cls, timeout: float = 15.0) -> Self:
        """Create a client with a new HTTP client."""
        http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0"
            },
        )
        return cls(http, timeout)

    @property
    def current_domain(self) -> str:
        """Get the current active domain."""
        return MIRROR_DOMAINS[self._current_domain_idx % len(MIRROR_DOMAINS)]

    def _rotate_domain(self) -> str:
        """Rotate to the next domain, returns the new domain."""
        old_domain = self.current_domain
        self._current_domain_idx = (self._current_domain_idx + 1) % len(MIRROR_DOMAINS)
        new_domain = self.current_domain
        logger.info(
            "Rotating Anna's Archive mirror: %s -> %s", old_domain, new_domain
        )
        return new_domain

    @staticmethod
    def _is_recoverable_error(exc: Exception) -> bool:
        """Check if an error is recoverable by trying another domain."""
        if isinstance(exc, httpx.TimeoutException | httpx.ConnectError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code >= 500
        return False

    async def get_download_url(
        self, hash: str, secret_key: str, domain_index: int | None = None
    ) -> str:
        """Get download URL for a book with automatic failover.

        Args:
            hash: MD5 hash of the book
            secret_key: Anna's Archive API key
            domain_index: Optional CDN server index (0-9+)

        Returns:
            Direct download URL from CDN
        """
        last_error: Exception | None = None

        for attempt in range(len(MIRROR_DOMAINS)):
            domain = self.current_domain
            url = f"{domain}/dyn/api/fast_download.json?md5={hash}&key={secret_key}"
            if domain_index is not None:
                url += f"&domain_index={domain_index}"

            logger.debug(
                "Fetching download URL (attempt %d): %s",
                attempt + 1,
                url.replace(secret_key, "***"),
            )

            try:
                response = await self._http.get(url, timeout=self._timeout)
                response.raise_for_status()
                data = response.json()

                if error := data.get("error"):
                    raise AnnasClientError(f"API error: {error}")

                download_url = data.get("download_url")
                if not download_url:
                    raise AnnasClientError("No download URL in response")

                return download_url

            except Exception as exc:
                logger.warning(
                    "Download URL request failed (domain=%s, attempt=%d): %s",
                    domain,
                    attempt + 1,
                    exc,
                )

                if self._is_recoverable_error(exc) and attempt < len(MIRROR_DOMAINS) - 1:
                    self._rotate_domain()
                    last_error = exc
                    continue

                raise AnnasClientError(str(exc)) from exc

        raise AnnasClientError(
            f"All mirrors failed: {last_error}"
        ) from last_error

    async def fetch_metadata(self, hash: str) -> BookMetadata:
        """Fetch metadata by hash with automatic failover."""
        last_error: Exception | None = None

        for attempt in range(len(MIRROR_DOMAINS)):
            domain = self.current_domain
            url = f"{domain}/db/aarecord_elasticsearch/md5:{hash}.json"

            logger.debug("Fetching metadata (attempt %d): %s", attempt + 1, url)

            try:
                response = await self._http.get(url, timeout=self._timeout)
                response.raise_for_status()
                data = response.json()

                unified = data.get("file_unified_data", {})

                # Extract IPFS CID if available
                ipfs_infos = unified.get("ipfs_infos", [])
                ipfs_cid = ipfs_infos[0].get("ipfs_cid") if ipfs_infos else None

                # Extract language
                lang_codes = unified.get("language_codes", [])
                language = lang_codes[0] if lang_codes else None

                # Format file size
                size_bytes = unified.get("filesize_best")
                size = _format_file_size(size_bytes) if size_bytes else None

                return BookMetadata(
                    title=unified.get("title_best", ""),
                    author=unified.get("author_best", ""),
                    publisher=unified.get("publisher_best", ""),
                    format=unified.get("extension_best", ""),
                    year=unified.get("year_best", ""),
                    language=language,
                    size=size,
                    ipfs_cid=ipfs_cid,
                )

            except Exception as exc:
                logger.warning(
                    "Metadata request failed (domain=%s, attempt=%d): %s",
                    domain,
                    attempt + 1,
                    exc,
                )

                if self._is_recoverable_error(exc) and attempt < len(MIRROR_DOMAINS) - 1:
                    self._rotate_domain()
                    last_error = exc
                    continue

                raise AnnasClientError(str(exc)) from exc

        raise AnnasClientError(
            f"All mirrors failed: {last_error}"
        ) from last_error

    async def close(self):
        """Close the HTTP client."""
        await self._http.aclose()


def _format_file_size(bytes_: int) -> str | None:
    """Format file size in human-readable form."""
    if bytes_ <= 0:
        return None
    for unit, threshold in [("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)]:
        if bytes_ >= threshold:
            return f"{bytes_ / threshold:.1f} {unit}"
    return f"{bytes_} bytes"
