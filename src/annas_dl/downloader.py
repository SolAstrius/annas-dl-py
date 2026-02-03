"""Book downloader with parallel CDN failover.

This module leverages Python 3.13's free-threaded mode to perform
truly parallel downloads without GIL contention.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .annas_client import AnnasClient
    from .config import Settings

logger = logging.getLogger(__name__)

# Common ebook extensions for format detection
EBOOK_EXTENSIONS = ["epub", "pdf", "mobi", "azw3", "fb2", "djvu", "cbr", "cbz"]


@dataclass
class DownloadResult:
    """Result of a successful download."""

    content: bytes
    format: str
    hash: str
    cdn_host: str
    duration_ms: int
    size_bytes: int


class DownloadError(Exception):
    """Download failed after all retry attempts."""

    def __init__(self, message: str, last_status: int | None = None):
        super().__init__(message)
        self.last_status = last_status


def detect_format_from_url(url: str) -> str | None:
    """Detect file format from Anna's Archive download URL.

    URLs look like: https://host/path/hash.epub~/token/filename.epub
    """
    url_lower = url.lower()
    for ext in EBOOK_EXTENSIONS:
        if f".{ext}" in url_lower:
            return ext
    return None


async def download_book(
    client: "AnnasClient",
    settings: "Settings",
    secret_key: str,
    hash: str,
    format_hint: str = "pdf",
) -> DownloadResult:
    """Download a book from Anna's Archive CDN with failover.

    Tries multiple CDN servers (domain_index) starting from settings.cdn_start_index.
    Each attempt gets a fresh download URL which may route to a different CDN node.

    Args:
        client: Anna's Archive API client
        settings: Service settings
        hash: MD5 hash of the book
        format_hint: Expected format (used if detection fails)

    Returns:
        DownloadResult with content bytes and metadata

    Raises:
        DownloadError: If all CDN attempts fail
    """
    last_error: Exception | None = None
    last_status: int | None = None
    detected_format = format_hint

    for attempt in range(settings.cdn_max_attempts):
        cdn_index = settings.cdn_start_index + attempt
        attempt_start = time.monotonic()

        logger.info(
            "Starting download attempt %d/%d (cdn_index=%d) for hash=%s",
            attempt + 1,
            settings.cdn_max_attempts,
            cdn_index,
            hash,
        )

        # Step 1: Get download URL from API (with specific CDN index)
        try:
            download_url = await client.get_download_url(
                hash, secret_key, domain_index=cdn_index
            )
        except Exception as exc:
            logger.warning(
                "Failed to get download URL (attempt=%d, cdn_index=%d): %s",
                attempt + 1,
                cdn_index,
                exc,
            )
            last_error = exc
            await asyncio.sleep(0.5)
            continue

        # Detect format from URL
        if url_format := detect_format_from_url(download_url):
            detected_format = url_format

        # Extract CDN host for logging
        cdn_host = download_url.split("/")[2] if "/" in download_url else "unknown"
        logger.info("Got CDN URL from %s (cdn_index=%d)", cdn_host, cdn_index)

        # Step 2: Download from CDN
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=settings.cdn_connect_timeout,
                    read=settings.cdn_download_timeout,
                    write=settings.cdn_download_timeout,
                    pool=settings.cdn_download_timeout,
                ),
                follow_redirects=True,
            ) as http:
                response = await http.get(download_url)

                if not response.is_success:
                    last_status = response.status_code
                    logger.warning(
                        "CDN returned error status %d (cdn_host=%s, attempt=%d)",
                        response.status_code,
                        cdn_host,
                        attempt + 1,
                    )

                    # Retry on 5xx errors
                    if 500 <= response.status_code < 600:
                        last_error = DownloadError(
                            f"CDN returned {response.status_code}",
                            response.status_code,
                        )
                        await asyncio.sleep(0.5)
                        continue

                    raise DownloadError(
                        f"CDN returned {response.status_code}",
                        response.status_code,
                    )

                content = response.content
                duration_ms = int((time.monotonic() - attempt_start) * 1000)
                size_mb = len(content) / 1024 / 1024

                logger.info(
                    "Download complete: %.2f MB in %d ms from %s (attempt=%d)",
                    size_mb,
                    duration_ms,
                    cdn_host,
                    attempt + 1,
                )

                return DownloadResult(
                    content=content,
                    format=detected_format,
                    hash=hash,
                    cdn_host=cdn_host,
                    duration_ms=duration_ms,
                    size_bytes=len(content),
                )

        except httpx.TimeoutException as exc:
            logger.warning(
                "CDN timeout (cdn_host=%s, attempt=%d): %s",
                cdn_host,
                attempt + 1,
                exc,
            )
            last_error = exc
            await asyncio.sleep(0.5)
            continue

        except httpx.ConnectError as exc:
            logger.warning(
                "CDN connection failed (cdn_host=%s, attempt=%d): %s",
                cdn_host,
                attempt + 1,
                exc,
            )
            last_error = exc
            await asyncio.sleep(0.5)
            continue

        except Exception as exc:
            logger.warning(
                "Unexpected error during download (cdn_host=%s, attempt=%d): %s",
                cdn_host,
                attempt + 1,
                exc,
            )
            last_error = exc
            await asyncio.sleep(0.5)
            continue

    # All attempts failed
    raise DownloadError(
        f"All {settings.cdn_max_attempts} CDN attempts failed: {last_error}",
        last_status,
    )


async def download_books_parallel(
    client: "AnnasClient",
    settings: "Settings",
    hashes: list[str],
    format_hints: dict[str, str] | None = None,
) -> dict[str, DownloadResult | DownloadError]:
    """Download multiple books in parallel.

    With free-threaded Python 3.13, this achieves true parallelism
    when combined with thread-based concurrency.

    Args:
        client: Anna's Archive API client
        settings: Service settings
        hashes: List of book hashes to download
        format_hints: Optional dict mapping hash -> expected format

    Returns:
        Dict mapping hash -> DownloadResult or DownloadError
    """
    format_hints = format_hints or {}

    async def download_one(hash: str) -> tuple[str, DownloadResult | DownloadError]:
        try:
            result = await download_book(
                client, settings, hash, format_hints.get(hash, "pdf")
            )
            return hash, result
        except DownloadError as exc:
            return hash, exc

    # Use semaphore to limit concurrent downloads
    semaphore = asyncio.Semaphore(settings.max_concurrent_downloads)

    async def bounded_download(hash: str) -> tuple[str, DownloadResult | DownloadError]:
        async with semaphore:
            return await download_one(hash)

    # Run all downloads concurrently
    tasks = [bounded_download(h) for h in hashes]
    results = await asyncio.gather(*tasks)

    return dict(results)
