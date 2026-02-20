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

from .annas_client import (
    InvalidKeyError,
    NoDownloadsLeftError,
    NotMemberError,
    RecordNotFoundError,
)
from .torrent import TorrentError, TorrentPath, TorrentSession

if TYPE_CHECKING:
    from .annas_client import AnnasClient
    from .config import Settings

logger = logging.getLogger(__name__)

# Global torrent session — initialized in main.py lifespan
_torrent_session: TorrentSession | None = None


def set_torrent_session(session: TorrentSession | None) -> None:
    """Set the global torrent session (called from main.py lifespan)."""
    global _torrent_session
    _torrent_session = session


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
    # Quota info from Anna's Archive API
    downloads_left: int = 0
    downloads_per_day: int = 0
    downloads_done_today: int = 0


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
            api_result = await client.get_download_url(
                hash, secret_key, domain_index=cdn_index
            )
            download_url = api_result.download_url
            quota_info = (api_result.downloads_left, api_result.downloads_per_day, api_result.downloads_done_today)
        except (NoDownloadsLeftError, InvalidKeyError, NotMemberError, RecordNotFoundError) as exc:
            # Non-retriable API errors — break out to IPFS fallback
            last_error = exc
            break
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
        logger.info("Got CDN URL from %s (cdn_index=%d, downloads_left=%d)", cdn_host, cdn_index, quota_info[0])

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
                    downloads_left=quota_info[0],
                    downloads_per_day=quota_info[1],
                    downloads_done_today=quota_info[2],
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

    # All CDN attempts failed — try torrent fallback
    # Skip for auth errors (InvalidKeyError, NotMemberError) since they
    # indicate account problems, not content availability issues
    fallback_eligible = not isinstance(last_error, (InvalidKeyError, NotMemberError))

    torrent_exc: Exception | None = None
    if fallback_eligible and settings.torrent_enabled and _torrent_session is not None:
        reason = type(last_error).__name__ if last_error else "CDN exhausted"
        logger.info("CDN failed for hash=%s (%s), trying torrent fallback", hash, reason)
        try:
            return await download_from_torrent(
                client, settings, hash, format_hint
            )
        except Exception as exc:
            logger.warning("Torrent fallback failed for hash=%s: %s", hash, exc)
            torrent_exc = exc

    if fallback_eligible and settings.ipfs_enabled and settings.ipfs_gateways:
        reason = type(last_error).__name__ if last_error else "CDN+torrent exhausted"
        logger.info("Trying IPFS fallback for hash=%s (%s)", hash, reason)
        try:
            return await download_from_ipfs(client, settings, hash, format_hint)
        except DownloadError as ipfs_exc:
            logger.warning("IPFS fallback also failed for hash=%s: %s", hash, ipfs_exc)

    # If torrent was attempted and failed, include both CDN and torrent reasons
    if torrent_exc is not None:
        cdn_reason = type(last_error).__name__ if last_error else "unknown"
        raise DownloadError(
            f"CDN failed ({cdn_reason}: {last_error}), "
            f"torrent fallback also failed ({torrent_exc})",
        )

    # Re-raise typed errors so callers (main.py) can map them to proper HTTP status
    if isinstance(last_error, NoDownloadsLeftError):
        raise last_error
    if isinstance(last_error, RecordNotFoundError):
        raise last_error
    if isinstance(last_error, InvalidKeyError):
        raise last_error
    if isinstance(last_error, NotMemberError):
        raise last_error

    raise DownloadError(
        f"All {settings.cdn_max_attempts} CDN attempts failed: {last_error}",
        last_status,
    )


async def download_from_torrent(
    client: "AnnasClient",
    settings: "Settings",
    hash: str,
    format_hint: str = "pdf",
) -> DownloadResult:
    """Try downloading a book via BitTorrent using torrent paths from metadata.

    Fetches torrent path info, downloads the .torrent file, then uses
    libtorrent's selective download to grab only the target file.

    Raises:
        DownloadError: If no torrent paths or download fails
        TorrentError: If libtorrent-level error occurs
    """
    if _torrent_session is None:
        raise DownloadError("Torrent session not initialized")

    # Fetch torrent paths from metadata
    try:
        torrent_paths = await client.fetch_torrent_paths(hash)
    except Exception as exc:
        raise DownloadError(f"Failed to fetch torrent paths: {exc}") from exc

    if not torrent_paths:
        raise DownloadError(f"No torrent paths available for hash={hash}")

    # Try each torrent path
    last_error: Exception | None = None
    for tp_data in torrent_paths:
        tp = TorrentPath(
            collection=tp_data.get("collection", ""),
            torrent_path=tp_data.get("torrent_path", ""),
            file_level1=tp_data.get("file_level1", ""),
            file_level2=tp_data.get("file_level2", ""),
        )

        if not tp.torrent_path or not tp.file_level1:
            continue

        logger.info(
            "Trying torrent path: %s (file=%s)",
            tp.torrent_path,
            tp.file_level1,
        )

        try:
            # Use the client's current mirror domain for fetching .torrent
            domain = client.current_domain
            async with httpx.AsyncClient(
                follow_redirects=True,
                headers={"User-Agent": "annas-dl/0.1"},
            ) as http:
                # Apply session cookies if available
                if client.session:
                    http.cookies.update(client.session.as_cookies_dict())

                result = await _torrent_session.download_file(
                    http, domain, tp, timeout=settings.torrent_timeout
                )

            return DownloadResult(
                content=result.content,
                format=format_hint,
                hash=hash,
                cdn_host=f"torrent:{tp.collection}",
                duration_ms=int(result.duration_s * 1000),
                size_bytes=len(result.content),
            )

        except TorrentError as exc:
            logger.warning("Torrent download failed for path=%s: %s", tp.torrent_path, exc)
            last_error = exc
            continue

    raise DownloadError(
        f"All torrent paths failed for hash={hash}: {last_error}",
    )


async def download_from_ipfs(
    client: "AnnasClient",
    settings: "Settings",
    hash: str,
    format_hint: str = "pdf",
) -> DownloadResult:
    """Try downloading a book from IPFS gateways using CIDs from metadata.

    Fetches metadata to get IPFS CIDs, then tries each gateway in order.
    This is meant as a fallback when CDN downloads fail.

    Args:
        client: Anna's Archive API client (for metadata fetch)
        settings: Service settings with gateway list
        hash: MD5 hash of the book
        format_hint: Expected format

    Returns:
        DownloadResult with content bytes

    Raises:
        DownloadError: If no IPFS CIDs available or all gateways fail
    """
    # Fetch metadata to get IPFS CIDs
    try:
        meta = await client.fetch_metadata(hash)
    except Exception as exc:
        raise DownloadError(f"Failed to fetch metadata for IPFS CIDs: {exc}") from exc

    ipfs_cids = [
        info["ipfs_cid"]
        for info in meta.ipfs_infos
        if info.get("ipfs_cid")
    ]

    if not ipfs_cids:
        raise DownloadError(f"No IPFS CIDs available for hash={hash}")

    # Detect format from metadata
    detected_format = meta.extension_best or format_hint

    last_error: Exception | None = None

    for cid in ipfs_cids:
        for gateway in settings.ipfs_gateways:
            gateway_url = f"{gateway}/ipfs/{cid}"
            gateway_host = gateway.split("//")[1] if "//" in gateway else gateway
            attempt_start = time.monotonic()

            logger.info("Trying IPFS gateway %s for CID=%s (hash=%s)", gateway_host, cid[:12], hash)

            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=settings.cdn_connect_timeout,
                        read=settings.ipfs_timeout,
                        write=settings.ipfs_timeout,
                        pool=settings.ipfs_timeout,
                    ),
                    follow_redirects=True,
                ) as http:
                    response = await http.get(gateway_url)

                    if not response.is_success:
                        logger.warning(
                            "IPFS gateway %s returned %d for CID=%s",
                            gateway_host, response.status_code, cid[:12],
                        )
                        last_error = DownloadError(
                            f"IPFS gateway {gateway_host} returned {response.status_code}",
                            response.status_code,
                        )
                        continue

                    content = response.content
                    duration_ms = int((time.monotonic() - attempt_start) * 1000)
                    size_mb = len(content) / 1024 / 1024

                    logger.info(
                        "IPFS download complete: %.2f MB in %d ms from %s (CID=%s)",
                        size_mb, duration_ms, gateway_host, cid[:12],
                    )

                    return DownloadResult(
                        content=content,
                        format=detected_format,
                        hash=hash,
                        cdn_host=f"ipfs:{gateway_host}",
                        duration_ms=duration_ms,
                        size_bytes=len(content),
                    )

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                logger.warning("IPFS gateway %s failed: %s", gateway_host, exc)
                last_error = exc
                continue
            except Exception as exc:
                logger.warning("IPFS gateway %s error: %s", gateway_host, exc)
                last_error = exc
                continue

    raise DownloadError(
        f"All IPFS gateways failed for hash={hash} ({len(ipfs_cids)} CIDs tried): {last_error}",
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
