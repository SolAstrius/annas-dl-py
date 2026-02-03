"""Anna's Archive download microservice.

A FastAPI service that downloads books from Anna's Archive CDN,
caches them in S3, and returns presigned URLs.

Designed to run with Python 3.13 free-threaded mode for true parallelism.
"""

import logging
import time
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from .annas_client import AnnasClient, AnnasClientError
from .config import Settings, get_settings
from .downloader import DownloadError, download_book
from .s3 import S3Storage, build_key, build_meta_key, content_type_for_format
from .urn import parse_urn, to_urn, WrongResolverError, InvalidUrnError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Global state (initialized in lifespan)
_annas_client: AnnasClient | None = None
_s3_storage: S3Storage | None = None


@lru_cache
def get_cached_settings() -> Settings:
    """Get cached settings instance."""
    return get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources."""
    global _annas_client, _s3_storage

    settings = get_cached_settings()

    # Initialize Anna's Archive client
    _annas_client = AnnasClient.create(timeout=15.0)
    logger.info("Initialized Anna's Archive client")

    # Initialize S3 storage
    _s3_storage = S3Storage.create(settings)
    logger.info("Initialized S3 storage (bucket=%s)", settings.s3_bucket)

    yield

    # Cleanup
    if _annas_client:
        await _annas_client.close()
        logger.info("Closed Anna's Archive client")


app = FastAPI(
    title="Anna's Archive Download Service",
    description="Microservice for downloading books from Anna's Archive with CDN failover",
    version="0.1.0",
    lifespan=lifespan,
)


# Request/Response models


class DownloadRequest(BaseModel):
    """Request to download a book."""

    title: str = ""
    format: str = "pdf"


class DownloadResponse(BaseModel):
    """Response from download endpoint."""

    id: str  # URN: urn:anna:<hash>
    hash: str
    title: str
    format: str
    download_url: str
    size_bytes: int
    duration_ms: int
    cdn_host: str
    cached: bool


class BatchDownloadRequest(BaseModel):
    """Request to download multiple books."""

    books: list[dict]  # List of {hash, title, format}


class BatchDownloadResponse(BaseModel):
    """Response from batch download endpoint."""

    results: dict[str, DownloadResponse | dict]
    duration_ms: int
    successful: int
    failed: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


# Endpoints


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0")


@app.post("/book/{id:path}/download", response_model=DownloadResponse)
async def download_book_endpoint(
    id: str,
    request: DownloadRequest | None = None,
    x_annas_key: str | None = Header(None, alias="X-Annas-Key"),
) -> DownloadResponse:
    """Download a book from Anna's Archive.

    If the book is already cached in S3, returns the presigned URL immediately.
    Otherwise, downloads from Anna's Archive CDN with automatic failover.

    Args:
        id: URN (urn:anna:<hash>) or raw MD5 hash
        request: Optional request body with title and format hints
        x_annas_key: Optional API key via header (overrides env var)

    Returns:
        DownloadResponse with presigned URL and metadata
    """
    if _annas_client is None or _s3_storage is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Parse URN or raw hash
    try:
        parsed = parse_urn(id)
        hash = parsed.hash
        urn = to_urn(hash)
    except WrongResolverError as e:
        raise HTTPException(status_code=421, detail=str(e))  # 421 Misdirected Request
    except InvalidUrnError as e:
        raise HTTPException(status_code=400, detail=str(e))

    settings = get_cached_settings()

    # Get API key: header takes precedence over env
    secret_key = x_annas_key or settings.annas_secret_key
    if not secret_key:
        raise HTTPException(status_code=401, detail="Missing API key (X-Annas-Key header or ANNAS_DL_ANNAS_SECRET_KEY env)")

    title = request.title if request else ""
    format_hint = request.format if request else "pdf"

    # Build S3 key
    key = build_key(hash, format_hint)

    # Check cache first
    if _s3_storage.exists(key):
        logger.info("Cache hit for hash=%s", hash)
        filename = f"{title or hash}.{format_hint}" if title else f"{hash}.{format_hint}"
        url = _s3_storage.get_presigned_url(key, filename)

        return DownloadResponse(
            id=urn,
            hash=hash,
            title=title,
            format=format_hint,
            download_url=url,
            size_bytes=0,  # Could fetch from S3 metadata if needed
            duration_ms=0,
            cdn_host="s3-cache",
            cached=True,
        )

    logger.info("Cache miss for hash=%s, downloading from CDN", hash)
    start_time = time.monotonic()

    try:
        result = await download_book(_annas_client, settings, secret_key, hash, format_hint)
    except DownloadError as exc:
        logger.error("Download failed for hash=%s: %s", hash, exc)
        status_code = 502 if exc.last_status and exc.last_status >= 500 else 500
        raise HTTPException(status_code=status_code, detail=str(exc))
    except AnnasClientError as exc:
        logger.error("Anna's Archive API error for hash=%s: %s", hash, exc)
        raise HTTPException(status_code=502, detail=str(exc))

    # Upload to S3
    actual_key = build_key(hash, result.format)
    content_type = content_type_for_format(result.format)
    _s3_storage.upload(actual_key, result.content, content_type)

    # Store minimal metadata
    import json

    metadata = {
        "hash": hash,
        "title": title,
        "format": result.format,
        "size_bytes": result.size_bytes,
    }
    meta_key = build_meta_key(hash)
    _s3_storage.upload(meta_key, json.dumps(metadata).encode(), "application/json")

    # Generate presigned URL
    filename = f"{title or hash}.{result.format}"
    url = _s3_storage.get_presigned_url(actual_key, filename)

    total_duration_ms = int((time.monotonic() - start_time) * 1000)

    return DownloadResponse(
        id=urn,
        hash=hash,
        title=title,
        format=result.format,
        download_url=url,
        size_bytes=result.size_bytes,
        duration_ms=total_duration_ms,
        cdn_host=result.cdn_host,
        cached=False,
    )


@app.post("/books/download", response_model=BatchDownloadResponse)
async def download_books_batch(request: BatchDownloadRequest) -> BatchDownloadResponse:
    """Download multiple books in parallel.

    Leverages free-threaded Python 3.13 for true parallel downloads.

    Args:
        request: Batch download request with list of books

    Returns:
        BatchDownloadResponse with results for each book
    """
    import asyncio

    if _annas_client is None or _s3_storage is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    settings = get_cached_settings()
    start_time = time.monotonic()

    async def download_one(book: dict) -> tuple[str, DownloadResponse | dict]:
        hash = book.get("hash", "")
        if not hash:
            return "", {"error": "Missing hash"}

        title = book.get("title", "")
        format_hint = book.get("format", "pdf")

        try:
            # Reuse the endpoint logic
            request = DownloadRequest(title=title, format=format_hint)
            response = await download_book_endpoint(hash, request)
            return hash, response
        except HTTPException as exc:
            return hash, {"error": exc.detail}
        except Exception as exc:
            return hash, {"error": str(exc)}

    # Run all downloads concurrently (semaphore is in downloader)
    tasks = [download_one(book) for book in request.books]
    results_list = await asyncio.gather(*tasks)

    results = {}
    successful = 0
    failed = 0

    for hash, result in results_list:
        if not hash:
            failed += 1
            continue

        results[hash] = result
        if isinstance(result, DownloadResponse):
            successful += 1
        else:
            failed += 1

    total_duration_ms = int((time.monotonic() - start_time) * 1000)

    return BatchDownloadResponse(
        results=results,
        duration_ms=total_duration_ms,
        successful=successful,
        failed=failed,
    )


def main():
    """Run the service with uvicorn."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "annas_dl.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
