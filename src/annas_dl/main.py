"""Anna's Archive download microservice.

A FastAPI service that downloads books from Anna's Archive CDN,
caches them in S3, and returns presigned URLs.

Designed to run with Python 3.13 free-threaded mode for true parallelism.
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from .annas_client import (
    AnnasClient,
    AnnasClientError,
    BookMetadata,
    DDoSGuardError,
    InvalidKeyError,
    NoDownloadsLeftError,
    NotMemberError,
    RecordNotFoundError,
)
from .config import Settings, get_settings
from .db import Database
from .downloader import DownloadError, download_book, set_torrent_session
from .s3 import S3Storage, content_type_for_format
from .search import ContentType, SearchFilters, SortField
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
_db: Database | None = None


@lru_cache
def get_cached_settings() -> Settings:
    """Get cached settings instance."""
    return get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources."""
    global _annas_client, _s3_storage, _db

    settings = get_cached_settings()

    # Validate required settings for server mode
    if not settings.s3_bucket:
        raise RuntimeError("ANNAS_DL_S3_BUCKET is required for server mode")

    # Initialize Anna's Archive client with FlareSolverr for DDoS-Guard bypass
    _annas_client = AnnasClient.create(
        timeout=15.0,
        flaresolverr_url=settings.flaresolverr_url,
        secret_key=settings.annas_secret_key,
    )
    logger.info(
        "Initialized Anna's Archive client (flaresolverr=%s)",
        "enabled" if settings.flaresolverr_url else "disabled",
    )

    # Initialize S3 storage
    _s3_storage = S3Storage.create(settings)
    logger.info("Initialized S3 storage (bucket=%s)", settings.s3_bucket)

    # Initialize PostgreSQL (shared annas-mcp database, optional)
    if settings.database_url:
        try:
            _db = Database(settings.database_url)
            await _db.connect()
            logger.info("Connected to PostgreSQL")
        except Exception as exc:
            logger.warning("Failed to connect to PostgreSQL: %s", exc)
            _db = None

    # Initialize torrent session (persistent DHT)
    if settings.torrent_enabled:
        try:
            from .torrent import TorrentSession

            torrent_session = TorrentSession(settings)
            set_torrent_session(torrent_session)
            logger.info("Initialized torrent session for BitTorrent fallback")
        except ImportError:
            logger.warning("libtorrent not available — torrent fallback disabled")
        except Exception as exc:
            logger.warning("Failed to initialize torrent session: %s", exc)

    yield

    # Cleanup
    from .downloader import _torrent_session

    if _torrent_session is not None:
        _torrent_session.shutdown()
        set_torrent_session(None)
        logger.info("Shut down torrent session")

    if _db:
        await _db.close()

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


class QuotaInfo(BaseModel):
    """Account fast download quota information."""

    downloads_left: int
    downloads_per_day: int
    downloads_done_today: int


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
    quota: QuotaInfo | None = None  # Only present for non-cached downloads


class BatchDownloadRequest(BaseModel):
    """Request to download multiple books."""

    books: list[dict]  # List of {hash, title, format}


class BatchDownloadResponse(BaseModel):
    """Response from batch download endpoint."""

    results: dict[str, DownloadResponse | dict]
    duration_ms: int
    successful: int
    failed: int


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    status: str  # "ok" or "unavailable"


class HealthResponse(BaseModel):
    """Health check response."""

    status: str  # "ok" or "degraded"
    version: str
    components: dict[str, ComponentHealth] = {}


# =============================================================================
# Search Models
# =============================================================================


class SearchResultItem(BaseModel):
    """A single search result from Anna's Archive.

    Scraped from table view HTML at /search?display=table.
    """

    id: str  # URN: urn:anna:<hash>
    hash: str  # Raw MD5 hash
    title: str
    author: str | None = None
    publisher: str | None = None
    year: str | None = None
    language: str | None = None
    content_type: str | None = None  # book_fiction, book_nonfiction, magazine, etc.
    format: str | None = None  # pdf, epub, mobi, etc.
    size: str | None = None  # Human-readable: "1.2MB", "500KB"
    url: str | None = None  # Direct link to Anna's Archive page


class SearchResponseModel(BaseModel):
    """Paginated search results from Anna's Archive.

    Results are scraped from the table view which provides cleaner HTML
    structure compared to the card view.
    """

    total_on_page: int  # Number of results on this page
    page: int
    items: list[SearchResultItem]
    query: str


# =============================================================================
# RFC 2483 Resolution Service Models
# =============================================================================


class I2LResponse(BaseModel):
    """I2L: URN to URL resolution response."""

    urn: str
    url: str


class I2LsResponse(BaseModel):
    """I2Ls: URN to multiple URLs resolution response."""

    urn: str
    urls: list[str]


class I2CResponse(BaseModel):
    """I2C: URN to URC (Uniform Resource Characteristics) response.

    URC provides metadata about the resource without fetching it.
    """

    urn: str
    title: str | None = None
    format: str | None = None
    size_bytes: int | None = None
    content_type: str | None = None
    cached: bool = False
    # Additional metadata fields
    hash: str | None = None
    created_at: str | None = None


class I2NResponse(BaseModel):
    """I2N: URN to canonical URN resolution response."""

    input_urn: str
    canonical_urn: str


class BookInfoResponse(BaseModel):
    """Book metadata from Anna's Archive (without downloading).

    Fetched directly from Anna's Archive /db/aarecord_elasticsearch/ endpoint.
    """

    urn: str
    hash: str

    # Core fields
    title_best: str
    author_best: str
    publisher_best: str
    extension_best: str
    year_best: str

    # Additional values
    title_additional: list[str] = []
    author_additional: list[str] = []
    publisher_additional: list[str] = []

    # Language
    language_codes: list[str] = []

    # Size
    filesize_best: int = 0

    # Content info
    content_type_best: str = ""
    stripped_description_best: str = ""

    # Cover images
    cover_url_best: str = ""
    cover_url_additional: list[str] = []

    # Edition
    edition_varia_best: str = ""

    # Dates
    added_date_best: str = ""

    # Identifiers
    identifiers_unified: dict[str, list[str]] = {}

    # IPFS
    ipfs_infos: list[dict[str, str]] = []

    # Availability flags
    has_aa_downloads: int = 0
    has_torrent_paths: int = 0


class ErrorResponse(BaseModel):
    """RFC 2483 compliant error response.

    Error categories from RFC 2483:
    - malformed_uri: URI syntax is invalid (400)
    - wrong_resolver: Valid URN but wrong namespace for this resolver (421)
    - not_found: URI doesn't exist in any form (404)
    - gone: URI existed in the past but no longer available (410)
    - access_denied: Authentication/authorization failure (401/403)
    - quota_exceeded: Rate limit or quota exhausted (429)
    - upstream_error: Resolution service dependency failed (502)
    - unavailable: Service temporarily unavailable (503)
    """

    error: str  # Error category (e.g., "not_found", "wrong_resolver")
    detail: str  # Human-readable description
    urn: str | None = None  # The URN that caused the error, if applicable


# Error helper
def error_response(status_code: int, error: str, detail: str, urn: str | None = None) -> HTTPException:
    """Create an HTTPException with RFC 2483 compliant error body."""
    return HTTPException(
        status_code=status_code,
        detail=ErrorResponse(error=error, detail=detail, urn=urn).model_dump(exclude_none=True),
    )


# Endpoints


@app.get("/health", response_model=HealthResponse)
@app.get("/healthz", response_model=HealthResponse, include_in_schema=False)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns component-level status for S3 and PostgreSQL.
    Overall status is "ok" if all configured components are healthy,
    "degraded" if any optional component (db) is down.
    """
    components: dict[str, ComponentHealth] = {}

    # S3: required
    s3_ok = _s3_storage is not None
    components["s3"] = ComponentHealth(status="ok" if s3_ok else "unavailable")

    # PostgreSQL: optional
    if _db is not None:
        db_ok = await _db.ping()
        components["db"] = ComponentHealth(status="ok" if db_ok else "unavailable")

    degraded = any(c.status != "ok" for c in components.values())

    return HealthResponse(
        status="degraded" if degraded else "ok",
        version="0.1.0",
        components=components,
    )


# =============================================================================
# Search Endpoint
# =============================================================================


@app.get("/search", response_model=SearchResponseModel, tags=["Search"])
@app.get("/books/search", response_model=SearchResponseModel, tags=["Search"], deprecated=True)
async def search_books(
    q: str = Query(..., description="Search query"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    sort: str = Query("", description="Sort: '', 'newest', 'oldest', 'largest', 'smallest'"),
    content: list[str] = Query([], description="Content types: book_fiction, book_nonfiction, magazine, book_comic"),
    ext: list[str] = Query([], description="File formats: pdf, epub, mobi, djvu, etc."),
    lang: list[str] = Query([], description="Language codes: en, ru, zh, de, etc."),
) -> SearchResponseModel:
    """Search Anna's Archive for books.

    ## Query Examples

    | Query | Finds |
    |-------|-------|
    | `Hitchhiker's Guide` | Books with title containing these words |
    | `Douglas Adams` | Books by Douglas Adams |
    | `python programming` | Books about Python programming |

    ## Filters

    - **content**: Filter by content type (book_fiction, book_nonfiction, magazine, book_comic)
    - **ext**: Filter by file format (pdf, epub, mobi, djvu, azw3, fb2)
    - **lang**: Filter by language code (en, ru, zh, de, fr, es)

    ## Response

    Results are scraped from Anna's Archive table view which provides:
    - **id**: URN identifier (`urn:anna:<hash>`)
    - **hash**: Raw MD5 hash for use with download endpoints
    - **format**: File extension (lowercase)
    - **size**: Human-readable file size

    ## Notes

    - Uses FlareSolverr for DDoS-Guard bypass when configured
    - Automatic domain failover across Anna's Archive mirrors
    - Results limited to what fits on one page (~100 items)
    """
    if _annas_client is None:
        raise error_response(503, "unavailable", "Service not initialized")

    # Map sort string to enum
    sort_map = {
        "": SortField.RELEVANCE,
        "newest": SortField.NEWEST,
        "oldest": SortField.OLDEST,
        "largest": SortField.LARGEST,
        "smallest": SortField.SMALLEST,
    }
    sort_field = sort_map.get(sort.lower(), SortField.RELEVANCE)

    # Map content type strings to enums
    content_type_list = []
    for ct in content:
        try:
            content_type_list.append(ContentType(ct.lower()))
        except ValueError:
            pass  # Ignore invalid content types

    filters = SearchFilters(
        query=q,
        content_types=content_type_list,
        formats=[f.lower() for f in ext],
        languages=[l.lower() for l in lang],
        page=page,
        sort=sort_field,
    )

    try:
        results = await _annas_client.search(filters)
    except DDoSGuardError as exc:
        raise error_response(502, "upstream_error", f"DDoS-Guard bypass failed: {exc}")
    except AnnasClientError as exc:
        raise error_response(502, "upstream_error", str(exc))

    # Convert to response model
    items = [
        SearchResultItem(
            id=r.id,
            hash=r.hash,
            title=r.title,
            author=r.author,
            publisher=r.publisher,
            year=r.year,
            language=r.language,
            content_type=r.content_type.value if r.content_type else None,
            format=r.format,
            size=r.size,
            url=r.url,
        )
        for r in results
    ]

    return SearchResponseModel(
        total_on_page=len(items),
        page=page,
        items=items,
        query=q,
    )


@app.post("/download/{id:path}", response_model=DownloadResponse)
@app.post("/book/{id:path}/download", response_model=DownloadResponse, deprecated=True)
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
        raise error_response(503, "unavailable", "Service not initialized")

    # Parse URN or raw hash
    try:
        parsed = parse_urn(id)
        hash = parsed.hash
        urn = to_urn(hash)
    except WrongResolverError as e:
        raise error_response(421, "wrong_resolver", str(e), urn=id)
    except InvalidUrnError as e:
        raise error_response(400, "malformed_uri", str(e), urn=id)

    settings = get_cached_settings()

    # Get API key: header takes precedence over env
    secret_key = x_annas_key or settings.annas_secret_key
    if not secret_key:
        raise error_response(401, "access_denied", "Missing API key (X-Annas-Key header or ANNAS_DL_ANNAS_SECRET_KEY env)", urn=urn)

    title = request.title if request else ""
    format_hint = request.format if request else "pdf"

    # Build S3 key
    key = _s3_storage.book_key(hash, format_hint)

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
        # Map upstream status to appropriate RFC 2483 error category
        if exc.last_status == 404:
            raise error_response(404, "not_found", str(exc), urn=urn)
        elif exc.last_status and exc.last_status >= 500:
            raise error_response(502, "upstream_error", str(exc), urn=urn)
        else:
            raise error_response(500, "upstream_error", str(exc), urn=urn)
    except NoDownloadsLeftError as exc:
        logger.warning("Fast downloads exhausted for hash=%s: %s", hash, exc)
        raise error_response(429, "quota_exceeded", str(exc), urn=urn)
    except InvalidKeyError as exc:
        logger.error("Invalid API key for hash=%s: %s", hash, exc)
        raise error_response(401, "access_denied", str(exc), urn=urn)
    except NotMemberError as exc:
        logger.error("Not a member for hash=%s: %s", hash, exc)
        raise error_response(403, "access_denied", str(exc), urn=urn)
    except RecordNotFoundError as exc:
        logger.warning("Book not found in Anna's Archive for hash=%s: %s", hash, exc)
        raise error_response(404, "not_found", str(exc), urn=urn)
    except AnnasClientError as exc:
        logger.error("Anna's Archive API error for hash=%s: %s", hash, exc)
        raise error_response(502, "upstream_error", str(exc), urn=urn)

    # Upload to S3
    actual_key = _s3_storage.book_key(hash, result.format)
    content_type = content_type_for_format(result.format)
    _s3_storage.upload(actual_key, result.content, content_type)

    # Fetch and store full metadata from Anna's Archive
    metadata = {
        "hash": hash,
        "title": title,  # User-provided title (may differ from AA)
        "format": result.format,
        "size_bytes": result.size_bytes,
    }

    # Try to fetch rich metadata from Anna's Archive
    book_meta: BookMetadata | None = None
    try:
        book_meta = await _annas_client.fetch_metadata(hash)
        metadata["anna"] = asdict(book_meta)
        # Promote key fields to top level for MCP server compatibility (BookMetadata)
        if book_meta.title_best:
            metadata["title"] = book_meta.title_best
        if book_meta.author_best:
            metadata["authors"] = book_meta.author_best
        if book_meta.publisher_best:
            metadata["publisher"] = book_meta.publisher_best
        if book_meta.language_codes:
            metadata["language"] = ", ".join(book_meta.language_codes)
        if book_meta.filesize_best:
            b = book_meta.filesize_best
            metadata["size"] = (
                f"{b / 1024**3:.1f}GB" if b >= 1024**3
                else f"{b / 1024**2:.1f}MB" if b >= 1024**2
                else f"{b / 1024:.1f}KB" if b >= 1024
                else f"{b} bytes"
            )
        metadata["url"] = f"https://annas-archive.org/md5/{hash}"
        # Include JSON-LD for interoperability
        settings = get_cached_settings()
        base_url = settings.base_url or ""
        metadata["jsonld"] = book_meta.to_jsonld(urn, base_url)
    except Exception as exc:
        logger.warning("Failed to fetch metadata for hash=%s: %s", hash, exc)
        # Continue without rich metadata - download succeeded

    meta_key = _s3_storage.meta_key(hash)
    _s3_storage.upload(meta_key, json.dumps(metadata).encode(), "application/json")

    # Persist to PostgreSQL (shared annas-mcp database)
    if _db and book_meta:
        await _db.upsert_book(hash, book_meta)

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
        quota=QuotaInfo(
            downloads_left=result.downloads_left,
            downloads_per_day=result.downloads_per_day,
            downloads_done_today=result.downloads_done_today,
        ),
    )


# =============================================================================
# RFC 2483 Resolution Service Endpoints
# =============================================================================


def _parse_and_validate_urn(id: str) -> tuple[str, str]:
    """Parse URN and return (hash, canonical_urn). Raises HTTPException on error."""
    try:
        parsed = parse_urn(id)
        return parsed.hash, to_urn(parsed.hash)
    except WrongResolverError as e:
        raise error_response(421, "wrong_resolver", str(e), urn=id)
    except InvalidUrnError as e:
        raise error_response(400, "malformed_uri", str(e), urn=id)


async def _ensure_cached(hash: str, urn: str, format_hint: str = "pdf") -> str:
    """Ensure resource is cached, fetching from upstream if needed. Returns actual format."""
    if _annas_client is None or _s3_storage is None:
        raise error_response(503, "unavailable", "Service not initialized")

    # Check if already cached
    for fmt in [format_hint, "pdf", "epub", "djvu", "mobi"]:
        key = _s3_storage.book_key(hash, fmt)
        if _s3_storage.exists(key):
            return fmt

    # Not cached - fetch from upstream
    settings = get_cached_settings()
    secret_key = settings.annas_secret_key
    if not secret_key:
        raise error_response(401, "access_denied", "No API key configured for upstream fetch", urn=urn)

    try:
        result = await download_book(_annas_client, settings, secret_key, hash, format_hint)
    except RecordNotFoundError as exc:
        raise error_response(404, "not_found", str(exc), urn=urn)
    except NoDownloadsLeftError as exc:
        raise error_response(429, "quota_exceeded", str(exc), urn=urn)
    except (DownloadError, AnnasClientError) as exc:
        raise error_response(502, "upstream_error", str(exc), urn=urn)

    # Cache the result
    actual_key = _s3_storage.book_key(hash, result.format)
    _s3_storage.upload(actual_key, result.content, content_type_for_format(result.format))

    metadata = {"hash": hash, "format": result.format, "size_bytes": result.size_bytes}

    # Try to fetch rich metadata from Anna's Archive
    book_meta: BookMetadata | None = None
    try:
        book_meta = await _annas_client.fetch_metadata(hash)
        metadata["anna"] = asdict(book_meta)
        # Promote key fields to top level for MCP server compatibility (BookMetadata)
        if book_meta.title_best:
            metadata["title"] = book_meta.title_best
        if book_meta.author_best:
            metadata["authors"] = book_meta.author_best
        if book_meta.publisher_best:
            metadata["publisher"] = book_meta.publisher_best
        if book_meta.language_codes:
            metadata["language"] = ", ".join(book_meta.language_codes)
        if book_meta.filesize_best:
            b = book_meta.filesize_best
            metadata["size"] = (
                f"{b / 1024**3:.1f}GB" if b >= 1024**3
                else f"{b / 1024**2:.1f}MB" if b >= 1024**2
                else f"{b / 1024:.1f}KB" if b >= 1024
                else f"{b} bytes"
            )
        metadata["url"] = f"https://annas-archive.org/md5/{hash}"
        # Include JSON-LD for interoperability
        settings = get_cached_settings()
        base_url = settings.base_url or ""
        metadata["jsonld"] = book_meta.to_jsonld(f"urn:anna:{hash}", base_url)
    except Exception as exc:
        logger.warning("Failed to fetch metadata in _ensure_cached for hash=%s: %s", hash, exc)

    _s3_storage.upload(_s3_storage.meta_key(hash), json.dumps(metadata).encode(), "application/json")

    # Persist to PostgreSQL
    if _db and book_meta:
        await _db.upsert_book(hash, book_meta)

    return result.format


@app.get("/urn/{id:path}/urls", response_model=I2LsResponse)
async def resolve_i2ls(
    id: str,
    format: str = Query("pdf", description="Preferred format if fetch needed"),
) -> I2LsResponse:
    """I2Ls: Resolve URN to multiple URLs.

    RFC 2483 I2Ls operation. Returns all available URLs for the resource,
    which may include different formats. Fetches from upstream if not cached.
    """
    if _s3_storage is None:
        raise error_response(503, "unavailable", "Service not initialized")

    hash, urn = _parse_and_validate_urn(id)

    # Ensure at least one format is cached
    await _ensure_cached(hash, urn, format)

    urls = []
    # Find all cached formats
    for fmt in ["pdf", "epub", "djvu", "mobi", "azw3", "fb2", "cbr", "cbz"]:
        key = _s3_storage.book_key(hash, fmt)
        if _s3_storage.exists(key):
            url = _s3_storage.get_presigned_url(key, f"{hash}.{fmt}")
            urls.append(url)

    return I2LsResponse(urn=urn, urls=urls)


@app.get("/urn/{id:path}/resource")
async def resolve_i2r(
    id: str,
    format: str = Query("pdf", description="Preferred format"),
) -> StreamingResponse:
    """I2R: Resolve URN directly to resource bytes.

    RFC 2483 I2R operation. Streams the actual resource content.
    Fetches from upstream if not cached.
    """
    if _s3_storage is None:
        raise error_response(503, "unavailable", "Service not initialized")

    hash, urn = _parse_and_validate_urn(id)

    # Ensure resource is cached
    actual_format = await _ensure_cached(hash, urn, format)

    key = _s3_storage.book_key(hash, actual_format)
    content = _s3_storage.download(key)

    return StreamingResponse(
        iter([content]),
        media_type=content_type_for_format(actual_format),
        headers={
            "Content-Disposition": f'attachment; filename="{hash}.{actual_format}"',
            "X-URN": urn,
        },
    )


@app.get("/urn/{id:path}/metadata", response_model=I2CResponse)
async def resolve_i2c(
    id: str,
    format: str = Query("pdf", description="Preferred format if fetch needed"),
) -> I2CResponse:
    """I2C: Resolve URN to URC (Uniform Resource Characteristics).

    RFC 2483 I2C operation. Returns metadata about the resource.
    Fetches from upstream if not cached.
    """
    if _s3_storage is None:
        raise error_response(503, "unavailable", "Service not initialized")

    hash, urn = _parse_and_validate_urn(id)

    # Ensure resource is cached
    actual_format = await _ensure_cached(hash, urn, format)

    # Load metadata
    meta_key = _s3_storage.meta_key(hash)
    try:
        meta_bytes = _s3_storage.download(meta_key)
        metadata = json.loads(meta_bytes)
    except Exception:
        # Metadata file missing but resource exists
        metadata = {"hash": hash, "format": actual_format}

    return I2CResponse(
        urn=urn,
        hash=metadata.get("hash", hash),
        title=metadata.get("title"),
        format=metadata.get("format", actual_format),
        size_bytes=int(sb) if (sb := metadata.get("size_bytes")) is not None else None,
        content_type=content_type_for_format(metadata.get("format", actual_format)),
        cached=True,
    )


@app.get("/urn/{id:path}/canonical", response_model=I2NResponse)
async def resolve_i2n(id: str) -> I2NResponse:
    """I2N: Resolve to canonical URN.

    RFC 2483 I2N operation. Normalizes the input URN to its canonical form.
    This is useful for deduplication and comparison.
    """
    _, canonical = _parse_and_validate_urn(id)
    return I2NResponse(input_urn=id, canonical_urn=canonical)


@app.get("/urn/{id:path}/info", response_model=None)
async def get_book_info(
    id: str,
    format: str = Query("json", description="Response format: 'json' or 'jsonld'"),
) -> BookInfoResponse | JSONResponse:
    """Get book metadata from Anna's Archive without downloading.

    Fetches metadata directly from Anna's Archive /db/aarecord_elasticsearch/
    endpoint. Does NOT download the book file - only retrieves metadata.

    Args:
        id: URN (urn:anna:<hash>) or raw MD5 hash
        format: Response format - 'json' (default) or 'jsonld' (JSON-LD with schema.org/Dublin Core)

    This is useful for:
    - Checking if a book exists before downloading
    - Getting cover URLs, descriptions, identifiers
    - Building search indexes or catalogs
    - Linked data integration (with format=jsonld)
    """
    if _annas_client is None:
        raise error_response(503, "unavailable", "Service not initialized")

    hash, urn = _parse_and_validate_urn(id)

    # Try serving from S3 cache if rich metadata is present
    cached_anna: dict | None = None
    meta_key = _s3_storage.meta_key(hash) if _s3_storage is not None else ""
    if _s3_storage is not None:
        try:
            existing = json.loads(_s3_storage.download(meta_key))
            if "anna" in existing:
                cached_anna = existing["anna"]
        except Exception:
            existing = None
    else:
        existing = None

    if cached_anna is not None:
        # Serve from cache — reconstruct BookMetadata from the stored dict
        meta = BookMetadata(**cached_anna)
        logger.debug("Serving cached metadata for hash=%s", hash)
    else:
        # Fetch from upstream
        try:
            meta = await _annas_client.fetch_metadata(hash)
        except DDoSGuardError as exc:
            raise error_response(502, "upstream_error", f"DDoS-Guard bypass failed: {exc}", urn=urn)
        except RecordNotFoundError as exc:
            raise error_response(404, "not_found", str(exc), urn=urn)
        except AnnasClientError as exc:
            raise error_response(502, "upstream_error", str(exc), urn=urn)

        # Backfill S3 metadata with rich data
        if _s3_storage is not None:
            if existing is None:
                existing = {"hash": hash}
            existing["anna"] = asdict(meta)
            base_url = get_cached_settings().base_url or ""
            existing["jsonld"] = meta.to_jsonld(urn, base_url)
            if meta.title_best:
                existing["title"] = meta.title_best
            if meta.author_best:
                existing["authors"] = meta.author_best
            if meta.publisher_best:
                existing["publisher"] = meta.publisher_best
            if meta.language_codes:
                existing["language"] = ", ".join(meta.language_codes)
            if meta.filesize_best:
                b = meta.filesize_best
                existing["size"] = (
                    f"{b / 1024**3:.1f}GB" if b >= 1024**3
                    else f"{b / 1024**2:.1f}MB" if b >= 1024**2
                    else f"{b / 1024:.1f}KB" if b >= 1024
                    else f"{b} bytes"
                )
            existing["url"] = f"https://annas-archive.org/md5/{hash}"
            _s3_storage.upload(meta_key, json.dumps(existing).encode(), "application/json")

    # Return JSON-LD format if requested
    if format.lower() == "jsonld":
        settings = get_cached_settings()
        base_url = settings.base_url or ""
        jsonld = meta.to_jsonld(urn, base_url)
        return JSONResponse(
            content=jsonld,
            media_type="application/ld+json",
        )

    return BookInfoResponse(
        urn=urn,
        hash=hash,
        title_best=meta.title_best,
        author_best=meta.author_best,
        publisher_best=meta.publisher_best,
        extension_best=meta.extension_best,
        year_best=meta.year_best,
        title_additional=meta.title_additional,
        author_additional=meta.author_additional,
        publisher_additional=meta.publisher_additional,
        language_codes=meta.language_codes,
        filesize_best=meta.filesize_best,
        content_type_best=meta.content_type_best,
        stripped_description_best=meta.stripped_description_best,
        cover_url_best=meta.cover_url_best,
        cover_url_additional=meta.cover_url_additional,
        edition_varia_best=meta.edition_varia_best,
        added_date_best=meta.added_date_best,
        identifiers_unified=meta.identifiers_unified,
        ipfs_infos=meta.ipfs_infos,
        has_aa_downloads=meta.has_aa_downloads,
        has_torrent_paths=meta.has_torrent_paths,
    )


@app.get("/urn/{id:path}/cover")
async def resolve_cover(
    id: str,
    size: str = Query("L", description="Cover size: S, M, or L (only affects Open Library URLs)"),
) -> RedirectResponse:
    """Resolve URN to a book cover image.

    Returns a 302 redirect to the cover image (cached in S3 or fetched from upstream).
    Tries cover_url_best first, then cover_url_additional in order.
    """
    import httpx

    if _s3_storage is None:
        raise error_response(503, "unavailable", "Service not initialized")

    hash, urn = _parse_and_validate_urn(id)

    # Check S3 cache first
    for ext in ("jpg", "png", "webp"):
        cover_key = _s3_storage.cover_key(hash, ext)
        if _s3_storage.exists(cover_key):
            logger.info("Cover cache hit for hash=%s", hash)
            url = _s3_storage.get_presigned_url(cover_key)
            return RedirectResponse(url=url, status_code=302)

    # Load metadata to find cover URLs
    meta_key = _s3_storage.meta_key(hash)
    cover_urls: list[str] = []
    try:
        meta_bytes = _s3_storage.download(meta_key)
        metadata = json.loads(meta_bytes)
        anna = metadata.get("anna", {})
        if anna.get("cover_url_best"):
            cover_urls.append(anna["cover_url_best"])
        cover_urls.extend(anna.get("cover_url_additional") or [])
    except Exception:
        pass

    # If no cached metadata, try fetching from upstream
    if not cover_urls and _annas_client is not None:
        try:
            meta = await _annas_client.fetch_metadata(hash)
            if meta.cover_url_best:
                cover_urls.append(meta.cover_url_best)
            cover_urls.extend(meta.cover_url_additional)
        except Exception:
            pass

    if not cover_urls:
        raise error_response(404, "not_found", "No cover image available", urn=urn)

    # Apply size param to Open Library URLs
    size = size.upper()
    if size not in ("S", "M", "L"):
        size = "L"

    def _apply_ol_size(url: str) -> str:
        if "covers.openlibrary.org" in url:
            for s in ("-S.", "-M.", "-L."):
                if s in url:
                    return url.replace(s, f"-{size}.")
        return url

    # Known placeholder image hashes (e.g. libgen "BOOK COVER NOT AVAILABLE")
    _PLACEHOLDER_HASHES = {
        "6516a47fc69b0f3956f12e7efc984eb1",
    }

    def _is_placeholder(data: bytes) -> bool:
        import hashlib
        return hashlib.md5(data).hexdigest() in _PLACEHOLDER_HASHES

    # Try each URL until one works
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as http:
        for cover_url in cover_urls:
            cover_url = _apply_ol_size(cover_url)
            try:
                response = await http.get(cover_url)
                if response.status_code != 200:
                    continue
                content = response.content
                if len(content) < 100 or _is_placeholder(content):
                    continue

                # Determine extension from content type
                ct = response.headers.get("content-type", "")
                if "png" in ct:
                    ext = "png"
                elif "webp" in ct:
                    ext = "webp"
                else:
                    ext = "jpg"

                # Cache to S3
                cover_key = _s3_storage.cover_key(hash, ext)
                _s3_storage.upload(cover_key, content, ct or "image/jpeg")

                url = _s3_storage.get_presigned_url(cover_key)
                return RedirectResponse(url=url, status_code=302)
            except Exception as exc:
                logger.warning("Cover fetch failed for %s: %s", cover_url, exc)
                continue

    raise error_response(404, "not_found", "All cover sources failed", urn=urn)


@app.get("/urn/{id:path}", response_model=I2LResponse)
async def resolve_i2l(
    id: str,
    redirect: bool = Query(False, description="If true, return 302 redirect instead of JSON"),
    format: str = Query("pdf", description="Preferred format"),
) -> I2LResponse | RedirectResponse:
    """I2L: Resolve URN to a single URL.

    RFC 2483 I2L operation. Returns a URL where the resource can be accessed.
    If not cached, fetches from Anna's Archive automatically.

    Args:
        id: URN (urn:anna:<hash>) or raw MD5 hash
        redirect: If true, returns HTTP 302 redirect to the URL
        format: Preferred format (pdf, epub, etc.)
    """
    if _s3_storage is None:
        raise error_response(503, "unavailable", "Service not initialized")

    hash, urn = _parse_and_validate_urn(id)

    # Ensure resource is cached (fetches from upstream if needed)
    actual_format = await _ensure_cached(hash, urn, format)

    key = _s3_storage.book_key(hash, actual_format)
    url = _s3_storage.get_presigned_url(key, f"{hash}.{actual_format}")

    if redirect:
        return RedirectResponse(url=url, status_code=302)
    return I2LResponse(urn=urn, url=url)


@app.post("/download", response_model=BatchDownloadResponse)
@app.post("/books/download", response_model=BatchDownloadResponse, deprecated=True)
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
        raise error_response(503, "unavailable", "Service not initialized")

    start_time = time.monotonic()

    async def download_one(book: dict) -> tuple[str, DownloadResponse | dict]:
        hash = book.get("hash", "")
        if not hash:
            return "", {"error": "malformed_uri", "detail": "Missing hash"}

        title = book.get("title", "")
        format_hint = book.get("format", "pdf")

        try:
            # Reuse the endpoint logic
            request = DownloadRequest(title=title, format=format_hint)
            response = await download_book_endpoint(hash, request)
            return hash, response
        except HTTPException as exc:
            # exc.detail is already an ErrorResponse dict
            return hash, exc.detail if isinstance(exc.detail, dict) else {"error": "unknown", "detail": exc.detail}
        except Exception as exc:
            return hash, {"error": "unknown", "detail": str(exc)}

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
